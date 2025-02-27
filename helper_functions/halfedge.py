from helper_functions.mesh_funcs import *
import pickle


def compute_edge_length(v1, v2):
    """Compute Euclidean distance between two vertices."""
    return np.linalg.norm(np.array(v1) - np.array(v2))

def compute_face_normal(v0, v1, v2):
    """Compute the normal of a triangle using the cross product."""
    edge1 = np.array(v1) - np.array(v0)
    edge2 = np.array(v2) - np.array(v0)
    normal = np.cross(edge1, edge2)
    norm = np.linalg.norm(normal)
    return tuple(normal / norm) if norm != 0 else (0, 0, 0)

def compute_face_area(v0, v1, v2):
    """Compute the area of a triangle using the cross product."""
    edge1 = np.array(v1) - np.array(v0)
    edge2 = np.array(v2) - np.array(v0)
    return 0.5 * np.linalg.norm(np.cross(edge1, edge2))

def mesh_to_half_edge(mesh):
    """
    Convert an Open3D mesh into a half-edge data structure.
    Now includes:
      - edge lengths in half-edges
      - face normals and areas in faces
      - edges explicitly stored with associated half-edges
    """
    vertices_array = np.asarray(mesh.vertices)
    normals_array = np.asarray(mesh.vertex_normals)
    vertices = {
        i: {
            "coords": tuple(vertices_array[i]),
            "normal": tuple(normals_array[i]),
            "half_edge": None  # Will be assigned later.
        }
        for i in range(len(vertices_array))
    }

    faces = {}
    half_edges = {}
    edges = {}  # Stores edges with references to half-edges
    edge_dict = {}  # Helper dictionary for twin lookups
    he_id = 0  # Unique ID for each half-edge

    triangles = np.asarray(mesh.triangles)
    for face_id, face in enumerate(triangles):
        v0, v1, v2 = face

        # Compute face normal and area
        face_normal = compute_face_normal(vertices[v0]["coords"], vertices[v1]["coords"], vertices[v2]["coords"])
        face_area = compute_face_area(vertices[v0]["coords"], vertices[v1]["coords"], vertices[v2]["coords"])

        # Create three half-edges in counter-clockwise order.
        he0 = he_id; he_id += 1
        he1 = he_id; he_id += 1
        he2 = he_id; he_id += 1

        half_edges[he0] = {
            "origin": v0, "face": face_id, "next": he1, "twin": None,
            "length": compute_edge_length(vertices[v0]["coords"], vertices[v1]["coords"])
        }
        half_edges[he1] = {
            "origin": v1, "face": face_id, "next": he2, "twin": None,
            "length": compute_edge_length(vertices[v1]["coords"], vertices[v2]["coords"])
        }
        half_edges[he2] = {
            "origin": v2, "face": face_id, "next": he0, "twin": None,
            "length": compute_edge_length(vertices[v2]["coords"], vertices[v0]["coords"])
        }

        faces[face_id] = {
            "vertices": (v0, v1, v2),
            "half_edge": he0,
            "normal": face_normal,
            "area": face_area
        }

        # Assign twins and store edges explicitly
        for current_he, origin, dest in [(he0, v0, v1), (he1, v1, v2), (he2, v2, v0)]:
            twin_key = (dest, origin)
            edge_key = tuple(sorted((origin, dest)))  # Ensure consistent ordering

            if twin_key in edge_dict:
                twin_he = edge_dict[twin_key]
                half_edges[current_he]["twin"] = twin_he
                half_edges[twin_he]["twin"] = current_he
            else:
                edge_dict[(origin, dest)] = current_he

            # Store edges explicitly
            if edge_key not in edges:
                edges[edge_key] = {
                    "half_edge": current_he,
                    "length": half_edges[current_he]["length"]
                }

        # Assign a half-edge to each vertex
        for vertex, he in [(v0, he0), (v1, he1), (v2, he2)]:
            if vertices[vertex]["half_edge"] is None:
                vertices[vertex]["half_edge"] = he

    return {
        "vertices": vertices,
        "half_edges": half_edges,
        "edges": edges,
        "faces": faces
    }


def df_to_half_edge_structure(df, indices, output_dir="Data/half_edge_structures/"):
    """
    Convert selected rows from a DataFrame into individual pickle files containing half-edge structures.
    
    Parameters:
        df : pandas.DataFrame
            DataFrame containing at least a 'path' column (paths to OFF files) and an 'object_path' column.
        indices : list-like
            Indices of rows to process.
        output_dir : str
            Directory to save the output pickle files.
    
    Returns:
        dict: Mapping of DataFrame indices to their processing results.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for idx in indices:
        try:
            # Retrieve file and object paths.
            file_path = df.loc[idx, 'path']
            # Use os.path.join to build the full output path.
            object_path = os.path.join(output_dir, df.loc[idx, 'object_path'])
            object_folder = os.path.dirname(object_path)
            object_name = os.path.splitext(os.path.basename(object_path))[0]
            os.makedirs(object_folder, exist_ok=True)
            
            # Load the mesh and convert it to a half-edge structure.
            mesh = load_off_mesh(file_path)
            half_edge_structure = mesh_to_half_edge(mesh)
            
            # Save the half-edge structure to a pickle file.
            output_file = os.path.join(object_folder, f"{object_name}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(half_edge_structure, f)
            
            
            results[idx] = {
                'status': 'success',
                'file_path': output_file,
                'object_name': object_name,
                'structure_size': {
                    'vertices': len(half_edge_structure['vertices']),
                    'half_edges': len(half_edge_structure['half_edges']),
                    'faces': len(half_edge_structure['faces'])
                }
            }
        except Exception as e:
            results[idx] = {
                'status': 'error',
                'error_message': str(e)
            }
    
    return results

def load_and_process_batch(df, batch_size=10, start_idx=0, output_dir="Data/half_edge_structures/"):
    """
    Process a batch of OFF files from the DataFrame into half-edge structures.
    
    Parameters:
        df : pandas.DataFrame
            DataFrame containing mesh file paths and object paths.
        batch_size : int
            Number of meshes to process in this batch.
        start_idx : int
            Starting index in the DataFrame.
    
    Returns:
        dict: Processing results for the batch.
    """
    end_idx = min(start_idx + batch_size, len(df))
    indices = range(start_idx, end_idx)
    return df_to_half_edge_structure(df, indices, output_dir=output_dir)

def process_all_meshes(df, batch_size=10, output_dir="Data/half_edge_structures/"):
    """
    Process all OFF meshes in the DataFrame using batches.
    
    Parameters:
        df : pandas.DataFrame
            DataFrame containing mesh file paths.
        batch_size : int
            Number of meshes to process in each batch.
    
    Returns:
        dict: Combined processing results for all batches.
    """
    all_results = {}
    for start_idx in range(0, len(df), batch_size):
        batch_results = load_and_process_batch(df, batch_size, start_idx, output_dir= output_dir)
        all_results.update(batch_results)
        
        # Print progress
        print(f"Processed {min(start_idx + batch_size, len(df))} out of {len(df)} meshes")
    
    return all_results



def get_vertex_info(mesh, vertex_id):
    """
    Retrieve neighboring vertices, connected edges, and associated faces for a given vertex.
    
    Parameters:
        mesh (dict): The half-edge data structure.
        vertex_id (int): The ID of the vertex.
        
    Returns:
        dict: Contains lists of neighbor vertices, connected edges, and associated face IDs.
    """
    vertices = mesh['vertices']
    half_edges = mesh['half_edges']
    
    if vertex_id not in vertices:
        raise ValueError(f"Vertex ID {vertex_id} not found in mesh.")
    
    # Collect all half-edges originating from the vertex
    edges = []
    neighbors = set()
    face_ids = set()
    
    for he_id, he in half_edges.items():
        if he['origin'] == vertex_id:
            edges.append(he_id)
            face_ids.add(he['face'])
            twin_he = he['twin']
            if twin_he is not None:
                neighbors.add(half_edges[twin_he]['origin'])
    
    return {
        'neighbors': list(neighbors),
        'edges': edges,
        'faces': list(face_ids)
    }

def get_edge_info(mesh, edge_id):
    """
    Retrieve adjacent edges, connected vertices, and associated faces for a given edge.
    
    Parameters:
        mesh (dict): The half-edge data structure.
        edge_id (int): The ID of the half-edge.
        
    Returns:
        dict: Contains lists of adjacent edges, connected vertices, and associated face IDs.
    """
    half_edges = mesh['half_edges']
    
    if edge_id not in half_edges:
        raise ValueError(f"Edge ID {edge_id} not found in mesh.")
    
    he = half_edges[edge_id]
    origin_v = he['origin']
    twin_he = he['twin']
    dest_v = half_edges[twin_he]['origin'] if twin_he is not None else None
    
    # Collect adjacent edges (edges connected to origin or destination vertices)
    adjacent_edges = []
    # From origin vertex
    for he_id, h in half_edges.items():
        if h['origin'] == origin_v and he_id != edge_id and he_id != twin_he:
            adjacent_edges.append(he_id)
    # From destination vertex (if exists)
    if dest_v is not None:
        for he_id, h in half_edges.items():
            if h['origin'] == dest_v and he_id != twin_he and he_id != edge_id:
                adjacent_edges.append(he_id)
    
    # Collect associated faces
    faces = [he['face']]
    if twin_he is not None:
        faces.append(half_edges[twin_he]['face'])
    
    return {
        "adjacent_edges": adjacent_edges,
        "vertices": [origin_v, dest_v] if dest_v is not None else [origin_v],
        "faces": faces
    }

def get_face_info(mesh, face_id):
    """
    Retrieve edges, vertices, and adjacent faces for a given face.
    
    Parameters:
        mesh (dict): The half-edge data structure.
        face_id (int): The ID of the face.
        
    Returns:
        dict: Contains lists of edges, vertices, and adjacent face IDs.
    """
    faces = mesh['faces']
    half_edges = mesh['half_edges']
    
    if face_id not in faces:
        raise ValueError(f"Face ID {face_id} not found in mesh.")
    
    start_he = faces[face_id]['half_edge']
    current_he = start_he
    face_edges = []
    vertices = []
    adjacent_faces = set()
    
    # Traverse all edges in the face
    while True:
        face_edges.append(current_he)
        vertices.append(half_edges[current_he]['origin'])
        twin_he = half_edges[current_he]['twin']
        if twin_he is not None:
            adj_face = half_edges[twin_he]['face']
            if adj_face != face_id:
                adjacent_faces.add(adj_face)
        current_he = half_edges[current_he]['next']
        if current_he == start_he:
            break
    
    return {
        'edges': face_edges,
        'vertices': vertices,
        'adjacent_faces': list(adjacent_faces)
    }

def load_half_edge_structure(file_path):
    """
    Load a half-edge structure from a pickle file given a DataFrame row.
    
    Parameters:
        df : pandas.DataFrame
            DataFrame containing a column 'object_path'.
        idx : int
            Row index of the DataFrame to load.
        base_dir : str
            Base directory where the pickle files are stored.
            
    Returns:
        dict: The loaded half-edge structure.
    """
    
    file_path = file_path.replace('.off', '.pkl')
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)