from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from helper_functions.train_funcs import *
import plotly.express as px
import umap
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_and_plot_latent_embeddings_from_df(
    model, 
    df, 
    n=5, 
    classes=None, 
    columns=None, 
    device=None, 
    reduction_method='tsne', 
    plot_title="Latent Space Visualization"
):
    """
    Extracts latent space embeddings for n randomly selected objects from specified classes across different dataset variants 
    from a DataFrame and plots them with Plotly.
    
    Parameters:
    - model (nn.Module): Pre-loaded EnhancedSNN model.
    - df (pandas.DataFrame): DataFrame containing object paths across different processing variants.
    - n (int): Number of random objects to select from the filtered DataFrame. Default: 5.
    - classes (list of str or None): List of class names to filter the DataFrame (e.g., ['bathtub', 'chair']). 
                                     If None, uses all classes. Default: None.
    - columns (list of str or None): List of column names containing file paths for different variants. 
                                     If None, defaults to ["he_25_05/", "he_25_33/", "he_10_05/", "he_10_33/", "he_50_05/", "he_50_33/"].
    - device (torch.device or None): Device for computation ('cuda' or 'cpu'). If None, uses CUDA if available.
    - reduction_method (str): Method for dimensionality reduction ('tsne', 'pca', or 'umap'). Default: 'tsne'.
    - plot_title (str): Title for the plot. Default: "Latent Space Visualization".
    
    Returns:
    - embeddings (dict): Dictionary mapping object_id to its list of embeddings across variants.
    - reduced_embeddings (np.ndarray): 2D reduced embeddings for plotting.
    """
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Default columns if not provided
    if columns is None:
        columns = ["he_25_05/", "he_25_33/", "he_10_05/", "he_10_33/", "he_50_05/", "he_50_33/"]
    
    # Verify columns exist in DataFrame
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in DataFrame")
    
    # Filter DataFrame by classes if specified
    if classes is not None:
        if not isinstance(classes, list):
            raise ValueError("classes must be a list of strings or None")
        filtered_df = df[df['class'].isin(classes)].copy()
        if len(filtered_df) == 0:
            raise ValueError(f"No objects found for classes {classes}")
    else:
        filtered_df = df.copy()
    
    # Randomly sample n rows from the filtered DataFrame
    if n > len(filtered_df):
        raise ValueError(f"n ({n}) cannot be larger than the number of rows in the filtered DataFrame ({len(filtered_df)})")
    sampled_df = filtered_df.sample(n=n, random_state=42)  # random_state for reproducibility
    
    # Prepare data
    data_list = []
    labels = []
    object_ids = sampled_df['object_id'].tolist()
    for _, row in sampled_df.iterrows():
        for col in columns:
            # Replace .off with .pkl assuming half-edge structures are stored as pickles
            file_path = row[col].replace('.off', '.pkl')
            data_list.append((file_path, 0))  # Dummy label (0) since we only need embeddings
            labels.append(f"{row['object_id']}_{col[:-1]}")  # e.g., "bathtub_0107_he_25_05"
    
    dataset = MeshDataset(data_list)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Extract embeddings
    embeddings = {}
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            x_dict = {
                'v': data['v'].x,
                'e': data['e'].x,
                'f': data['f'].x
            }
            edge_index_dict = {
                ('v', 'to', 'v'): data['v', 'to', 'v'].edge_index,
                ('v', 'to', 'e'): data['v', 'to', 'e'].edge_index,
                ('v', 'to', 'f'): data['v', 'to', 'f'].edge_index,
                ('e', 'to', 'v'): data['e', 'to', 'v'].edge_index,
                ('e', 'to', 'f'): data['e', 'to', 'f'].edge_index,
                ('f', 'to', 'v'): data['f', 'to', 'v'].edge_index,
                ('f', 'to', 'e'): data['f', 'to', 'e'].edge_index,
            }
            
            # Pass through layers to get latent representation
            for layer_idx in range(model.num_layers):
                original_x = x_dict.copy() if layer_idx > 0 and model.residual_connections else None
                x_dict = model._forward_layer(data, layer_idx, x_dict, edge_index_dict, original_x)
            
            # Pool the features
            v_pooled = x_dict['v'].mean(dim=0)
            e_pooled = x_dict['e'].mean(dim=0)
            f_pooled = x_dict['f'].mean(dim=0)
            embedding = torch.cat([v_pooled, e_pooled, f_pooled], dim=0).cpu().numpy()
            

            # Store embedding by object_id
            obj_id = labels[i].split('_he_')[0]  # Extract object_id (e.g., "bathtub_0107")
            if obj_id not in embeddings:
                embeddings[obj_id] = []
            embeddings[obj_id].append(embedding)
    
    # Flatten embeddings for reduction
    all_embeddings = np.vstack([emb for emb_list in embeddings.values() for emb in emb_list])
    
    # Dimensionality reduction
    if reduction_method.lower() == 'tsne':
        reduced_embeddings = TSNE(n_components=2, random_state=42).fit_transform(all_embeddings)
    elif reduction_method.lower() == 'pca':
        reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(all_embeddings)
    elif reduction_method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
    else:
        raise ValueError("reduction_method must be 'tsne', 'pca', or 'umap'")
    
    # Prepare data for Plotly
    plot_data = {
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels,
        'object_id': [lbl.split('_he_')[0] for lbl in labels],
        'variant': [lbl.split('_he_')[1] for lbl in labels],
        'class': [sampled_df[sampled_df['object_id'] == lbl.split('_he_')[0]]['class'].iloc[0] for lbl in labels],
    }
    plot_df = pd.DataFrame(plot_data)
    
    # Create interactive scatter plot with Plotly
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color='object_id', 
        symbol='class',  
        hover_data=['label', 'class'],  # Show full label and class on hover
        title=f"{plot_title} (n={n} Random Objects, {reduction_method.upper()}{', Classes: ' + ', '.join(classes) if classes else ''})",
        labels={
            'x': f"{reduction_method.upper()} Component 1",
            'y': f"{reduction_method.upper()} Component 2",
            'variant': 'Processing Variant',
            'object_id': 'Object ID',
            'class': 'Class'
        },
        width=800,
        height=600
    )
    
    # Customize layout
    fig.update_traces(marker=dict(size=12, opacity=0.8),
                      selector=dict(mode='markers'))
    fig.update_layout(
        legend_title_text='Variants & Objects',
        showlegend=True,
        template='plotly_dark',  # Dark theme for a modern look
        title_x=0.5  # Center the title
    )
    
    # Show the plot
    fig.show()
    
    return embeddings, reduced_embeddings


def visualize_meshes_from_df(df, object_ids, titles=None, color='lightblue', size=(800, 400), root_path = None):
    """Visualizes multiple 3D meshes side by side from a DataFrame in a Jupyter Notebook.
    
    Args:
        df (pd.DataFrame): DataFrame containing a column 'path' with file paths to the 3D meshes.
        object_ids (list): List of object IDs in the DataFrame specifying which meshes to visualize.
        titles (list, optional): List of titles for each mesh. Defaults to None.
        color (str, optional): Color of the meshes. Defaults to 'lightblue'.
        size (tuple, optional): Figure size (width, height). Defaults to (800, 400).
    """
    if root_path == None:
        df['path'] = 'D:/ModelNet10/' + df['object_path']
    else:
        df['path'] = root_path + df['object_path']

    mesh_paths = df[df['object_id'].isin(object_ids)]['path'].tolist()
    num_meshes = len(mesh_paths)
    fig = make_subplots(rows=1, cols=num_meshes, subplot_titles=titles, specs=[[{'type': 'surface'}]*num_meshes])
    
    for i, mesh_path in enumerate(mesh_paths):
        mesh = trimesh.load_mesh(mesh_path)
        x, y, z = mesh.vertices.T
        i_faces, j_faces, k_faces = mesh.faces.T
        
        mesh_trace = go.Mesh3d(
            x=x, y=y, z=z,
            i=i_faces, j=j_faces, k=k_faces,
            color=color, opacity=0.7
        )
        fig.add_trace(mesh_trace, row=1, col=i+1)
    
    fig.update_layout(
        height=size[1], width=size[0]*num_meshes,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(aspectmode='data')
    )
    fig.show()


"""
def get_and_plot_latent_embeddings_from_df(model, df, n=5, columns=None, device=None, reduction_method='tsne', plot_title="Latent Space Visualization"):
    
    Extracts latent space embeddings for n randomly selected objects across different dataset variants from a DataFrame and plots them with Plotly.
    
    Parameters:
    - model (nn.Module): Pre-loaded EnhancedSNN model.
    - df (pandas.DataFrame): DataFrame containing object paths across different processing variants.
    - n (int): Number of random objects to select from the DataFrame. Default: 5.
    - columns (list of str or None): List of column names containing file paths for different variants. 
                                     If None, defaults to ["he_25_05/", "he_25_33/", "he_10_05/", "he_10_33/", "he_50_05/", "he_50_33/"].
    - device (torch.device or None): Device for computation ('cuda' or 'cpu'). If None, uses CUDA if available.
    - reduction_method (str): Method for dimensionality reduction ('tsne', 'pca', or 'umap'). Default: 'tsne'.
    - plot_title (str): Title for the plot. Default: "Latent Space Visualization".
    
    Returns:
    - embeddings (dict): Dictionary mapping object_id to its list of embeddings across variants.
    - reduced_embeddings (np.ndarray): 2D reduced embeddings for plotting.
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Default columns if not provided
    if columns is None:
        columns = ["he_25_05/", "he_25_33/", "he_10_05/", "he_10_33/", "he_50_05/", "he_50_33/"]
    
    # Verify columns exist in DataFrame
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in DataFrame")
    
    # Randomly sample n rows from the DataFrame
    if n > len(df):
        raise ValueError(f"n ({n}) cannot be larger than the number of rows in the DataFrame ({len(df)})")
    sampled_df = df.sample(n=n, random_state=42)  # random_state for reproducibility
    
    # Prepare data
    data_list = []
    labels = []
    object_ids = sampled_df['object_id'].tolist()
    for _, row in sampled_df.iterrows():
        for col in columns:
            # Replace .off with .pkl assuming half-edge structures are stored as pickles
            file_path = row[col].replace('.off', '.pkl')
            data_list.append((file_path, 0))  # Dummy label (0) since we only need embeddings
            labels.append(f"{row['object_id']}_{col[:-1]}")  # e.g., "bathtub_0107_he_25_05"
    
    dataset = MeshDataset(data_list)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Extract embeddings
    embeddings = {}
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            x_dict = {
                'v': data['v'].x,
                'e': data['e'].x,
                'f': data['f'].x
            }
            edge_index_dict = {
                ('v', 'to', 'v'): data['v', 'to', 'v'].edge_index,
                ('v', 'to', 'e'): data['v', 'to', 'e'].edge_index,
                ('v', 'to', 'f'): data['v', 'to', 'f'].edge_index,
                ('e', 'to', 'v'): data['e', 'to', 'v'].edge_index,
                ('e', 'to', 'f'): data['e', 'to', 'f'].edge_index,
                ('f', 'to', 'v'): data['f', 'to', 'v'].edge_index,
                ('f', 'to', 'e'): data['f', 'to', 'e'].edge_index,
            }
            
            # Pass through layers to get latent representation
            for layer_idx in range(model.num_layers):
                original_x = x_dict.copy() if layer_idx > 0 and model.residual_connections else None
                x_dict = model._forward_layer(data, layer_idx, x_dict, edge_index_dict, original_x)
            
            # Pool the features
            v_pooled = x_dict['v'].mean(dim=0)
            e_pooled = x_dict['e'].mean(dim=0)
            f_pooled = x_dict['f'].mean(dim=0)
            embedding = torch.cat([v_pooled, e_pooled, f_pooled], dim=0).cpu().numpy()
            
            # Store embedding by object_id
            obj_id = labels[i].split('_he_')[0]  # Extract object_id (e.g., "bathtub_0107")
            if obj_id not in embeddings:
                embeddings[obj_id] = []
            embeddings[obj_id].append(embedding)
    
    # Flatten embeddings for reduction
    all_embeddings = np.vstack([emb for emb_list in embeddings.values() for emb in emb_list])
    
    # Dimensionality reduction
    if reduction_method.lower() == 'tsne':
        reduced_embeddings = TSNE(n_components=2, random_state=42).fit_transform(all_embeddings)
    elif reduction_method.lower() == 'pca':
        reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(all_embeddings)
    elif reduction_method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
    else:
        raise ValueError("reduction_method must be 'tsne', 'pca', or 'umap'")
    
    # Prepare data for Plotly
    plot_data = {
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels,
        'object_id': [lbl.split('_he_')[0] for lbl in labels],
        'variant': [lbl.split('_he_')[1] for lbl in labels]
    }
    plot_df = pd.DataFrame(plot_data)
    
    # Create interactive scatter plot with Plotly
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color='variant',  # Color by variant (e.g., he_25_05, he_25_33)
        symbol='object_id',  # Different symbols for each object_id
        hover_data=['label'],  # Show full label on hover
        title=f"{plot_title} (n={n} Random Objects, {reduction_method.upper()})",
        labels={
            'x': f"{reduction_method.upper()} Component 1",
            'y': f"{reduction_method.upper()} Component 2",
            'variant': 'Processing Variant',
            'object_id': 'Object ID'
        },
        width=800,
        height=600
    )
    
    # Customize layout
    fig.update_traces(marker=dict(size=12, opacity=0.8),
                      selector=dict(mode='markers'))
    fig.update_layout(
        legend_title_text='Variants & Objects',
        showlegend=True,
        template='plotly_dark',  # Dark theme for a modern look
        title_x=0.5  # Center the title
    )
    
    # Show the plot
    fig.show()
    
    return embeddings, reduced_embeddings

def get_and_plot_latent_embeddings_from_df(model, df, columns=None, device=None, reduction_method='tsne', plot_title="Latent Space Visualization"):
    
    # Extracts latent space embeddings for objects across different dataset variants from a DataFrame and plots them.
    
    # Parameters:
    # - model (nn.Module): Pre-loaded EnhancedSNN model.
    # - df (pandas.DataFrame): DataFrame containing object paths across different processing variants.
    # - columns (list of str or None): List of column names containing file paths for different variants. 
    #                                  If None, defaults to ["he_25_05/", "he_25_33/", "he_10_05/", "he_10_33/", "he_50_05/", "he_50_33/"].
    # - device (torch.device or None): Device for computation ('cuda' or 'cpu'). If None, uses CUDA if available.
    # - reduction_method (str): Method for dimensionality reduction ('tsne' or 'pca'). Default: 'tsne'.
    # - plot_title (str): Title for the plot. Default: "Latent Space Visualization".
    
    # Returns:
    # - embeddings (dict): Dictionary mapping object_id to its list of embeddings across variants.
    # - reduced_embeddings (np.ndarray): 2D reduced embeddings for plotting.

    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Default columns if not provided
    if columns is None:
        columns = ["he_25_05/", "he_25_33/", "he_10_05/", "he_10_33/", "he_50_05/", "he_50_33/"]
    
    # Verify columns exist in DataFrame
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in DataFrame")
    
    # Prepare data
    data_list = []
    labels = []
    object_ids = df['object_id'].tolist()
    for _, row in df.iterrows():
        for col in columns:
            # Replace .off with .pkl assuming half-edge structures are stored as pickles
            file_path = row[col].replace('.off', '.pkl')
            data_list.append((file_path, 0))  # Dummy label (0) since we only need embeddings
            labels.append(f"{row['object_id']}_{col[:-1]}")  # e.g., "bathtub_0107_he_25_05"
    
    dataset = MeshDataset(data_list)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Extract embeddings
    embeddings = {}
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            x_dict = {
                'v': data['v'].x,
                'e': data['e'].x,
                'f': data['f'].x
            }
            edge_index_dict = {
                ('v', 'to', 'v'): data['v', 'to', 'v'].edge_index,
                ('v', 'to', 'e'): data['v', 'to', 'e'].edge_index,
                ('v', 'to', 'f'): data['v', 'to', 'f'].edge_index,
                ('e', 'to', 'v'): data['e', 'to', 'v'].edge_index,
                ('e', 'to', 'f'): data['e', 'to', 'f'].edge_index,
                ('f', 'to', 'v'): data['f', 'to', 'v'].edge_index,
                ('f', 'to', 'e'): data['f', 'to', 'e'].edge_index,
            }
            
            # Pass through layers to get latent representation
            for layer_idx in range(model.num_layers):
                original_x = x_dict.copy() if layer_idx > 0 and model.residual_connections else None
                x_dict = model._forward_layer(data, layer_idx, x_dict, edge_index_dict, original_x)
            
            # Pool the features
            v_pooled = x_dict['v'].mean(dim=0)
            e_pooled = x_dict['e'].mean(dim=0)
            f_pooled = x_dict['f'].mean(dim=0)
            embedding = torch.cat([v_pooled, e_pooled, f_pooled], dim=0).cpu().numpy()
            
            # Store embedding by object_id
            obj_id = labels[i].split('_he_')[0]  # Extract object_id (e.g., "bathtub_0107")
            if obj_id not in embeddings:
                embeddings[obj_id] = []
            embeddings[obj_id].append(embedding)
    
    # Flatten embeddings for reduction
    all_embeddings = np.vstack([emb for emb_list in embeddings.values() for emb in emb_list])
    
    # Dimensionality reduction
    if reduction_method.lower() == 'tsne':
        reduced_embeddings = TSNE(n_components=2, random_state=42).fit_transform(all_embeddings)
    elif reduction_method.lower() == 'pca':
        from sklearn.decomposition import PCA
        reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(all_embeddings)
    else:
        raise ValueError("reduction_method must be 'tsne' or 'pca'")
    
    # Plotting
    plt.figure(figsize=(12, 10))
    colors = plt.cm.get_cmap('tab10', len(columns))  # One color per variant
    marker_styles = ['o', 's', '^', 'D', 'v', '*']  # Different markers for each variant
    
    start_idx = 0
    for obj_id in embeddings.keys():
        num_variants = len(embeddings[obj_id])
        for i, (x, y) in enumerate(reduced_embeddings[start_idx:start_idx + num_variants]):
            variant = columns[i % len(columns)][:-1]  # Remove trailing "/"
            plt.scatter(x, y, 
                        c=[colors(i)], 
                        marker=marker_styles[i % len(marker_styles)], 
                        label=f"{obj_id}_{variant}" if i == 0 else None, 
                        s=100)
        start_idx += num_variants
    
    plt.legend(title="Object_Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(plot_title)
    plt.xlabel(f"{reduction_method.upper()} Component 1")
    plt.ylabel(f"{reduction_method.upper()} Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return embeddings, reduced_embeddings

def get_and_plot_latent_embeddings_from_df_2(model, df, columns=None, device=None, reduction_method='tsne', plot_title="Latent Space Visualization"):
    Extracts latent space embeddings for objects across different dataset variants from a DataFrame and plots them with Plotly.
    
    Parameters:
    - model (nn.Module): Pre-loaded EnhancedSNN model.
    - df (pandas.DataFrame): DataFrame containing object paths across different processing variants.
    - columns (list of str or None): List of column names containing file paths for different variants. 
                                     If None, defaults to ["he_25_05/", "he_25_33/", "he_10_05/", "he_10_33/", "he_50_05/", "he_50_33/"].
    - device (torch.device or None): Device for computation ('cuda' or 'cpu'). If None, uses CUDA if available.
    - reduction_method (str): Method for dimensionality reduction ('tsne' or 'pca'). Default: 'tsne'.
    - plot_title (str): Title for the plot. Default: "Latent Space Visualization".
    
    Returns:
    - embeddings (dict): Dictionary mapping object_id to its list of embeddings across variants.
    - reduced_embeddings (np.ndarray): 2D reduced embeddings for plotting.

    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Default columns if not provided
    if columns is None:
        columns = ["he_25_05/", "he_25_33/", "he_10_05/", "he_10_33/", "he_50_05/", "he_50_33/"]
    
    # Verify columns exist in DataFrame
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in DataFrame")
    
    # Prepare data
    data_list = []
    labels = []
    object_ids = df['object_id'].tolist()
    for _, row in df.iterrows():
        for col in columns:
            # Replace .off with .pkl assuming half-edge structures are stored as pickles
            file_path = row[col].replace('.off', '.pkl')
            data_list.append((file_path, 0))  # Dummy label (0) since we only need embeddings
            labels.append(f"{row['object_id']}_{col[:-1]}")  # e.g., "bathtub_0107_he_25_05"
    
    dataset = MeshDataset(data_list)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Extract embeddings
    embeddings = {}
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            x_dict = {
                'v': data['v'].x,
                'e': data['e'].x,
                'f': data['f'].x
            }
            edge_index_dict = {
                ('v', 'to', 'v'): data['v', 'to', 'v'].edge_index,
                ('v', 'to', 'e'): data['v', 'to', 'e'].edge_index,
                ('v', 'to', 'f'): data['v', 'to', 'f'].edge_index,
                ('e', 'to', 'v'): data['e', 'to', 'v'].edge_index,
                ('e', 'to', 'f'): data['e', 'to', 'f'].edge_index,
                ('f', 'to', 'v'): data['f', 'to', 'v'].edge_index,
                ('f', 'to', 'e'): data['f', 'to', 'e'].edge_index,
            }
            
            # Pass through layers to get latent representation
            for layer_idx in range(model.num_layers):
                original_x = x_dict.copy() if layer_idx > 0 and model.residual_connections else None
                x_dict = model._forward_layer(data, layer_idx, x_dict, edge_index_dict, original_x)
            
            # Pool the features
            v_pooled = x_dict['v'].mean(dim=0)
            e_pooled = x_dict['e'].mean(dim=0)
            f_pooled = x_dict['f'].mean(dim=0)
            embedding = torch.cat([v_pooled, e_pooled, f_pooled], dim=0).cpu().numpy()
            
            # Store embedding by object_id
            obj_id = labels[i].split('_he_')[0]  # Extract object_id (e.g., "bathtub_0107")
            if obj_id not in embeddings:
                embeddings[obj_id] = []
            embeddings[obj_id].append(embedding)
    
    # Flatten embeddings for reduction
    all_embeddings = np.vstack([emb for emb_list in embeddings.values() for emb in emb_list])
    
    # Dimensionality reduction
    if reduction_method.lower() == 'tsne':
        reduced_embeddings = TSNE(n_components=2, random_state=42).fit_transform(all_embeddings)
    elif reduction_method.lower() == 'pca':
        from sklearn.decomposition import PCA
        reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(all_embeddings)
    else:
        raise ValueError("reduction_method must be 'tsne' or 'pca'")
    
    # Prepare data for Plotly
    plot_data = {
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels,
        'object_id': [lbl.split('_he_')[0] for lbl in labels],
        'variant': [lbl.split('_he_')[1] for lbl in labels]
    }
    plot_df = pd.DataFrame(plot_data)
    
    # Create interactive scatter plot with Plotly
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color='variant',  # Color by variant (e.g., he_25_05, he_25_33)
        symbol='object_id',  # Different symbols for each object_id
        hover_data=['label'],  # Show full label on hover
        title=plot_title,
        labels={
            'x': f"{reduction_method.upper()} Component 1",
            'y': f"{reduction_method.upper()} Component 2",
            'variant': 'Processing Variant',
            'object_id': 'Object ID'
        },
        width=800,
        height=600
    )
    
    # Customize layout
    fig.update_traces(marker=dict(size=12, opacity=0.8),
                      selector=dict(mode='markers'))
    fig.update_layout(
        legend_title_text='Variants & Objects',
        showlegend=True,
        template='plotly_dark',  # Dark theme for a modern look (optional: 'plotly', 'plotly_white', etc.)
        title_x=0.5  # Center the title
    )
    
    # Show the plot
    fig.show()
    
    return embeddings, reduced_embeddings

def get_and_plot_latent_embeddings_from_df_n(model, df, n=5, columns=None, device=None, reduction_method='tsne', plot_title="Latent Space Visualization"):

    Extracts latent space embeddings for n randomly selected objects across different dataset variants from a DataFrame and plots them with Plotly.
    
    Parameters:
    - model (nn.Module): Pre-loaded EnhancedSNN model.
    - df (pandas.DataFrame): DataFrame containing object paths across different processing variants.
    - n (int): Number of random objects to select from the DataFrame. Default: 5.
    - columns (list of str or None): List of column names containing file paths for different variants. 
                                     If None, defaults to ["he_25_05/", "he_25_33/", "he_10_05/", "he_10_33/", "he_50_05/", "he_50_33/"].
    - device (torch.device or None): Device for computation ('cuda' or 'cpu'). If None, uses CUDA if available.
    - reduction_method (str): Method for dimensionality reduction ('tsne' or 'pca'). Default: 'tsne'.
    - plot_title (str): Title for the plot. Default: "Latent Space Visualization".
    
    Returns:
    - embeddings (dict): Dictionary mapping object_id to its list of embeddings across variants.
    - reduced_embeddings (np.ndarray): 2D reduced embeddings for plotting.

    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Default columns if not provided
    if columns is None:
        columns = ["he_25_05/", "he_25_33/", "he_10_05/", "he_10_33/", "he_50_05/", "he_50_33/"]
    
    # Verify columns exist in DataFrame
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in DataFrame")
    
    # Randomly sample n rows from the DataFrame
    if n > len(df):
        raise ValueError(f"n ({n}) cannot be larger than the number of rows in the DataFrame ({len(df)})")
    sampled_df = df.sample(n=n, random_state=42)  # random_state for reproducibility
    
    # Prepare data
    data_list = []
    labels = []
    object_ids = sampled_df['object_id'].tolist()
    for _, row in sampled_df.iterrows():
        for col in columns:
            # Replace .off with .pkl assuming half-edge structures are stored as pickles
            file_path = row[col].replace('.off', '.pkl')
            data_list.append((file_path, 0))  # Dummy label (0) since we only need embeddings
            labels.append(f"{row['object_id']}_{col[:-1]}")  # e.g., "bathtub_0107_he_25_05"
    
    dataset = MeshDataset(data_list)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Extract embeddings
    embeddings = {}
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            x_dict = {
                'v': data['v'].x,
                'e': data['e'].x,
                'f': data['f'].x
            }
            edge_index_dict = {
                ('v', 'to', 'v'): data['v', 'to', 'v'].edge_index,
                ('v', 'to', 'e'): data['v', 'to', 'e'].edge_index,
                ('v', 'to', 'f'): data['v', 'to', 'f'].edge_index,
                ('e', 'to', 'v'): data['e', 'to', 'v'].edge_index,
                ('e', 'to', 'f'): data['e', 'to', 'f'].edge_index,
                ('f', 'to', 'v'): data['f', 'to', 'v'].edge_index,
                ('f', 'to', 'e'): data['f', 'to', 'e'].edge_index,
            }
            
            # Pass through layers to get latent representation
            for layer_idx in range(model.num_layers):
                original_x = x_dict.copy() if layer_idx > 0 and model.residual_connections else None
                x_dict = model._forward_layer(data, layer_idx, x_dict, edge_index_dict, original_x)
            
            # Pool the features
            v_pooled = x_dict['v'].mean(dim=0)
            e_pooled = x_dict['e'].mean(dim=0)
            f_pooled = x_dict['f'].mean(dim=0)
            embedding = torch.cat([v_pooled, e_pooled, f_pooled], dim=0).cpu().numpy()
            
            # Store embedding by object_id
            obj_id = labels[i].split('_he_')[0]  # Extract object_id (e.g., "bathtub_0107")
            if obj_id not in embeddings:
                embeddings[obj_id] = []
            embeddings[obj_id].append(embedding)
    
    # Flatten embeddings for reduction
    all_embeddings = np.vstack([emb for emb_list in embeddings.values() for emb in emb_list])
    
    # Dimensionality reduction
    if reduction_method.lower() == 'tsne':
        reduced_embeddings = TSNE(n_components=2, random_state=42).fit_transform(all_embeddings)
    elif reduction_method.lower() == 'pca':
        from sklearn.decomposition import PCA
        reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(all_embeddings)
    else:
        raise ValueError("reduction_method must be 'tsne' or 'pca'")
    
    # Prepare data for Plotly
    plot_data = {
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels,
        'object_id': [lbl.split('_he_')[0] for lbl in labels],
        'variant': [lbl.split('_he_')[1] for lbl in labels]
    }
    plot_df = pd.DataFrame(plot_data)
    
    # Create interactive scatter plot with Plotly
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color='variant',  # Color by variant (e.g., he_25_05, he_25_33)
        symbol='object_id',  # Different symbols for each object_id
        hover_data=['label'],  # Show full label on hover
        title=f"{plot_title} (n={n} Random Objects)",
        labels={
            'x': f"{reduction_method.upper()} Component 1",
            'y': f"{reduction_method.upper()} Component 2",
            'variant': 'Processing Variant',
            'object_id': 'Object ID'
        },
        width=800,
        height=600
    )
    
    # Customize layout
    fig.update_traces(marker=dict(size=12, opacity=0.8),
                      selector=dict(mode='markers'))
    fig.update_layout(
        legend_title_text='Variants & Objects',
        showlegend=True,
        template='plotly_dark',  # Dark theme for a modern look
        title_x=0.5  # Center the title
    )
    
    # Show the plot
    fig.show()
    
    return embeddings, reduced_embeddings

 """