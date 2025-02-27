from helper_functions.SNN import *
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import json
from sklearn.metrics import confusion_matrix
 
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss implementation.

        Parameters:
        -----------
        alpha : float or list, optional
            Balancing factor for class imbalance. If None, no class weighting is applied.
        gamma : float
            Focusing parameter to down-weight easy examples and focus on hard ones.
        reduction : str
            Reduction mode: 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probabilities of the correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_factor = self.alpha[targets] if isinstance(self.alpha, torch.Tensor) else self.alpha
            focal_loss *= alpha_factor

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
def train_enhanced_model(
    csv_path="metadata_modelnet10.csv",
    root_path="half_edge_structures",
    batch_size=32,
    remove_indices=[4455, 4456, 4557],
    hidden_dim=64,
    lr=0.01,
    weight_decay=1e-5,
    num_epochs=40,
    save_path=None,
    confusion_interval=5,
    seed=42,
    # Model parameters
    num_layers=2,
    dropout_rate=0.2,
    use_batch_norm=True,
    use_weight_norm=False,
    residual_connections=True,
    conv_type='sage',
    # Training parameters
    optimizer_type='adam',  # 'adam', 'sgd', 'adamw'
    # LR scheduling parameters
    scheduler_type='cosine',  # 'cosine', 'onecycle', 'step', 'plateau'
    lr_min=1e-6,
    lr_max=0.01,
    warmup_epochs=5,
    # Regularization parameters
    label_smoothing=0.1,  # If > 0, use label smoothing
    gradient_clip_val=1.0,  # If > 0, clip gradients
    # EMA parameters
    use_ema=True,
    ema_decay=0.999,
    # Other parameters
    early_stopping=True,
    patience=15,
    log_file=None,
    focal_gamma=2.0
):
    """
    Enhanced training function with advanced LR scheduling and regularization.
    The test set is used for validation purposes.
    
    Parameters:
    - csv_path (str): Path to the CSV file with metadata.
    - root_path (str): Root directory for mesh files.
    - batch_size (int): Batch size for data loaders.
    - remove_indices (list): Indices to exclude from the dataset.
    - hidden_dim (int): Hidden dimension of the model.
    - lr (float): Initial learning rate.
    - weight_decay (float): Weight decay for regularization.
    - num_epochs (int): Number of training epochs.
    - save_path (str): Path to save the model; if None, uses default naming.
    - confusion_interval (int): Epoch interval to print confusion matrices.
    - seed (int): Random seed for reproducibility.
    - num_layers (int): Number of layers in the model.
    - dropout_rate (float): Dropout probability.
    - use_batch_norm (bool): Whether to use batch normalization.
    - use_weight_norm (bool): Whether to use weight normalization.
    - residual_connections (bool): Whether to use residual connections.
    - conv_type (str): Type of graph convolution to use.
    - optimizer_type (str): Type of optimizer ('adam', 'sgd', 'adamw').
    - scheduler_type (str): Type of LR scheduler ('cosine', 'onecycle', 'step', 'plateau').
    - lr_min (float): Minimum learning rate for schedulers.
    - lr_max (float): Maximum learning rate for OneCycleLR.
    - warmup_epochs (int): Number of epochs to warm up the learning rate.
    - label_smoothing (float): Label smoothing factor; if > 0, applies smoothing.
    - gradient_clip_val (float): Maximum gradient norm for clipping; if > 0, clips gradients.
    - use_ema (bool): Whether to use Exponential Moving Average.
    - ema_decay (float): Decay rate for EMA.
    - early_stopping (bool): Whether to use early stopping.
    - patience (int): Number of epochs to wait for improvement before stopping.
    - log_file (str): File to log training metrics; if None, no logging.
    - focal_gamma (float): Gamma parameter for Focal Loss if label_smoothing <= 0.
    
    Returns:
    - dict: Training history with losses and accuracies.
    """
    # Capture all input parameters at the start
    params = locals().copy()
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Fixed parameters necessary for model reconstruction
    num_classes = 10
    features = [4, 2, 5]  # Feature dimensions for vertices, edges, and faces
    
    # Add internally defined parameters to params for JSON saving
    params['num_classes'] = num_classes
    params['features'] = features
    
    # Load and split data
    df = pd.read_csv(csv_path)
    df['path'] = 'Data/' + root_path + df['object_path']
    
    # Create train/test splits
    df = df.drop(remove_indices)
    class_to_int = {cls: idx for idx, cls in enumerate(df['class'].unique())}
    
    train_indices = df[df['path'].str.contains('train')].index
    test_indices = df[df['path'].str.contains('test')].index
    
    data_list_train = [(df.loc[i, 'path'], class_to_int[df.loc[i, 'class']]) for i in train_indices]
    data_list_test = [(df.loc[i, 'path'], class_to_int[df.loc[i, 'class']]) for i in test_indices]
    
    train_dataset = MeshDataset_WOCORDS(data_list_train)
    test_dataset = MeshDataset_WOCORDS(data_list_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define save paths
    if save_path is None:
        base_name = f"Saved/Enhanced_SNNWOC_hd{hidden_dim}_lr{lr}_wd{weight_decay}_epochs{num_epochs}_seed{seed}_bs{batch_size}_drop{dropout_rate}_bn{int(use_batch_norm)}"
        final_save_path = base_name + "_final.pt"
        best_save_path = base_name + "_best.pt" 
    else:
        final_save_path = save_path
        best_save_path = save_path.replace(".pt", "_best.pt")
    
    # Initialize enhanced model
    model = EnhancedSNN(
        hidden_dim=hidden_dim, 
        num_classes=num_classes, 
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        use_weight_norm=use_weight_norm,
        residual_connections=residual_connections,
        conv_type=conv_type,
        features=features
    ).to(device)
    
    # Initialize EMA model if requested
    if use_ema:
        ema_model = EnhancedSNN(
            hidden_dim=hidden_dim, 
            num_classes=num_classes, 
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_weight_norm=use_weight_norm,
            residual_connections=residual_connections,
            conv_type=conv_type,
            features=features
        ).to(device)
        
        # Initialize EMA model with the same weights
        ema_model.load_state_dict(model.state_dict())
        for param in ema_model.parameters():
            param.requires_grad = False
    
    # Initialize optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")
    
    # Initialize loss function (with label smoothing if requested)
    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = FocalLoss(gamma=focal_gamma)
    
    # Initialize learning rate scheduler
    if scheduler_type.lower() == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min)
    elif scheduler_type.lower() == 'onecycle':
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(optimizer, max_lr=lr_max, 
                             steps_per_epoch=steps_per_epoch, 
                             epochs=num_epochs,
                             pct_start=warmup_epochs/num_epochs)
    elif scheduler_type.lower() == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type.lower() == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=lr_min)
    else:
        scheduler = None
    
    # Tracking variables
    best_val_acc = 0.0
    epochs_without_improvement = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Initialize log file
    if log_file:
        log_file = 'logs/' + log_file
        with open(log_file, 'w') as f:
            f.write("epoch,lr,train_loss,train_acc,val_loss,val_acc\n")
    
    # Update EMA model
    def update_ema_model(model, ema_model, decay):
        with torch.no_grad():
            for param_ema, param_model in zip(ema_model.parameters(), model.parameters()):
                param_ema.copy_(decay * param_ema + (1 - decay) * param_model)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Current LR: {current_lr:.6f}")
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Regular forward pass
            out = model(batch)
            target = batch.y.squeeze()
            loss = criterion(out, target)
            
            # Track predictions
            preds = out.argmax(dim=1).cpu().numpy()
            train_targets.extend(target.cpu().numpy())
            train_preds.extend(preds)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if requested
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            
            optimizer.step()
            
            # Update EMA model if requested
            if use_ema:
                update_ema_model(model, ema_model, ema_decay)
            
            # Step scheduler if using OneCycleLR (which updates per batch)
            if scheduler_type.lower() == 'onecycle':
                scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        # Validation phase (using test set)
        model.eval()
        if use_ema:
            ema_model.eval()
        
        val_loss = 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                target = batch.y.squeeze()
                
                # Use EMA model for inference if requested
                if use_ema:
                    out = ema_model(batch)
                else:
                    out = model(batch)
                
                loss = criterion(out, target)
                val_loss += loss.item()
                
                preds = out.argmax(dim=1).cpu().numpy()
                val_targets.extend(target.cpu().numpy())
                val_preds.extend(preds)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if scheduler_type.lower() == 'plateau':
                scheduler.step(avg_val_loss)
            elif scheduler_type.lower() != 'onecycle':  # OneCycleLR already stepped per batch
                scheduler.step()
        
        # Save best model (based on test set accuracy used as validation)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if use_ema:
                torch.save(ema_model.state_dict(), best_save_path)
            else:
                torch.save(model.state_dict(), best_save_path)
            # Save parameters to JSON
            json_path = best_save_path.replace(".pt", ".json")
            with open(json_path, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"New best model saved to {best_save_path} with val acc: {val_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if early_stopping and epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
        # Log metrics
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{current_lr:.6f},{avg_train_loss:.4f},{train_acc:.4f},"
                        f"{avg_val_loss:.4f},{val_acc:.4f}\n")
        
        # Plot confusion matrices at specified intervals
        if (epoch + 1) % confusion_interval == 0 or epoch == num_epochs - 1:
            cm_val = confusion_matrix(val_targets, val_preds)
            print(f"Validation Confusion Matrix (Epoch {epoch+1}):")
            print(cm_val)
    
    # Save final model
    if use_ema:
        torch.save(ema_model.state_dict(), final_save_path)
    else:
        torch.save(model.state_dict(), final_save_path)
    # Save parameters to JSON
    json_path = final_save_path.replace(".pt", ".json")
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Final model saved to {final_save_path}")
    
    # Print best accuracies
    print(f"Best validation accuracy (test set): {best_val_acc:.4f}")
    
    # Return training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    
    return history

def evaluate_model(
    model_path,
    csv_path="metadata_modelnet10.csv",
    root_path="half_edge_structures",
    batch_size=32,
    remove_indices=[4455, 4456, 4557],
    device=None,
    verbose=True,
    return_predictions=False
):
    """
    Evaluates a saved model on the test set using parameters from the corresponding JSON file.
    
    Parameters:
    - model_path (str): Path to the saved model state dictionary (e.g., 'model.pt').
    - csv_path (str): Path to the CSV file containing metadata (default: "metadata_modelnet10.csv").
    - root_path (str): Root directory for the mesh files (default: "half_edge_structures").
    - batch_size (int): Batch size for the test data loader (default: 32).
    - remove_indices (list): List of indices to remove from the dataset (default: [4455, 4456, 4557]).
    - device (torch.device or None): Device for evaluation ('cuda' or 'cpu'). If None, uses CUDA if available.
    - verbose (bool): If True, prints evaluation metrics and confusion matrix (default: True).
    - return_predictions (bool): If True, returns predictions and true labels (default: False).
    
    Returns:
    - If return_predictions is False: tuple (test_loss, test_acc)
    - If return_predictions is True: tuple (test_loss, test_acc, all_preds, all_targets)
    """
    # Load parameters from JSON file
    json_path = model_path.replace(".pt", ".json")
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    # Extract model configuration parameters from JSON
    hidden_dim = params['hidden_dim']
    num_layers = params['num_layers']
    dropout_rate = params['dropout_rate']
    use_batch_norm = params['use_batch_norm']
    use_weight_norm = params['use_weight_norm']
    residual_connections = params['residual_connections']
    conv_type = params['conv_type']
    features = params['features']  # List of feature dimensions, e.g., [7, 2, 5]
    num_classes = params['num_classes']  # Number of classes, e.g., 10
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CSV and prepare test data
    df = pd.read_csv(csv_path)
    df['path'] = 'Data/' + root_path + df['object_path']
    df = df.drop(remove_indices)
    class_to_int = {cls: idx for idx, cls in enumerate(df['class'].unique())}
    test_indices = df[df['path'].str.contains('test')].index
    data_list_test = [(df.loc[i, 'path'], class_to_int[df.loc[i, 'class']]) for i in test_indices]
    test_dataset = MeshDataset_WOCORDS(data_list_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with parameters from JSON
    model = EnhancedSNN(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        use_weight_norm=use_weight_norm,
        residual_connections=residual_connections,
        conv_type=conv_type,
        features=features
    ).to(device)
    
    # Load saved state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluation loop
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            target = batch.y.squeeze()
            loss = criterion(out, target)
            test_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            if return_predictions:
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
    
    # Compute average loss and accuracy
    avg_test_loss = test_loss / len(test_loader)
    test_acc = correct / total
    
    # Print results if verbose
    if verbose:
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        if return_predictions:
            cm = confusion_matrix(all_targets, all_preds)
            print("Confusion Matrix:")
            print(cm)
    
    # Return results
    if return_predictions:
        return avg_test_loss, test_acc, all_preds, all_targets
    else:
        return avg_test_loss, test_acc

def train_multiple_datasets(
    csv_path="metadata_modelnet10.csv",
    root_paths=["half_edge_structures"],  # Changed to a list of paths
    batch_size=32,
    remove_indices=[4455, 4456, 4557],
    hidden_dim=64,
    lr=0.01,
    weight_decay=1e-5,
    num_epochs=40,
    save_path=None,
    confusion_interval=5,
    seed=42,
    # Model parameters
    num_layers=2,
    dropout_rate=0.2,
    use_batch_norm=True,
    use_weight_norm=False,
    residual_connections=True,
    conv_type='sage',
    # Training parameters
    optimizer_type='adam',  # 'adam', 'sgd', 'adamw'
    # LR scheduling parameters
    scheduler_type='cosine',  # 'cosine', 'onecycle', 'step', 'plateau'
    lr_min=1e-6,
    lr_max=0.01,
    warmup_epochs=5,
    # Regularization parameters
    label_smoothing=0.1,  # If > 0, use label smoothing
    gradient_clip_val=1.0,  # If > 0, clip gradients
    # EMA parameters
    use_ema=True,
    ema_decay=0.999,
    # Other parameters
    early_stopping=True,
    patience=15,
    log_file=None,
    focal_gamma=2.0
):
    """
    Enhanced training function that supports multiple preprocessed datasets.
    The test set is used for validation purposes.
    
    Parameters:
    - csv_path (str): Path to the CSV file with metadata.
    - root_paths (list): List of root directories for mesh files.
    - batch_size (int): Batch size for data loaders.
    - remove_indices (list): Indices to exclude from the dataset.
    - hidden_dim (int): Hidden dimension of the model.
    - lr (float): Initial learning rate.
    - weight_decay (float): Weight decay for regularization.
    - num_epochs (int): Number of training epochs.
    - save_path (str): Path to save the model; if None, uses default naming.
    - confusion_interval (int): Epoch interval to print confusion matrices.
    - seed (int): Random seed for reproducibility.
    - num_layers (int): Number of layers in the model.
    - dropout_rate (float): Dropout probability.
    - use_batch_norm (bool): Whether to use batch normalization.
    - use_weight_norm (bool): Whether to use weight normalization.
    - residual_connections (bool): Whether to use residual connections.
    - conv_type (str): Type of graph convolution to use.
    - optimizer_type (str): Type of optimizer ('adam', 'sgd', 'adamw').
    - scheduler_type (str): Type of LR scheduler ('cosine', 'onecycle', 'step', 'plateau').
    - lr_min (float): Minimum learning rate for schedulers.
    - lr_max (float): Maximum learning rate for OneCycleLR.
    - warmup_epochs (int): Number of epochs to warm up the learning rate.
    - label_smoothing (float): Label smoothing factor; if > 0, applies smoothing.
    - gradient_clip_val (float): Maximum gradient norm for clipping; if > 0, clips gradients.
    - use_ema (bool): Whether to use Exponential Moving Average.
    - ema_decay (float): Decay rate for EMA.
    - early_stopping (bool): Whether to use early stopping.
    - patience (int): Number of epochs to wait for improvement before stopping.
    - log_file (str): File to log training metrics; if None, no logging.
    - focal_gamma (float): Gamma parameter for Focal Loss if label_smoothing <= 0.
    
    Returns:
    - dict: Training history with losses and accuracies.
    """
    # Capture all input parameters at the start
    params = locals().copy()
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Fixed parameters necessary for model reconstruction
    num_classes = 10
    features = [4, 2, 5]  # Feature dimensions for vertices, edges, and faces
    
    # Add internally defined parameters to params for JSON saving
    params['num_classes'] = num_classes
    params['features'] = features
    
    # Load original data to determine class mapping
    df = pd.read_csv(csv_path)
        # Create train/test splits
    df = df.drop(remove_indices)

    # Create a global class mapping to ensure consistency across all root paths
    # This is created once from the original dataframe

    class_to_int = {'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'night': 6, 'sofa': 7, 'table': 8, 'toilet': 9}

    # Create combined dataset from multiple root paths
    data_list_train = []
    data_list_test = []

    # Process each root path
    for root_path in root_paths:
        # Create a copy of the dataframe for this root path
        df_root = df.copy()
        df_root['path'] = 'Data/' + root_path + df_root['object_path']
        
        # Identify train and test samples
        train_indices = df_root[df_root['path'].str.contains('train')].index
        test_indices = df_root[df_root['path'].str.contains('test')].index
        
        # Add data to combined lists using the global class mapping
        data_list_train.extend([(df_root.loc[i, 'path'], class_to_int[df_root.loc[i, 'class']]) for i in train_indices])
        data_list_test.extend([(df_root.loc[i, 'path'], class_to_int[df_root.loc[i, 'class']]) for i in test_indices])

    train_dataset = MeshDataset_WOCORDS(data_list_train)
    test_dataset = MeshDataset_WOCORDS(data_list_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define save paths
    if save_path is None:
        base_name = f"Saved/Augmented_SNNWOC_hd{hidden_dim}_lr{lr}_wd{weight_decay}_epochs{num_epochs}_seed{seed}_bs{batch_size}_drop{dropout_rate}_bn{int(use_batch_norm)}"
        final_save_path = base_name + "_final.pt"
        best_save_path = base_name + "_best.pt" 
    else:
        final_save_path = save_path
        best_save_path = save_path.replace(".pt", "_best.pt")
    
    # Initialize enhanced model
    model = EnhancedSNN(
        hidden_dim=hidden_dim, 
        num_classes=num_classes, 
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        use_weight_norm=use_weight_norm,
        residual_connections=residual_connections,
        conv_type=conv_type,
        features=features
    ).to(device)
    
    # Initialize EMA model if requested
    if use_ema:
        ema_model = EnhancedSNN(
            hidden_dim=hidden_dim, 
            num_classes=num_classes, 
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_weight_norm=use_weight_norm,
            residual_connections=residual_connections,
            conv_type=conv_type,
            features=features
        ).to(device)
        
        # Initialize EMA model with the same weights
        ema_model.load_state_dict(model.state_dict())
        for param in ema_model.parameters():
            param.requires_grad = False
    
    # Initialize optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")
    
    # Initialize loss function (with label smoothing if requested)
    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = FocalLoss(gamma=focal_gamma)
    
    # Initialize learning rate scheduler
    if scheduler_type.lower() == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min)
    elif scheduler_type.lower() == 'onecycle':
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(optimizer, max_lr=lr_max, 
                             steps_per_epoch=steps_per_epoch, 
                             epochs=num_epochs,
                             pct_start=warmup_epochs/num_epochs)
    elif scheduler_type.lower() == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type.lower() == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=lr_min)
    else:
        scheduler = None
    
    # Tracking variables
    best_val_acc = 0.0
    epochs_without_improvement = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Initialize log file
    if log_file:
        log_file = 'logs/' + 'Augmented_' + log_file
        with open(log_file, 'w') as f:
            f.write("epoch,lr,train_loss,train_acc,val_loss,val_acc\n")
    
    # Update EMA model
    def update_ema_model(model, ema_model, decay):
        with torch.no_grad():
            for param_ema, param_model in zip(ema_model.parameters(), model.parameters()):
                param_ema.copy_(decay * param_ema + (1 - decay) * param_model)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Current LR: {current_lr:.6f}")
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Regular forward pass
            out = model(batch)
            target = batch.y.squeeze()
            loss = criterion(out, target)
            
            # Track predictions
            preds = out.argmax(dim=1).cpu().numpy()
            train_targets.extend(target.cpu().numpy())
            train_preds.extend(preds)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if requested
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            
            optimizer.step()
            
            # Update EMA model if requested
            if use_ema:
                update_ema_model(model, ema_model, ema_decay)
            
            # Step scheduler if using OneCycleLR (which updates per batch)
            if scheduler_type.lower() == 'onecycle':
                scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        # Validation phase (using test set)
        model.eval()
        if use_ema:
            ema_model.eval()
        
        val_loss = 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                target = batch.y.squeeze()
                
                # Use EMA model for inference if requested
                if use_ema:
                    out = ema_model(batch)
                else:
                    out = model(batch)
                
                loss = criterion(out, target)
                val_loss += loss.item()
                
                preds = out.argmax(dim=1).cpu().numpy()
                val_targets.extend(target.cpu().numpy())
                val_preds.extend(preds)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if scheduler_type.lower() == 'plateau':
                scheduler.step(avg_val_loss)
            elif scheduler_type.lower() != 'onecycle':  # OneCycleLR already stepped per batch
                scheduler.step()
        
        # Save best model (based on test set accuracy used as validation)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if use_ema:
                torch.save(ema_model.state_dict(), best_save_path)
            else:
                torch.save(model.state_dict(), best_save_path)
            # Save parameters to JSON
            json_path = best_save_path.replace(".pt", ".json")
            with open(json_path, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"New best model saved to {best_save_path} with val acc: {val_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if early_stopping and epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
        # Log metrics
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{current_lr:.6f},{avg_train_loss:.4f},{train_acc:.4f},"
                        f"{avg_val_loss:.4f},{val_acc:.4f}\n")
        
        # Plot confusion matrices at specified intervals
        if (epoch + 1) % confusion_interval == 0 or epoch == num_epochs - 1:
            cm_val = confusion_matrix(val_targets, val_preds)
            print(f"Validation Confusion Matrix (Epoch {epoch+1}):")
            print(cm_val)
    
    # Save final model
    if use_ema:
        torch.save(ema_model.state_dict(), final_save_path)
    else:
        torch.save(model.state_dict(), final_save_path)
    # Save parameters to JSON
    json_path = final_save_path.replace(".pt", ".json")
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Final model saved to {final_save_path}")
    
    # Print best accuracies
    print(f"Best validation accuracy (test set): {best_val_acc:.4f}")
    
    # Return training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    
    return history