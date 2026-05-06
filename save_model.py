"""
save_model.py
=============
Utility to save aggregated federated model from Flower parameters to PyTorch .pth
"""

import os
import torch
from collections import OrderedDict
from flwr.common import parameters_to_ndarrays
from model import ModelA


def save_global_model(parameters, round_num: int, input_size: int, output_dir: str = "models"):
    """
    Convert Flower aggregated parameters into a PyTorch model and save as .pth checkpoint.
    
    Args:
        parameters: Aggregated Flower parameters
        round_num: Current federated round number
        input_size: Number of input features
        output_dir: Directory to save models (default: "models")
    """
    
    # Create models directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert Flower parameters -> NumPy arrays
    ndarrays = parameters_to_ndarrays(parameters)
    
    # Initialize model with default architecture
    model = ModelA(
        input_dim=input_size,
        hidden_dims=[64, 32],
        dropout=0.3
    )
    
    # Map weights to model state_dict
    params_dict = zip(model.state_dict().keys(), ndarrays)
    
    state_dict = OrderedDict({
        k: torch.tensor(v)
        for k, v in params_dict
    })
    
    # Load weights into model
    model.load_state_dict(state_dict, strict=True)
    
    # Save round checkpoint
    round_path = os.path.join(output_dir, f"federated_round_{round_num}.pth")
    torch.save(model.state_dict(), round_path)
    print(f"[MODEL SAVED] {round_path}")
    
    # Save as latest model
    latest_path = os.path.join(output_dir, "federated_latest.pth")
    torch.save(model.state_dict(), latest_path)
    print(f"[MODEL SAVED] {latest_path} (latest)")


def load_model(model_path: str, input_size: int, hidden_dims: list = None, dropout: float = 0.3):
    """
    Load a saved ModelA from checkpoint.
    
    Args:
        model_path: Path to .pth file
        input_size: Number of input features
        hidden_dims: List of hidden layer dimensions (default: [64, 32])
        dropout: Dropout probability (default: 0.3)
    
    Returns:
        ModelA instance with loaded weights
    """
    
    if hidden_dims is None:
        hidden_dims = [64, 32]
    
    model = ModelA(
        input_dim=input_size,
        hidden_dims=hidden_dims,
        dropout=dropout
    )
    
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    
    print(f"[MODEL LOADED] {model_path}")
    return model
