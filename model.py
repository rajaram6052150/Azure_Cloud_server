"""
model.py
========
ModelA Architecture for Federated Learning
Multi-Layer Perceptron with BatchNorm, ReLU, and Dropout for regularization
"""

import torch
import torch.nn as nn


class ModelA(nn.Module):
    """
    Multi-Layer Perceptron for binary classification with regularization.
    
    Architecture per hidden layer:
        Linear -> BatchNorm -> ReLU -> Dropout
    
    Output:
        Sigmoid activation -> probability in [0,1]
    
    Args:
        input_dim (int): Number of input features
        hidden_dims (list): List of hidden layer dimensions, e.g., [64, 32]
        dropout (float): Dropout probability (default: 0.3)
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with BatchNorm + ReLU + Dropout
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            prev_dim = h_dim
        
        # Output layer with Sigmoid
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Tensor of shape (batch_size, 1) with probabilities in [0,1]
        """
        return self.network(x)
