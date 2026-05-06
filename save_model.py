"""
save_model.py
=============
Save and load federated global model checkpoints.
"""

import os
from collections import OrderedDict

import torch

from flwr.common import parameters_to_ndarrays

from model import ModelA


# =========================================================
# SAVE GLOBAL MODEL
# =========================================================

def save_global_model(
    parameters,
    round_num: int,
    input_size: int,
    output_dir: str = "models",
):
    """
    Save aggregated federated model as PyTorch checkpoint.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Convert Flower parameters -> NumPy arrays
    ndarrays = parameters_to_ndarrays(parameters)

    # Initialize model
    model = ModelA(
        input_dim=input_size,
        hidden_dims=[64, 32],
        dropout=0.3
    )

    # Map weights
    params_dict = zip(
        model.state_dict().keys(),
        ndarrays
    )

    state_dict = OrderedDict({
        key: torch.tensor(value)
        for key, value in params_dict
    })

    # Load weights
    model.load_state_dict(
        state_dict,
        strict=True
    )

    # Save round checkpoint
    round_model_path = os.path.join(
        output_dir,
        f"federated_round_{round_num}.pth"
    )

    torch.save(
        model.state_dict(),
        round_model_path
    )

    print(f"[MODEL SAVED] {round_model_path}")

    # Save latest model
    latest_model_path = os.path.join(
        output_dir,
        "federated_latest.pth"
    )

    torch.save(
        model.state_dict(),
        latest_model_path
    )

    print(f"[MODEL SAVED] {latest_model_path}")


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(
    model_path: str,
    input_size: int,
    hidden_dims: list = None,
    dropout: float = 0.3,
):
    """
    Load ModelA checkpoint.
    """

    if hidden_dims is None:

        hidden_dims = [64, 32]

    model = ModelA(
        input_dim=input_size,
        hidden_dims=hidden_dims,
        dropout=dropout
    )

    state_dict = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print(f"[MODEL LOADED] {model_path}")

    return model