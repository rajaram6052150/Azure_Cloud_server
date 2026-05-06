"""
save_model.py
=============
Utility functions for:
- Saving aggregated federated global model
- Loading saved checkpoints
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
    Convert Flower aggregated parameters into a
    PyTorch ModelA checkpoint.

    Args:
        parameters:
            Flower aggregated parameters

        round_num:
            Current federated round number

        input_size:
            Number of input features

        output_dir:
            Directory to save checkpoints
    """

    print("\n[SAVE_MODEL] Creating output directory...")

    os.makedirs(output_dir, exist_ok=True)

    print("[SAVE_MODEL] Output directory ready!")

    # =====================================================
    # CONVERT FLOWER PARAMETERS
    # =====================================================

    print("[SAVE_MODEL] Converting parameters...")

    ndarrays = parameters_to_ndarrays(
        parameters
    )

    print(
        f"[SAVE_MODEL] "
        f"{len(ndarrays)} tensors received!"
    )

    # =====================================================
    # INITIALIZE MODEL
    # =====================================================

    print("[SAVE_MODEL] Initializing ModelA...")

    model = ModelA(
        input_dim=input_size,
        hidden_dims=[64, 32],
        dropout=0.3
    )

    print("[SAVE_MODEL] Model initialized!")

    # =====================================================
    # MAP PARAMETERS
    # =====================================================

    print("[SAVE_MODEL] Mapping tensors...")

    params_dict = zip(
        model.state_dict().keys(),
        ndarrays
    )

    state_dict = OrderedDict({

        key: torch.tensor(value)

        for key, value in params_dict
    })

    print("[SAVE_MODEL] State dict created!")

    # =====================================================
    # LOAD WEIGHTS
    # =====================================================

    print("[SAVE_MODEL] Loading weights...")

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print("[SAVE_MODEL] Weights loaded!")

    # =====================================================
    # SAVE ROUND CHECKPOINT
    # =====================================================

    round_model_path = os.path.join(
        output_dir,
        f"federated_round_{round_num}.pth"
    )

    print(
        f"[SAVE_MODEL] Saving:"
        f" {round_model_path}"
    )

    torch.save(
        model.state_dict(),
        round_model_path
    )

    print(
        f"[MODEL SAVED] "
        f"{round_model_path}"
    )

    # =====================================================
    # SAVE LATEST MODEL
    # =====================================================

    latest_model_path = os.path.join(
        output_dir,
        "federated_latest.pth"
    )

    print(
        f"[SAVE_MODEL] Saving latest:"
        f" {latest_model_path}"
    )

    torch.save(
        model.state_dict(),
        latest_model_path
    )

    print(
        f"[MODEL SAVED] "
        f"{latest_model_path}"
    )

    print("[SAVE_MODEL] Completed successfully!")


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

    Args:
        model_path:
            Path to .pth file

        input_size:
            Number of input features

        hidden_dims:
            Hidden layer sizes

        dropout:
            Dropout probability

    Returns:
        Loaded ModelA instance
    """

    print("\n[LOAD_MODEL] Loading model...")

    if hidden_dims is None:

        hidden_dims = [64, 32]

    # =====================================================
    # INITIALIZE MODEL
    # =====================================================

    model = ModelA(
        input_dim=input_size,
        hidden_dims=hidden_dims,
        dropout=dropout
    )

    print("[LOAD_MODEL] Model initialized!")

    # =====================================================
    # LOAD STATE DICT
    # =====================================================

    print(
        f"[LOAD_MODEL] Reading:"
        f" {model_path}"
    )

    state_dict = torch.load(
        model_path,
        map_location="cpu"
    )

    print("[LOAD_MODEL] State dict loaded!")

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print(
        f"[MODEL LOADED] "
        f"{model_path}"
    )

    return model