import os

# =========================================================
# FIX PYTORCH / OPENMP HANGS
# =========================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("1 - Starting imports")

# =========================================================
# IMPORT TORCH EARLY
# =========================================================

print("2 - importing torch...")

import torch

print("3 - torch imported successfully")

# =========================================================
# NORMAL IMPORTS
# =========================================================

from collections import OrderedDict

print("4 - collections imported")

from flwr.common import parameters_to_ndarrays

print("5 - flower common imported")

from model import ModelA

print("6 - ModelA imported")


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
    Save aggregated federated model.
    """

    print("\n[SAVE_MODEL] Starting save...")

    # =====================================================
    # CREATE DIRECTORY
    # =====================================================

    os.makedirs(output_dir, exist_ok=True)

    print("[SAVE_MODEL] Output directory ready")

    # =====================================================
    # CONVERT FLOWER PARAMETERS
    # =====================================================

    print("[SAVE_MODEL] Converting parameters...")

    ndarrays = parameters_to_ndarrays(
        parameters
    )

    print(
        f"[SAVE_MODEL] "
        f"{len(ndarrays)} tensors received"
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

    print("[SAVE_MODEL] Model initialized")

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

    print("[SAVE_MODEL] State dict created")

    # =====================================================
    # LOAD WEIGHTS
    # =====================================================

    print("[SAVE_MODEL] Loading weights...")

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print("[SAVE_MODEL] Weights loaded")

    # =====================================================
    # SAVE ROUND MODEL
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

    print("[LOAD_MODEL] Model initialized")

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

    print("[LOAD_MODEL] State dict loaded")

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print(
        f"[MODEL LOADED] "
        f"{model_path}"
    )

    return model