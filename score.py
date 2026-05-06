"""
score.py
========
Azure ML scoring script for ModelA
"""

import json
import os
import logging

import torch
import numpy as np

from model import ModelA


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_SIZE = 10

HIDDEN_DIMS = [64, 32]

DROPOUT = 0.3


# =========================================================
# AZURE MODEL PATH
# =========================================================

MODEL_DIR = os.getenv("AZUREML_MODEL_DIR")

print(f"AZUREML_MODEL_DIR: {MODEL_DIR}")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "federated_latest.pth"
)

print(f"Resolved MODEL_PATH: {MODEL_PATH}")


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# =========================================================
# GLOBAL MODEL
# =========================================================

model = None


# =========================================================
# INITIALIZE MODEL
# =========================================================

def init():

    global model

    try:

        logger.info(
            "Initializing ModelA..."
        )

        # =================================================
        # CREATE MODEL
        # =================================================

        model = ModelA(

            input_dim=INPUT_SIZE,

            hidden_dims=HIDDEN_DIMS,

            dropout=DROPOUT,
        )

        logger.info(
            "Loading model weights..."
        )

        # =================================================
        # LOAD MODEL
        # =================================================

        state_dict = torch.load(

            MODEL_PATH,

            map_location=torch.device("cpu")
        )

        model.load_state_dict(
            state_dict,
            strict=True
        )

        # =================================================
        # EVAL MODE
        # =================================================

        model.eval()

        logger.info(
            "✓ Model loaded successfully!"
        )

        logger.info(
            f"Input Size: {INPUT_SIZE}"
        )

        logger.info(
            f"Hidden Dims: {HIDDEN_DIMS}"
        )

    except Exception as e:

        logger.error(
            f"Failed to initialize model: {e}"
        )

        raise


# =========================================================
# INFERENCE
# =========================================================

def run(raw_data):

    try:

        # =================================================
        # PARSE REQUEST
        # =================================================

        data = json.loads(raw_data)

        # =================================================
        # VALIDATE INPUT
        # =================================================

        if "features" not in data:

            return {

                "error": (
                    "Missing 'features' field"
                )
            }

        features = data["features"]

        # =================================================
        # VALIDATE FEATURE SIZE
        # =================================================

        if len(features) != INPUT_SIZE:

            return {

                "error": (

                    f"Expected "
                    f"{INPUT_SIZE} features, "

                    f"got {len(features)}"
                )
            }

        # =================================================
        # TO NUMPY
        # =================================================

        features_array = np.array(

            features,

            dtype=np.float32
        )

        # =================================================
        # ADD BATCH DIMENSION
        # =================================================

        features_array = np.expand_dims(

            features_array,

            axis=0
        )

        # =================================================
        # TO TENSOR
        # =================================================

        features_tensor = torch.tensor(
            features_array
        )

        # =================================================
        # INFERENCE
        # =================================================

        with torch.no_grad():

            output = model(
                features_tensor
            )

            # Model already applies sigmoid
            probability = output.item()

        # =================================================
        # PREDICTION
        # =================================================

        prediction = (
            1 if probability > 0.5 else 0
        )

        confidence = (

            probability

            if prediction == 1

            else (1 - probability)
        )

        logger.info(

            f"Prediction={prediction}, "

            f"Probability={probability:.4f}"
        )

        # =================================================
        # RESPONSE
        # =================================================

        return {

            "prediction": int(prediction),

            "probability": float(probability),

            "confidence": float(confidence),
        }

    except json.JSONDecodeError:

        logger.error(
            "Invalid JSON received"
        )

        return {
            "error": "Invalid JSON format"
        }

    except Exception as e:

        logger.error(
            f"Inference failed: {e}"
        )

        return {
            "error": str(e)
        }
    