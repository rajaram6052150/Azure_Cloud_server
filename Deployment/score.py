import json
import os
import torch
import numpy as np

from model import ModelA


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_SIZE = 10

HIDDEN_DIMS = [64, 32]

DROPOUT = 0.3


# ==========================================
# MODEL PATH
# ==========================================

MODEL_DIR = os.getenv("AZUREML_MODEL_DIR")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "federated_latest.pth"
)

print(f"MODEL PATH: {MODEL_PATH}")


# ==========================================
# GLOBAL MODEL
# ==========================================

model = None


# ==========================================
# LOAD MODEL
# ==========================================

def init():

    global model

    model = ModelA(

        input_dim=INPUT_SIZE,

        hidden_dims=HIDDEN_DIMS,

        dropout=DROPOUT
    )

    model.load_state_dict(

        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    print("Model loaded successfully!")


# ==========================================
# INFERENCE FUNCTION
# ==========================================

def run(raw_data):

    try:

        data = json.loads(raw_data)

        features = np.array(

            data["features"],

            dtype=np.float32
        )

        features = np.expand_dims(
            features,
            axis=0
        )

        x = torch.tensor(features)

        with torch.no_grad():

            output = model(x)

            probability = output.item()

        prediction = (

            "Churn"

            if probability > 0.5

            else "No Churn"
        )

        return {

            "prediction": prediction,

            "probability": probability
        }

    except Exception as e:

        return {
            "error": str(e)
        }