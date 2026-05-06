"""
inference.py
=============
Inference API for Federated Learning Model (ModelA)
Loads final global model and exposes prediction endpoint
Updated with enhanced regularization (BatchNorm + Dropout + Sigmoid)
"""

from flask import Flask, request, jsonify
import torch
import numpy as np
from typing import Dict, List
import logging

from model import ModelA


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "models/federated_latest.pth"

# IMPORTANT: Update based on your actual feature count
# Calculate: feature count after preprocessing
INPUT_SIZE = 10  # Update this based on your dataset

HIDDEN_DIMS = [64, 32]
DROPOUT = 0.3
DEVICE = torch.device("cpu")  # Use "cuda" if GPU available

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# LOAD MODEL
# ==========================================

def load_inference_model(
    model_path: str = MODEL_PATH,
    input_size: int = INPUT_SIZE,
    hidden_dims: List[int] = None,
    dropout: float = DROPOUT,
    device: str = DEVICE
) -> ModelA:
    """
    Load trained ModelA for inference.
    
    Args:
        model_path: Path to saved model checkpoint
        input_size: Number of input features
        hidden_dims: Hidden layer dimensions
        dropout: Dropout probability
        device: Device to load model on ("cpu" or "cuda")
    
    Returns:
        ModelA instance in eval mode
    """
    
    if hidden_dims is None:
        hidden_dims = HIDDEN_DIMS
    
    # Create model
    model = ModelA(
        input_dim=input_size,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    logger.info(f"Model loaded from: {model_path}")
    return model


# Initialize model
try:
    model = load_inference_model()
    logger.info("=" * 60)
    logger.info("Federated Learning Inference Server (ModelA)")
    logger.info("Model loaded successfully!")
    logger.info("=" * 60)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/", methods=["GET"])
def home() -> Dict:
    """
    Health check endpoint.
    
    Returns:
        JSON with service status
    """
    return jsonify({
        "status": "healthy",
        "message": "FL ModelA Prediction API Running",
        "model": "federated_latest.pth",
        "device": str(DEVICE),
    })


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.route("/predict", methods=["POST"])
def predict() -> Dict:
    """
    Make prediction on input features.
    
    Request body:
    {
        "features": [float, float, ...],  # List of features (shape must match INPUT_SIZE)
    }
    
    Response:
    {
        "probability": float,  # Predicted probability [0, 1]
        "prediction": int,      # Binary prediction (0 or 1)
        "confidence": float,    # Confidence score
    }
    
    Returns:
        JSON with prediction and confidence
    """
    
    try:
        data = request.json
        
        if not data or "features" not in data:
            return jsonify({
                "error": "Missing 'features' in request body"
            }), 400
        
        features = data["features"]
        
        # Validate input shape
        if len(features) != INPUT_SIZE:
            return jsonify({
                "error": f"Expected {INPUT_SIZE} features, got {len(features)}"
            }), 400
        
        # Convert to tensor
        features_array = np.array(features, dtype=np.float32)
        features_tensor = torch.tensor(features_array).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            output = model(features_tensor)
            probability = output.item()  # Already sigmoid applied in ModelA
        
        # Binary prediction
        prediction = 1 if probability > 0.5 else 0
        confidence = probability if prediction == 1 else (1 - probability)
        
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
        
        return jsonify({
            "probability": float(probability),
            "prediction": int(prediction),
            "confidence": float(confidence),
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# BATCH PREDICTION
# ==========================================

@app.route("/predict_batch", methods=["POST"])
def predict_batch() -> Dict:
    """
    Make predictions on batch of samples.
    
    Request body:
    {
        "features": [
            [float, float, ...],
            [float, float, ...],
            ...
        ]
    }
    
    Returns:
        JSON with batch predictions
    """
    
    try:
        data = request.json
        
        if not data or "features" not in data:
            return jsonify({
                "error": "Missing 'features' in request body"
            }), 400
        
        features_list = data["features"]
        
        # Validate
        for features in features_list:
            if len(features) != INPUT_SIZE:
                return jsonify({
                    "error": f"Expected {INPUT_SIZE} features per sample"
                }), 400
        
        # Convert to tensor
        features_array = np.array(features_list, dtype=np.float32)
        features_tensor = torch.tensor(features_array).to(DEVICE)
        
        # Batch inference
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = outputs.squeeze().cpu().numpy()
        
        # Ensure 1D array
        if probabilities.ndim == 0:
            probabilities = np.array([probabilities.item()])
        
        predictions = (probabilities > 0.5).astype(int).tolist()
        confidences = np.where(
            probabilities > 0.5,
            probabilities,
            1 - probabilities
        ).tolist()
        
        return jsonify({
            "predictions": predictions,
            "probabilities": probabilities.tolist(),
            "confidences": confidences,
            "batch_size": len(predictions),
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":
    logger.info("Starting Inference Server...")
    app.run(host="0.0.0.0", port=5000, debug=False)
