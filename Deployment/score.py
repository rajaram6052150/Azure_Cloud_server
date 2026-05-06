"""
score.py
========
Azure ML Scoring Script for ModelA
Entry point for online endpoint predictions
Loads ModelA and processes inference requests
"""

import json
import torch
import numpy as np
from typing import Dict, List, Any
import logging

from model import ModelA


# ==========================================
# CONFIGURATION
# ==========================================

# Update based on your actual dataset
INPUT_SIZE = 10
HIDDEN_DIMS = [64, 32]
DROPOUT = 0.3
MODEL_PATH = "federated_latest.pth"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None


# ==========================================
# LOAD MODEL (Called by Azure ML on startup)
# ==========================================

def init():
    """
    Initialize scoring script by loading the ModelA.
    
    Called automatically by Azure ML when the endpoint starts.
    This function must set the global 'model' variable.
    """
    
    global model
    
    try:
        logger.info("Initializing ModelA for scoring...")
        
        # Create model
        model = ModelA(
            input_dim=INPUT_SIZE,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT
        )
        
        # Load weights
        state_dict = torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict, strict=True)
        
        # Set to evaluation mode
        model.eval()
        
        logger.info("✓ Model loaded successfully!")
        logger.info(f"  Architecture: ModelA")
        logger.info(f"  Input Size: {INPUT_SIZE}")
        logger.info(f"  Hidden Dims: {HIDDEN_DIMS}")
        logger.info(f"  Dropout: {DROPOUT}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# ==========================================
# INFERENCE FUNCTION (Called for each request)
# ==========================================

def run(raw_data: str) -> Dict[str, Any]:
    """
    Score/inference function called for each prediction request.
    
    Called by Azure ML for each online endpoint request.
    
    Args:
        raw_data: JSON string containing request data
                  Expected format:
                  {
                      "features": [float, float, ...],  # List of INPUT_SIZE features
                  }
    
    Returns:
        Dict with prediction results:
        {
            "prediction": 0 or 1,              # Binary prediction
            "probability": float,              # Probability [0, 1]
            "confidence": float,               # Confidence score
        }
        
        Or error response:
        {
            "error": "Error message"
        }
    """
    
    try:
        # Parse JSON input
        data = json.loads(raw_data)
        
        if "features" not in data:
            return {
                "error": "Missing 'features' field in request"
            }
        
        features = data["features"]
        
        # Validate input shape
        if len(features) != INPUT_SIZE:
            return {
                "error": f"Expected {INPUT_SIZE} features, got {len(features)}"
            }
        
        # Convert to NumPy array
        features_array = np.array(features, dtype=np.float32)
        
        # Add batch dimension [features] -> [1, features]
        features_array = np.expand_dims(features_array, axis=0)
        
        # Convert to PyTorch tensor
        features_tensor = torch.tensor(features_array)
        
        # Run inference
        with torch.no_grad():
            output = model(features_tensor)
            # ModelA already applies sigmoid in forward pass
            probability = output.item()
        
        # Binary prediction
        prediction = 1 if probability > 0.5 else 0
        confidence = probability if prediction == 1 else (1 - probability)
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "confidence": float(confidence),
        }
    
    except json.JSONDecodeError:
        logger.error("Invalid JSON in request")
        return {
            "error": "Invalid JSON format"
        }
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return {
            "error": f"Data validation error: {str(e)}"
        }
    
    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}")
        return {
            "error": f"Inference error: {str(e)}"
        }
