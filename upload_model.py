"""
upload_model.py
===============
Upload Final Federated Learning Model (ModelA) to Azure ML Registry
Updated with enhanced regularization and better error handling
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
import os
from typing import Optional


# =========================================================
# AZURE ML CONFIG
# =========================================================

SUBSCRIPTION_ID = "0746f4dd-3e9d-4b6c-be49-bdcd4149395f"
RESOURCE_GROUP = "flower-rg"
WORKSPACE_NAME = "flower-wsp"
MODEL_NAME = "federated-modelA"
MODEL_PATH = "models/federated_latest.pth"


# =========================================================
# CONNECT TO AZURE ML
# =========================================================

def get_ml_client(
    subscription_id: str = SUBSCRIPTION_ID,
    resource_group: str = RESOURCE_GROUP,
    workspace_name: str = WORKSPACE_NAME
) -> MLClient:
    """
    Create and return Azure ML client.
    
    Args:
        subscription_id: Azure subscription ID
        resource_group: Azure resource group name
        workspace_name: Azure ML workspace name
    
    Returns:
        Configured MLClient instance
    """
    
    print("=" * 60)
    print("Connecting to Azure ML Workspace...")
    print("=" * 60)
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    
    print("✓ Connected successfully!\n")
    return ml_client


# =========================================================
# UPLOAD MODEL FUNCTION
# =========================================================

def upload_model(
    model_path: str = MODEL_PATH,
    model_name: str = MODEL_NAME,
    description: Optional[str] = None
) -> Model:
    """
    Upload trained federated learning model to Azure ML Registry.
    
    Args:
        model_path: Path to the .pth model file
        model_name: Name for the model in Azure ML Registry
        description: Optional model description
    
    Returns:
        Registered model object from Azure ML
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If upload fails
    """
    
    # Validate model path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # Get Azure ML client
    ml_client = get_ml_client()
    
    # Create model object
    if description is None:
        description = "Federated Learning Model (ModelA) with enhanced regularization (BatchNorm + Dropout + Sigmoid)"
    
    print(f"Model Details:")
    print(f"  Path: {model_path}")
    print(f"  Size: {model_size_mb:.2f} MB")
    print(f"  Name: {model_name}")
    print(f"=" * 60)
    
    model = Model(
        path=model_path,
        name=model_name,
        description=description,
        type="custom_model",
    )
    
    # Upload model
    print("\nUploading model to Azure ML Registry...")
    registered_model = ml_client.models.create_or_update(model)
    
    # Print results
    print("\n" + "=" * 60)
    print("✓ MODEL UPLOADED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model Name    : {registered_model.name}")
    print(f"Model Version : {registered_model.version}")
    print(f"Model ID      : {registered_model.id}")
    print("=" * 60 + "\n")
    
    return registered_model


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print("Federated Learning Model Upload")
        print("=" * 60 + "\n")
        
        registered_model = upload_model()
        
        print(f"✓ Model ready for deployment:")
        print(f"  azureml:{registered_model.name}:{registered_model.version}\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("  Make sure to train the model first and save it to models/federated_latest.pth\n")
        exit(1)
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        raise
