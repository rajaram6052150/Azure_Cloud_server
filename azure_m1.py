"""
azure_m1.py
===========
Azure ML Integration for Federated Learning
Updated with ModelA architecture (BatchNorm + Dropout + Sigmoid)
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from typing import Optional
import os


# =========================================================
# AZURE ML CONFIGURATION
# =========================================================

SUBSCRIPTION_ID = "0746f4dd-3e9d-4b6c-be49-bdcd4149395f"
RESOURCE_GROUP = "flower-rg"
WORKSPACE_NAME = "flower-wsp"
MODEL_NAME = "federated-modelA"
MODEL_DESCRIPTION = "Federated Learning Model with enhanced regularization (BatchNorm + Dropout)"


# =========================================================
# CONNECT TO AZURE ML
# =========================================================

def connect_to_azure(
    subscription_id: str = SUBSCRIPTION_ID,
    resource_group: str = RESOURCE_GROUP,
    workspace_name: str = WORKSPACE_NAME
) -> MLClient:
    """
    Establish connection to Azure ML Workspace.
    
    Args:
        subscription_id: Azure subscription ID
        resource_group: Azure resource group name
        workspace_name: Azure ML workspace name
    
    Returns:
        MLClient instance
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
    
    print("✓ Connected successfully!")
    return ml_client


# =========================================================
# UPLOAD MODEL
# =========================================================

def upload_model_to_azure(
    model_path: str = "models/federated_latest.pth",
    model_name: str = MODEL_NAME,
    description: str = MODEL_DESCRIPTION,
    ml_client: Optional[MLClient] = None
) -> Model:
    """
    Upload trained federated model to Azure ML Registry.
    
    Args:
        model_path: Path to the .pth model file
        model_name: Name for the model in Azure ML Registry
        description: Model description
        ml_client: Azure ML client (creates if not provided)
    
    Returns:
        Registered model object
    """
    
    if ml_client is None:
        ml_client = connect_to_azure()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"\nUploading model: {model_path}")
    print("=" * 60)
    
    model = Model(
        path=model_path,
        name=model_name,
        description=description,
        type="custom_model",
    )
    
    registered_model = ml_client.models.create_or_update(model)
    
    print("\n✓ MODEL UPLOADED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model Name    : {registered_model.name}")
    print(f"Model Version : {registered_model.version}")
    print(f"Model ID      : {registered_model.id}")
    print("=" * 60)
    
    return registered_model


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    try:
        ml_client = connect_to_azure()
        registered_model = upload_model_to_azure(ml_client=ml_client)
        print(f"\n✓ Ready to use: {registered_model.name}:{registered_model.version}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
