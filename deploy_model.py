"""
deploy_model.py
===============
Azure ML Deployment for Federated Learning Model (ModelA)
Creates managed online endpoint and deployment with ModelA architecture
Updated with enhanced regularization (BatchNorm + Dropout + Sigmoid)
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential
from typing import Optional
import os


# ==========================================
# AZURE ML CONFIG
# ==========================================

SUBSCRIPTION_ID = "0746f4dd-3e9d-4b6c-be49-bdcd4149395f"
RESOURCE_GROUP = "flower-rg"
WORKSPACE_NAME = "flower-wsp"
ENDPOINT_NAME = "federated-modelA-endpoint"
DEPLOYMENT_NAME = "blue"
ENVIRONMENT_NAME = "federated-modelA-env"
MODEL_VERSION = "1"


# ==========================================
# CONNECT TO AZURE ML
# ==========================================

def get_ml_client(
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
        Configured MLClient instance
    """
    
    print("=" * 70)
    print("Connecting to Azure ML Workspace...")
    print("=" * 70)
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    
    print("✓ Connected successfully!\n")
    return ml_client


# ==========================================
# CREATE/UPDATE ENDPOINT
# ==========================================

def create_or_update_endpoint(
    ml_client: MLClient,
    endpoint_name: str = ENDPOINT_NAME,
    auth_mode: str = "key"
) -> ManagedOnlineEndpoint:
    """
    Create or update managed online endpoint.
    
    Args:
        ml_client: Azure ML client
        endpoint_name: Name of the endpoint
        auth_mode: Authentication mode ("key" or "aad")
    
    Returns:
        ManagedOnlineEndpoint object
    """
    
    print("Creating/Updating endpoint...")
    print("-" * 70)
    
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode=auth_mode,
        description="Federated Learning ModelA Endpoint",
    )
    
    endpoint = ml_client.begin_create_or_update(endpoint).result()
    
    print(f"✓ Endpoint created/updated: {endpoint.name}")
    print(f"  Provisioning State: {endpoint.provisioning_state}")
    print(f"  Scoring URI: {endpoint.scoring_uri}\n")
    
    return endpoint


# ==========================================
# CREATE ENVIRONMENT
# ==========================================

def create_or_update_environment(
    ml_client: MLClient,
    env_name: str = ENVIRONMENT_NAME,
    conda_file: str = "FL_Server/Deployment/environment.yaml",
    base_image: str = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
) -> Environment:
    """
    Create or update environment for deployment.
    
    Args:
        ml_client: Azure ML client
        env_name: Environment name
        conda_file: Path to conda environment file
        base_image: Base Docker image
    
    Returns:
        Environment object
    """
    
    print("Creating/Updating environment...")
    print("-" * 70)
    
    # Check if conda file exists
    if not os.path.exists(conda_file):
        print(f"⚠ Conda file not found: {conda_file}")
        print("  Using default dependencies...")
        
        # Create minimal environment
        env = Environment(
            name=env_name,
            image=base_image,
            description="Federated Learning ModelA Environment",
        )
    else:
        env = Environment(
            name=env_name,
            conda_file=conda_file,
            image=base_image,
            description="Federated Learning ModelA Environment",
        )
    
    env = ml_client.environments.create_or_update(env)
    
    print(f"✓ Environment created/updated: {env.name}")
    print(f"  Image: {env.image}\n")
    
    return env


# ==========================================
# CREATE DEPLOYMENT
# ==========================================

def create_or_update_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str = DEPLOYMENT_NAME,
    model_name: str = "federated-modelA",
    model_version: str = MODEL_VERSION,
    env_name: str = ENVIRONMENT_NAME,
    code_path: str = "FL_Server/Deployment",
    score_script: str = "score.py",
    instance_type: str = "Standard_DS1_v2",
    instance_count: int = 1
) -> ManagedOnlineDeployment:
    """
    Create or update managed online deployment.
    
    Args:
        ml_client: Azure ML client
        endpoint_name: Endpoint to deploy to
        deployment_name: Deployment name
        model_name: Model name in registry
        model_version: Model version
        env_name: Environment name
        code_path: Path to scoring code
        score_script: Name of scoring script
        instance_type: VM instance type
        instance_count: Number of instances
    
    Returns:
        ManagedOnlineDeployment object
    """
    
    print("Creating/Updating deployment...")
    print("-" * 70)
    
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=f"azureml:{model_name}:{model_version}",
        environment=f"azureml:{env_name}@latest",
        code_configuration=CodeConfiguration(
            code=code_path,
            scoring_script=score_script,
        ),
        instance_type=instance_type,
        instance_count=instance_count,
        description="Federated Learning ModelA Deployment",
    )
    
    deployment = ml_client.begin_create_or_update(deployment).result()
    
    print(f"✓ Deployment created/updated: {deployment.name}")
    print(f"  Endpoint: {deployment.endpoint_name}")
    print(f"  Model: {model_name}:{model_version}")
    print(f"  Instance Type: {instance_type}")
    print(f"  Instance Count: {instance_count}\n")
    
    return deployment


# ==========================================
# MAIN DEPLOYMENT PIPELINE
# ==========================================

def deploy_model(
    endpoint_name: str = ENDPOINT_NAME,
    deployment_name: str = DEPLOYMENT_NAME,
    model_name: str = "federated-modelA"
) -> None:
    """
    Complete deployment pipeline:
    1. Create/Update endpoint
    2. Create/Update environment
    3. Create/Update deployment
    """
    
    try:
        # Connect to Azure ML
        ml_client = get_ml_client()
        
        # Step 1: Create endpoint
        endpoint = create_or_update_endpoint(ml_client, endpoint_name)
        
        # Step 2: Create environment
        env = create_or_update_environment(ml_client)
        
        # Step 3: Create deployment
        deployment = create_or_update_deployment(
            ml_client,
            endpoint_name,
            deployment_name,
            model_name
        )
        
        # Print summary
        print("=" * 70)
        print("✓ DEPLOYMENT SUCCESSFUL!")
        print("=" * 70)
        print(f"Endpoint: {endpoint.name}")
        print(f"Scoring URI: {endpoint.scoring_uri}")
        print(f"Deployment: {deployment.name}")
        print("=" * 70 + "\n")
        
        return endpoint, deployment
    
    except Exception as e:
        print(f"\n✗ Deployment failed: {e}\n")
        raise


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Federated Learning ModelA - Azure ML Deployment")
    print("=" * 70 + "\n")
    
    try:
        endpoint, deployment = deploy_model()
        print("✓ Model is now available for inference!\n")
        print("To make predictions, call:")
        print(f"  curl -X POST '{endpoint.scoring_uri}' \\")
        print(f"    -H 'Content-Type: application/json' \\")
        print(f"    -d '{{\"features\": [...]}}'\n")
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        exit(1)
