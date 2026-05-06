"""
deploy_model.py
===============
Azure ML Deployment for Federated Learning ModelA
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
)

from azure.identity import DefaultAzureCredential

import os


# =========================================================
# AZURE CONFIG
# =========================================================

SUBSCRIPTION_ID = "0746f4dd-3e9d-4b6c-be49-bdcd4149395f"

RESOURCE_GROUP = "flower-rg"

WORKSPACE_NAME = "flower-wsp"

ENDPOINT_NAME = "federated-modela-endpoint"

DEPLOYMENT_NAME = "blue"

ENVIRONMENT_NAME = "federated-modela-env"

MODEL_NAME = "federated-modelA"

MODEL_VERSION = "1"


# =========================================================
# DEPLOYMENT CONFIG
# =========================================================

CODE_PATH = "Deployment"

SCORING_SCRIPT = "score.py"

CONDA_FILE = "Deployment/environment.yml"

INSTANCE_TYPE = "Standard_DS3_v2"

INSTANCE_COUNT = 1


# =========================================================
# CONNECT TO AZURE ML
# =========================================================

def get_ml_client():

    print("=" * 70)

    print("Connecting to Azure ML Workspace...")

    print("=" * 70)

    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    print("✓ Connected successfully!\n")

    return ml_client


# =========================================================
# CREATE ENDPOINT
# =========================================================

def create_endpoint(
    ml_client,
):

    print("Creating/Updating endpoint...")

    print("-" * 70)

    endpoint = ManagedOnlineEndpoint(

        name=ENDPOINT_NAME,

        auth_mode="key",

        description="Federated Learning ModelA Endpoint",
    )

    endpoint = ml_client.begin_create_or_update(
        endpoint
    ).result()

    print(
        f"✓ Endpoint created/updated:"
        f" {endpoint.name}"
    )

    print(
        f"  Provisioning State:"
        f" {endpoint.provisioning_state}"
    )

    print(
        f"  Scoring URI:"
        f" {endpoint.scoring_uri}\n"
    )

    return endpoint


# =========================================================
# CREATE ENVIRONMENT
# =========================================================

def create_environment(
    ml_client,
):

    print("Creating/Updating environment...")

    print("-" * 70)

    print(f"Conda File: {CONDA_FILE}")

    if not os.path.exists(CONDA_FILE):

        raise FileNotFoundError(
            f"Conda file not found: {CONDA_FILE}"
        )

    env = Environment(

        name=ENVIRONMENT_NAME,

        conda_file=CONDA_FILE,

        image=(
            "mcr.microsoft.com/azureml/"
            "openmpi4.1.0-ubuntu20.04:latest"
        ),

        description="Federated Learning Environment",
    )

    env = ml_client.environments.create_or_update(
        env
    )

    print(
        f"✓ Environment created/updated:"
        f" {env.name}"
    )

    print(f"  Version: {env.version}")

    print(f"  Image: {env.image}\n")

    return env


# =========================================================
# CREATE DEPLOYMENT
# =========================================================

def create_deployment(
    ml_client,
    env_version,
):

    print("Creating/Updating deployment...")

    print("-" * 70)

    print(f"Code Path: {CODE_PATH}")

    print(f"Scoring Script: {SCORING_SCRIPT}")

    print(f"Environment Version: {env_version}")

    if not os.path.exists(CODE_PATH):

        raise FileNotFoundError(
            f"Code path not found: {CODE_PATH}"
        )

    deployment = ManagedOnlineDeployment(

        name=DEPLOYMENT_NAME,

        endpoint_name=ENDPOINT_NAME,

        model=f"azureml:{MODEL_NAME}:{MODEL_VERSION}",

        # =================================================
        # IMPORTANT FIX
        # =================================================

        environment=(
            f"azureml:"
            f"{ENVIRONMENT_NAME}:"
            f"{env_version}"
        ),

        code_configuration=CodeConfiguration(

            code=CODE_PATH,

            scoring_script=SCORING_SCRIPT,
        ),

        instance_type=INSTANCE_TYPE,

        instance_count=INSTANCE_COUNT,

        description="Federated Learning ModelA Deployment",
    )

    deployment = ml_client.begin_create_or_update(
        deployment
    ).result()

    print(
        f"✓ Deployment created/updated:"
        f" {deployment.name}"
    )

    print(
        f"  Endpoint:"
        f" {deployment.endpoint_name}"
    )

    print(
        f"  Instance Type:"
        f" {INSTANCE_TYPE}"
    )

    print(
        f"  Instance Count:"
        f" {INSTANCE_COUNT}\n"
    )

    return deployment


# =========================================================
# ROUTE TRAFFIC
# =========================================================

def route_traffic(
    ml_client,
):

    print("Routing traffic...")

    print("-" * 70)

    endpoint = ml_client.online_endpoints.get(
        ENDPOINT_NAME
    )

    endpoint.traffic = {
        DEPLOYMENT_NAME: 100
    }

    endpoint = ml_client.begin_create_or_update(
        endpoint
    ).result()

    print("✓ Traffic routed successfully!\n")

    return endpoint


# =========================================================
# MAIN DEPLOYMENT PIPELINE
# =========================================================

def deploy():

    ml_client = get_ml_client()

    # =====================================================
    # ENDPOINT
    # =====================================================

    endpoint = create_endpoint(
        ml_client
    )

    # =====================================================
    # ENVIRONMENT
    # =====================================================

    env = create_environment(
        ml_client
    )

    env_version = env.version

    # =====================================================
    # DEPLOYMENT
    # =====================================================

    deployment = create_deployment(
        ml_client,
        env_version
    )

    # =====================================================
    # TRAFFIC
    # =====================================================

    endpoint = route_traffic(
        ml_client
    )

    print("=" * 70)

    print("✓ DEPLOYMENT SUCCESSFUL!")

    print("=" * 70)

    print(f"Endpoint Name : {endpoint.name}")

    print(f"Deployment    : {deployment.name}")

    print(f"Scoring URI   : {endpoint.scoring_uri}")

    print("=" * 70)

    return endpoint


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)

    print(
        "Federated Learning ModelA "
        "- Azure ML Deployment"
    )

    print("=" * 70 + "\n")

    try:

        endpoint = deploy()

        print("\n✓ Model deployed successfully!")

        print("\nTest using:")

        print(
            f"\ncurl -X POST "
            f"'{endpoint.scoring_uri}' \\"
        )

        print(
            "  -H 'Content-Type: application/json' \\"
        )

        print(
            "  -d '{\"features\": "
            "[0.1,0.2,0.3,0.4,0.5,"
            "0.6,0.7,0.8,0.9,1.0]}'"
        )

    except Exception as e:

        print(f"\n✗ Deployment failed: {e}")

        import traceback

        traceback.print_exc()
        