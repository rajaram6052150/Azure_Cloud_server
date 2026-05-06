from azure.ai.ml import MLClient

from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
)

from azure.identity import DefaultAzureCredential


# ==========================================
# AZURE CONFIG
# ==========================================

SUBSCRIPTION_ID = "0746f4dd-3e9d-4b6c-be49-bdcd4149395f"

RESOURCE_GROUP = "flower-rg"

WORKSPACE_NAME = "flower-wsp"

ENDPOINT_NAME = "federated-modela-endpoint"

DEPLOYMENT_NAME = "blue"


# ==========================================
# CONNECT TO AZURE ML
# ==========================================

ml_client = MLClient(

    DefaultAzureCredential(),

    SUBSCRIPTION_ID,

    RESOURCE_GROUP,

    WORKSPACE_NAME,
)

print("Connected to Azure ML!")


# ==========================================
# CREATE ENDPOINT
# ==========================================

endpoint = ManagedOnlineEndpoint(

    name=ENDPOINT_NAME,

    auth_mode="key",
)

print("\nCreating endpoint...")

ml_client.begin_create_or_update(
    endpoint
).result()

print("Endpoint created!")


# ==========================================
# CREATE ENVIRONMENT
# ==========================================

env = Environment(

    name="federated-modela-env",

    conda_file="Deployment/environment.yml",

    image=(
        "mcr.microsoft.com/azureml/"
        "openmpi4.1.0-ubuntu20.04:latest"
    ),
)

env = ml_client.environments.create_or_update(
    env
)

print("\nEnvironment created!")

print(f"Environment Version: {env.version}")


# ==========================================
# CREATE DEPLOYMENT
# ==========================================

deployment = ManagedOnlineDeployment(

    name=DEPLOYMENT_NAME,

    endpoint_name=ENDPOINT_NAME,

    model="azureml:federated-modelA:1",

    environment=(
        f"azureml:federated-modela-env:{env.version}"
    ),

    code_configuration=CodeConfiguration(

        code="Deployment",

        scoring_script="score.py",
    ),

    instance_type="Standard_DS3_v2",

    instance_count=1,
)

print("\nDeploying model...")

ml_client.begin_create_or_update(
    deployment
).result()

print("\nDEPLOYMENT SUCCESSFUL!")


# ==========================================
# ROUTE TRAFFIC
# ==========================================

endpoint.traffic = {
    "blue": 100
}

ml_client.begin_create_or_update(
    endpoint
).result()

print("\nTraffic assigned successfully!")

print("\nEndpoint Ready!")