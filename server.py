"""
server.py
=========
Federated Learning Server with Azure MLflow Tracking

Features:
- Flower Federated Learning Server
- FedAvg aggregation
- Azure MLflow experiment tracking
- Global model checkpoint saving
- Federated metrics aggregation
"""

import os
from typing import List, Tuple

import flwr as fl
from flwr.server.strategy import FedAvg

import mlflow

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from save_model import save_global_model


# =========================================================
# AZURE ML CONFIG
# =========================================================

SUBSCRIPTION_ID = "0746f4dd-3e9d-4b6c-be49-bdcd4149395f"

RESOURCE_GROUP = "flower-rg"

WORKSPACE_NAME = "flower-wsp"


# =========================================================
# FEDERATED CONFIG
# =========================================================

SERVER_ADDRESS = "0.0.0.0:8080"

NUM_ROUNDS = 5

MIN_CLIENTS = 2

INPUT_SIZE = 10


# =========================================================
# METRIC AGGREGATION
# =========================================================

def weighted_average(metrics):
    """
    Compute weighted average of client metrics.
    """

    if not metrics:
        return {}

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    aggregated = {}

    first_metrics = metrics[0][1]

    if not first_metrics:
        return {}

    metric_keys = first_metrics.keys()

    for key in metric_keys:

        weighted_sum = sum(
            num_examples * metric_dict.get(key, 0.0)
            for num_examples, metric_dict in metrics
        )

        aggregated[key] = (
            weighted_sum / total_examples
        )

    return aggregated


# =========================================================
# CUSTOM FEDAVG STRATEGY
# =========================================================

class FedAvgCustom(FedAvg):

    # =====================================================
    # AGGREGATE FIT
    # =====================================================

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[Tuple],
    ):

        print("\n" + "=" * 70)

        print(f"FEDERATED ROUND {server_round}")

        print("=" * 70)

        print(f"✓ Clients completed: {len(results)}")

        print(f"✗ Clients failed: {len(failures)}")

        # -------------------------------------------------
        # FLOWER DEFAULT AGGREGATION
        # -------------------------------------------------

        aggregated_parameters, aggregated_metrics = \
            super().aggregate_fit(
                server_round,
                results,
                failures
            )

        # -------------------------------------------------
        # SAVE GLOBAL MODEL
        # -------------------------------------------------

        if aggregated_parameters is not None:

            save_global_model(
                parameters=aggregated_parameters,
                round_num=server_round,
                input_dim=INPUT_SIZE
            )

        # -------------------------------------------------
        # LOG TRAIN METRICS
        # -------------------------------------------------

        if aggregated_metrics:

            print("\n[TRAIN METRICS]")

            for key, value in aggregated_metrics.items():

                print(f"{key}: {value:.4f}")

                mlflow.log_metric(
                    f"train_{key}",
                    value,
                    step=server_round
                )

        print("=" * 70)

        return (
            aggregated_parameters,
            aggregated_metrics
        )

    # =====================================================
    # AGGREGATE EVALUATION
    # =====================================================

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[Tuple],
    ):

        print("\n[EVALUATION]")

        print(f"Clients evaluated: {len(results)}")

        print(f"Failures: {len(failures)}")

        aggregated_loss, aggregated_metrics = \
            super().aggregate_evaluate(
                server_round,
                results,
                failures
            )

        # -------------------------------------------------
        # LOG GLOBAL LOSS
        # -------------------------------------------------

        if aggregated_loss is not None:

            print(
                f"\nGlobal Loss: "
                f"{aggregated_loss:.4f}"
            )

            mlflow.log_metric(
                "global_loss",
                aggregated_loss,
                step=server_round
            )

        # -------------------------------------------------
        # LOG EVALUATION METRICS
        # -------------------------------------------------

        if aggregated_metrics:

            print("\n[GLOBAL METRICS]")

            for key, value in aggregated_metrics.items():

                print(f"{key}: {value:.4f}")

                mlflow.log_metric(
                    key,
                    value,
                    step=server_round
                )

        return (
            aggregated_loss,
            aggregated_metrics
        )


# =========================================================
# MAIN SERVER
# =========================================================

def main():

    print("\n" + "=" * 70)

    print("FEDERATED LEARNING SERVER")

    print("Azure ML + MLflow + Flower")

    print("=" * 70)

    print("\n[CONFIGURATION]")

    print(f"Server Address : {SERVER_ADDRESS}")

    print(f"Federated Rounds : {NUM_ROUNDS}")

    print(f"Minimum Clients : {MIN_CLIENTS}")

    print(f"Input Size : {INPUT_SIZE}")

    print("=" * 70)

    # =====================================================
    # CONNECT TO AZURE ML
    # =====================================================

    print("\n[AZURE ML] Connecting...")

    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    print("[AZURE ML] Connected successfully!")

    # =====================================================
    # CONFIGURE AZURE MLFLOW
    # =====================================================

    tracking_uri = ml_client.workspaces.get(
        WORKSPACE_NAME
    ).mlflow_tracking_uri

    mlflow.set_tracking_uri(
        tracking_uri
    )

    print(f"\n[MLFLOW] Tracking URI:")

    print(tracking_uri)

    # =====================================================
    # CREATE / SET EXPERIMENT
    # =====================================================

    EXPERIMENT_NAME = "Federated-Churn-Training"

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"\n[MLFLOW] Experiment:"
        f" {EXPERIMENT_NAME}"
    )

    # =====================================================
    # START MLFLOW RUN
    # =====================================================

    mlflow.start_run()

    print("[MLFLOW] Run started!")

    # =====================================================
    # LOG PARAMETERS
    # =====================================================

    mlflow.log_param(
        "num_rounds",
        NUM_ROUNDS
    )

    mlflow.log_param(
        "min_clients",
        MIN_CLIENTS
    )

    mlflow.log_param(
        "input_size",
        INPUT_SIZE
    )

    mlflow.log_param(
        "model",
        "ModelA"
    )

    mlflow.log_param(
        "strategy",
        "FedAvg"
    )

    mlflow.log_param(
        "framework",
        "Flower"
    )

    # =====================================================
    # FEDAVG STRATEGY
    # =====================================================

    strategy = FedAvgCustom(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=MIN_CLIENTS,

        min_evaluate_clients=MIN_CLIENTS,

        min_available_clients=MIN_CLIENTS,

        fit_metrics_aggregation_fn=weighted_average,

        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # =====================================================
    # FLOWER SERVER CONFIG
    # =====================================================

    config = fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS
    )

    # =====================================================
    # START SERVER
    # =====================================================

    try:

        print("\n[SERVER] Starting Flower Server...\n")

        fl.server.start_server(

            server_address=SERVER_ADDRESS,

            config=config,

            strategy=strategy,

            grpc_max_message_length=1024 * 1024 * 1024,
        )

    except KeyboardInterrupt:

        print("\n[SERVER] Interrupted manually")

    except Exception as e:

        print(f"\n[ERROR] {e}")

        import traceback

        traceback.print_exc()

    finally:

        # =================================================
        # LOG FINAL MODEL
        # =================================================

        final_model_path = (
            "models/final_model.pth"
        )

        if os.path.exists(final_model_path):

            print(
                "\n[MLFLOW] Uploading final model..."
            )

            mlflow.log_artifact(
                final_model_path
            )

        # =================================================
        # END RUN
        # =================================================

        mlflow.end_run()

        print(
            "\n[MLFLOW] Run closed successfully"
        )

    print("\n" + "=" * 70)

    print("FEDERATED LEARNING COMPLETED!")

    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()