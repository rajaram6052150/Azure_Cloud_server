"""
server.py
=========
Federated Learning Server
Flower + Azure MLflow + Lazy Model Saving
"""

print("1 - Starting imports")

import os

print("2 - os imported")

from typing import List, Tuple

print("3 - typing imported")

import flwr as fl

print("4 - flower imported")

from flwr.server.strategy import FedAvg

print("5 - FedAvg imported")

# =========================================================
# MLFLOW
# =========================================================

print("6 - importing mlflow...")

import mlflow

print("7 - mlflow imported")

# =========================================================
# AZURE ML
# =========================================================

print("8 - importing azure.ai.ml...")

from azure.ai.ml import MLClient

print("9 - azure.ai.ml imported")

from azure.identity import DefaultAzureCredential

print("10 - azure identity imported")

# =========================================================
# LAZY IMPORT PLACEHOLDER
# =========================================================

save_global_model = None

print("11 - lazy import configured")


# =========================================================
# AZURE CONFIG
# =========================================================

SUBSCRIPTION_ID = "0746f4dd-3e9d-4b6c-be49-bdcd4149395f"

RESOURCE_GROUP = "flower-rg"

WORKSPACE_NAME = "flower-wsp"


# =========================================================
# FEDERATED LEARNING CONFIG
# =========================================================

SERVER_ADDRESS = "0.0.0.0:8080"

NUM_ROUNDS = 5

MIN_CLIENTS = 2

INPUT_SIZE = 10


# =========================================================
# METRIC AGGREGATION
# =========================================================

def weighted_average(metrics):

    if not metrics:
        return {}

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    aggregated = {}

    metric_keys = metrics[0][1].keys()

    for key in metric_keys:

        aggregated[key] = sum(
            num_examples * metric_dict[key]
            for num_examples, metric_dict in metrics
        ) / total_examples

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
        server_round,
        results,
        failures,
    ):

        global save_global_model

        print("\n" + "=" * 70)

        print(f"FEDERATED ROUND {server_round}")

        print("=" * 70)

        print(f"✓ Clients completed: {len(results)}")

        print(f"✗ Clients failed: {len(failures)}")

        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures
            )
        )

        # =================================================
        # LAZY IMPORT
        # =================================================

        if save_global_model is None:

            print("\n[MODEL] Importing save_model...")

            from save_model import save_global_model

            print("[MODEL] save_model imported!")

        # =================================================
        # SAVE MODEL
        # =================================================

        if aggregated_parameters is not None:

            print("[MODEL] Saving global model...")

            save_global_model(
                parameters=aggregated_parameters,
                round_num=server_round,
                input_size=INPUT_SIZE
            )

            print("[MODEL] Global model saved!")

        # =================================================
        # TRAIN METRICS
        # =================================================

        if aggregated_metrics:

            print("\n[TRAIN METRICS]")

            for key, value in aggregated_metrics.items():

                print(f"{key}: {value:.4f}")

                try:

                    mlflow.log_metric(
                        f"train_{key}",
                        value,
                        step=server_round
                    )

                except Exception as e:

                    print(
                        f"[MLFLOW ERROR] {e}"
                    )

        print("=" * 70)

        return (
            aggregated_parameters,
            aggregated_metrics
        )

    # =====================================================
    # AGGREGATE EVALUATE
    # =====================================================

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):

        aggregated_loss, aggregated_metrics = (
            super().aggregate_evaluate(
                server_round,
                results,
                failures
            )
        )

        print("\n[EVALUATION]")

        print(f"Clients evaluated: {len(results)}")

        print(f"Failures: {len(failures)}")

        # =================================================
        # LOSS
        # =================================================

        if aggregated_loss is not None:

            print(
                f"\nGlobal Loss:"
                f" {aggregated_loss:.4f}"
            )

            try:

                mlflow.log_metric(
                    "global_loss",
                    aggregated_loss,
                    step=server_round
                )

            except Exception as e:

                print(
                    f"[MLFLOW ERROR] {e}"
                )

        # =================================================
        # METRICS
        # =================================================

        if aggregated_metrics:

            print("\n[GLOBAL METRICS]")

            for key, value in aggregated_metrics.items():

                print(f"{key}: {value:.4f}")

                try:

                    mlflow.log_metric(
                        key,
                        value,
                        step=server_round
                    )

                except Exception as e:

                    print(
                        f"[MLFLOW ERROR] {e}"
                    )

        return (
            aggregated_loss,
            aggregated_metrics
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 70)

    print("FEDERATED LEARNING SERVER")

    print("Flower + Azure ML + MLflow")

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

    try:

        print("\n[AZURE] Creating credentials...")

        credential = DefaultAzureCredential()

        print("[AZURE] Credentials created!")

        print("[AZURE] Connecting ML Client...")

        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME,
        )

        print("[AZURE] ML Client connected!")

        print("[AZURE] Fetching workspace...")

        workspace = ml_client.workspaces.get(
            WORKSPACE_NAME
        )

        print("[AZURE] Workspace fetched!")

        tracking_uri = workspace.mlflow_tracking_uri

        print("[MLFLOW] Tracking URI fetched!")

        print(tracking_uri)

        # =================================================
        # SET TRACKING URI
        # =================================================

        mlflow.set_tracking_uri(
            tracking_uri
        )

        print("[MLFLOW] Tracking URI set!")

    except Exception as e:

        print("\n[AZURE ERROR]")

        print(e)

        import traceback

        traceback.print_exc()

        return

    # =====================================================
    # SET EXPERIMENT
    # =====================================================

    EXPERIMENT_NAME = "Federated-Churn-Training"

    try:

        print("\n[MLFLOW] Setting experiment...")

        mlflow.set_experiment(
            EXPERIMENT_NAME
        )

        print("[MLFLOW] Starting run...")

        mlflow.start_run()

        print("[MLFLOW] Run started!")

    except Exception as e:

        print("\n[MLFLOW ERROR]")

        print(e)

    # =====================================================
    # LOG PARAMETERS
    # =====================================================

    try:

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

        print("[MLFLOW] Parameters logged!")

    except Exception as e:

        print(
            f"[MLFLOW PARAM ERROR] {e}"
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
    # FLOWER CONFIG
    # =====================================================

    config = fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS
    )

    # =====================================================
    # START FLOWER SERVER
    # =====================================================

    try:

        print("\n[FLOWER] Starting Flower server...\n")

        fl.server.start_server(

            server_address=SERVER_ADDRESS,

            config=config,

            strategy=strategy,

            grpc_max_message_length=1024 * 1024 * 1024,
        )

    except KeyboardInterrupt:

        print("\n[SERVER] Interrupted manually")

    except Exception as e:

        print("\n[FLOWER ERROR]")

        print(e)

        import traceback

        traceback.print_exc()

    finally:

        print("\n[SERVER] Cleaning up...")

        # =================================================
        # LOG FINAL MODEL
        # =================================================

        final_model_path = (
            "models/federated_latest.pth"
        )

        if os.path.exists(final_model_path):

            try:

                mlflow.log_artifact(
                    final_model_path
                )

                print(
                    "[MLFLOW] Final model uploaded!"
                )

            except Exception as e:

                print(
                    f"[MLFLOW ARTIFACT ERROR] {e}"
                )

        # =================================================
        # END RUN
        # =================================================

        try:

            mlflow.end_run()

            print("[MLFLOW] Run ended!")

        except Exception as e:

            print(
                f"[MLFLOW END ERROR] {e}"
            )

    print("\n" + "=" * 70)

    print("FEDERATED LEARNING COMPLETED!")

    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()