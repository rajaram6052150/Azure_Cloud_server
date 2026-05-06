"""
server.py
=========
Minimal Debug Version
Flower + Azure MLflow Debuggable Server
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

# ---------------------------------------------------------
# MLFLOW IMPORT
# ---------------------------------------------------------

print("6 - importing mlflow...")

import mlflow

print("7 - mlflow imported")

# ---------------------------------------------------------
# AZURE IMPORTS
# ---------------------------------------------------------

print("8 - importing azure.ai.ml...")

from azure.ai.ml import MLClient

print("9 - azure.ai.ml imported")

from azure.identity import DefaultAzureCredential

print("10 - azure identity imported")

from save_model import save_global_model

print("11 - save_model imported")


# =========================================================
# AZURE CONFIG
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

    if not metrics:
        return {}

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    aggregated = {}

    keys = metrics[0][1].keys()

    for key in keys:

        aggregated[key] = sum(
            num_examples * metric_dict[key]
            for num_examples, metric_dict in metrics
        ) / total_examples

    return aggregated


# =========================================================
# CUSTOM STRATEGY
# =========================================================

class FedAvgCustom(FedAvg):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        print("\n" + "=" * 60)

        print(f"ROUND {server_round}")

        print("=" * 60)

        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures
            )
        )

        if aggregated_parameters is not None:

            print("[MODEL] Saving global model...")

            save_global_model(
                parameters=aggregated_parameters,
                round_num=server_round,
                input_dim=INPUT_SIZE
            )

            print("[MODEL] Saved!")

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

        return (
            aggregated_parameters,
            aggregated_metrics
        )

    # -----------------------------------------------------

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

        if aggregated_loss is not None:

            print(
                f"Global Loss:"
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

    print("=" * 70)

    # =====================================================
    # AZURE CONNECTION
    # =====================================================

    try:

        print("\n[AZURE] Creating credential...")

        credential = DefaultAzureCredential()

        print("[AZURE] Credential created!")

        print("[AZURE] Creating ML client...")

        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME,
        )

        print("[AZURE] ML client created!")

        print("[AZURE] Fetching workspace...")

        workspace = ml_client.workspaces.get(
            WORKSPACE_NAME
        )

        print("[AZURE] Workspace fetched!")

        tracking_uri = (
            workspace.mlflow_tracking_uri
        )

        print("[MLFLOW] URI fetched!")

        print(tracking_uri)

        # -------------------------------------------------
        # SET TRACKING URI
        # -------------------------------------------------

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
    # EXPERIMENT
    # =====================================================

    try:

        print("\n[MLFLOW] Setting experiment...")

        mlflow.set_experiment(
            "Federated-Churn-Training"
        )

        print("[MLFLOW] Starting run...")

        mlflow.start_run()

        print("[MLFLOW] Run started!")

    except Exception as e:

        print("\n[MLFLOW ERROR]")

        print(e)

    # =====================================================
    # LOG PARAMS
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

        print("[MLFLOW] Params logged!")

    except Exception as e:

        print(
            f"[MLFLOW PARAM ERROR] {e}"
        )

    # =====================================================
    # STRATEGY
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
    # SERVER CONFIG
    # =====================================================

    config = fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS
    )

    # =====================================================
    # START FLOWER SERVER
    # =====================================================

    try:

        print("\n[FLOWER] Starting server...\n")

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

        final_model_path = (
            "models/final_model.pth"
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

        try:

            mlflow.end_run()

            print("[MLFLOW] Run ended!")

        except Exception as e:

            print(
                f"[MLFLOW END ERROR] {e}"
            )

    print("\nSERVER FINISHED")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()