"""
server.py
=========
Federated Learning Server with MLflow Tracking
Coordinates training across Client A and Client B using Flower Framework
Aggregates model updates using FedAvg algorithm
"""

import os
from typing import List, Tuple

import flwr as fl
from flwr.server.strategy import FedAvg

import mlflow

from save_model import save_global_model


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_SIZE = 10

NUM_ROUNDS = 5


# =========================================================
# METRIC AGGREGATION FUNCTION
# =========================================================

def weighted_average(metrics):
    """
    Aggregate metrics from multiple clients using weighted average.
    """

    if not metrics:
        return {}

    aggregated = {}

    first_metrics = metrics[0][1]

    if not first_metrics:
        return {}

    metric_keys = first_metrics.keys()

    for key in metric_keys:

        weighted_sum = sum(
            num_examples * m.get(key, 0)
            for num_examples, m in metrics
        )

        total_examples = sum(
            num_examples
            for num_examples, _ in metrics
        )

        aggregated[key] = weighted_sum / total_examples

    return aggregated


# =========================================================
# CUSTOM FEDAVG STRATEGY
# =========================================================

class FedAvgCustom(FedAvg):
    """
    Custom FedAvg Strategy with:
    - Detailed logging
    - Global model saving
    - MLflow tracking
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    # =====================================================
    # AGGREGATE TRAINING
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
        # PRINT FAILURES
        # -------------------------------------------------

        if failures:

            print("\n[FAILURES]")

            for client_id, error in failures:

                print(f"Client {client_id}: {error}")

        # -------------------------------------------------
        # FEDAVG AGGREGATION
        # -------------------------------------------------

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
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
        # LOG TRAINING METRICS TO MLFLOW
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

        return aggregated_parameters, aggregated_metrics

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

        if failures:

            print(f"Clients failed: {len(failures)}")

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round,
            results,
            failures
        )

        # -------------------------------------------------
        # LOG LOSS
        # -------------------------------------------------

        if aggregated_loss is not None:

            print(f"\nGlobal Loss: {aggregated_loss:.4f}")

            mlflow.log_metric(
                "global_loss",
                aggregated_loss,
                step=server_round
            )

        # -------------------------------------------------
        # LOG METRICS
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

        return aggregated_loss, aggregated_metrics


# =========================================================
# MAIN SERVER FUNCTION
# =========================================================

def main():

    print("\n" + "=" * 70)

    print("FEDERATED LEARNING SERVER WITH MLFLOW")

    print("=" * 70)

    print("\n[CONFIGURATION]")

    print(f"Server Address      : 0.0.0.0:8080")

    print(f"Federated Rounds    : {NUM_ROUNDS}")

    print(f"Minimum Clients     : 2")

    print(f"Input Feature Size  : {INPUT_SIZE}")

    print(f"Model Architecture  : ModelA")

    print("=" * 70)

    # =====================================================
    # MLFLOW SETUP
    # =====================================================

    print("\n[MLFLOW] Initializing experiment tracking...")

    mlflow.set_experiment(
        "Federated-Churn-Training"
    )

    mlflow.start_run()

    # -----------------------------------------------------
    # LOG PARAMETERS
    # -----------------------------------------------------

    mlflow.log_param(
        "num_rounds",
        NUM_ROUNDS
    )

    mlflow.log_param(
        "input_size",
        INPUT_SIZE
    )

    mlflow.log_param(
        "min_clients",
        2
    )

    mlflow.log_param(
        "model",
        "ModelA"
    )

    mlflow.log_param(
        "optimizer",
        "Adam"
    )

    mlflow.log_param(
        "federated_strategy",
        "FedAvg"
    )

    print("[MLFLOW] Tracking started!")

    # =====================================================
    # FEDERATED STRATEGY
    # =====================================================

    strategy = FedAvgCustom(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=2,

        min_evaluate_clients=2,

        min_available_clients=2,

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

        print("\n[SERVER] Starting Flower Server...\n")

        fl.server.start_server(

            server_address="0.0.0.0:8080",

            config=config,

            strategy=strategy,

            grpc_max_message_length=1024 * 1024 * 1024,
        )

    except KeyboardInterrupt:

        print("\n[SERVER] Interrupted by user")

    except Exception as e:

        print(f"\n[ERROR] {type(e).__name__}: {e}")

        import traceback

        traceback.print_exc()

    finally:

        # =================================================
        # SAVE FINAL MODEL TO MLFLOW
        # =================================================

        final_model_path = "models/final_model.pth"

        if os.path.exists(final_model_path):

            print("\n[MLFLOW] Uploading final model artifact...")

            mlflow.log_artifact(final_model_path)

        # =================================================
        # END MLFLOW RUN
        # =================================================

        mlflow.end_run()

        print("\n[MLFLOW] Run closed successfully")

    print("\n" + "=" * 70)

    print("FEDERATED LEARNING COMPLETED!")

    print(f"Final model saved at: {final_model_path}")

    print("MLflow experiment logged successfully!")

    print("=" * 70 + "\n")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()