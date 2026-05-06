"""
server.py
=========
Federated Learning Server
Coordinates training across Client A and Client B using Flower Framework
Aggregates model updates using FedAvg algorithm
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from typing import List, Tuple
import os

from save_model import save_global_model


# IMPORTANT: Update this based on your actual dataset features
# Calculate INPUT_SIZE by running preprocessing on your data first:
# python -c "from utils import get_input_size; print(get_input_size('FL/data/client_A.csv'))"
INPUT_SIZE = 10  # Updated based on feature dimension after preprocessing


class FedAvgCustom(FedAvg):
    """
    Custom FedAvg strategy with detailed logging
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_num = 0
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[Tuple],
    ):
        """
        Aggregate model weights using FedAvg and save global model after each round.
        
        Args:
            server_round: Current round number
            results: List of (client_id, fit_res) tuples from successful clients
            failures: List of (client_id, error) tuples from failed clients
        
        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
        """
        
        print(f"\n{'=' * 70}")
        print(f"Federated Round {server_round}")
        print(f"{'=' * 70}")
        
        print(f"✓ Clients completed: {len(results)}")
        print(f"✗ Clients failed: {len(failures)}")
        
        # Print failures if any
        if failures:
            print("\n[FAILURES]")
            for client_id, error in failures:
                print(f"  Client {client_id}: {error}")
        
        # Perform FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures
        )
        
        # Save aggregated global model
        if aggregated_parameters is not None:
            save_global_model(
                parameters=aggregated_parameters,
                round_num=server_round,
                input_size=INPUT_SIZE
            )
        
        # Print metrics if available
        if aggregated_metrics:
            print(f"\n[AGGREGATED METRICS]")
            for key, value in aggregated_metrics.items():
                print(f"  {key}: {value}")
        
        print(f"{'=' * 70}\n")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[Tuple],
    ):
        """
        Aggregate evaluation metrics from clients.
        """
        
        print(f"[EVALUATION Round {server_round}]")
        print(f"  Clients evaluated: {len(results)}")
        
        if failures:
            print(f"  Clients failed: {len(failures)}")
        
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round,
            results,
            failures
        )
        
        if aggregated_metrics:
            print(f"  Global Loss: {aggregated_loss:.4f}")
            for key, value in aggregated_metrics.items():
                print(f"  {key}: {value}")
        
        return aggregated_loss, aggregated_metrics


def main():
    """Start Flower Federated Learning Server"""
    
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING SERVER")
    print("=" * 70)
    
    print("\n[Configuration]")
    print(f"  Server Address: 0.0.0.0:8080")
    print(f"  Federated Rounds: 5")
    print(f"  Min Clients Required: 2")
    print(f"  Input Feature Size: {INPUT_SIZE}")
    print(f"  Model Architecture: [Linear({INPUT_SIZE}, 64) -> BatchNorm -> ReLU -> Dropout]")
    print(f"                      [Linear(64, 32) -> BatchNorm -> ReLU -> Dropout]")
    print(f"                      [Linear(32, 1) -> Sigmoid]")
    
    print("\n" + "=" * 70)
    
    # Define Federated Learning Strategy
    strategy = FedAvgCustom(
        # Use all available clients for training
        fraction_fit=1.0,
        
        # Evaluate on all available clients
        fraction_evaluate=1.0,
        
        # Minimum clients required for training
        min_fit_clients=2,
        
        # Minimum clients required for evaluation
        min_evaluate_clients=2,
        
        # Wait until at least 2 clients connect
        min_available_clients=2,
    )
    
    # Create server config
    config = fl.server.ServerConfig(
        num_rounds=5
    )
    
    # Start Flower Server
    try:
        print("\n[SERVER] Starting Flower server...\n")
        
        fl.server.start_server(
            # 0.0.0.0 allows external Azure/public access
            server_address="0.0.0.0:8080",
            
            # Number of federated rounds
            config=config,
            
            strategy=strategy,
            
            # Prevent gRPC message size issues with large models
            grpc_max_message_length=1024 * 1024 * 1024,
        )
    
    except KeyboardInterrupt:
        print("\n\n[SERVER] Interrupted by user")
    
    except Exception as e:
        print(f"\n\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING COMPLETED!")
    print("Final model saved in: models/federated_latest.pth")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
