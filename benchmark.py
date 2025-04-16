import numpy as np
from mpi4py import MPI
from rng import get_rng, rng_context, register_rng
from mpiwrapper import mpi
from moe import SimpleMoE, MoE_EP, MoE_TP
import time
import tracemalloc
import argparse
import timeit
from functools import partial

def run_moe_forward(moe, X):
    """
    Run a forward pass of the MoE model.
    This function will be timed with timeit.
    """
    return moe(X)

def run_moe(
    moe_type="tp",
    batch_size=8,
    feature_dim=32,
    hidden_dim=128,
    output_dim=64,
    num_experts=None,
    topk=2,
    number=3,  # Number of times to run the forward pass for timing
    repeat=3   # Number of times to repeat the timer (taking the best time)
):
    """
    Unified function to run different types of MoE models and measure performance.

    Args:
        moe_type: Type of MoE ("simple", "ep", or "tp")
        batch_size: Number of samples in the batch
        feature_dim: Dimension of input features
        hidden_dim: Hidden dimension for experts
        output_dim: Output dimension
        num_experts: Number of experts (defaults to MPI world size)
        topk: Number of experts to route each input to
        number: Number of times to execute the statement for each timing run
        repeat: Number of times to repeat the timer (taking the best time)
    Returns:
        Dictionary with min duration and max memory usage (on rank 0)
    """
    if num_experts is None:
        num_experts = mpi.get_size()

    # Handle Simple MoE running only on rank 0
    if moe_type == "simple" and mpi.get_rank() != 0:
        min_duration_ms = 0
        peak_memory = 0
    else:
        # Generate input data
        if moe_type != "simple":
            # Synchronize input across all ranks for "ep" and "tp"
            if mpi.get_rank() == 0:
                X = get_rng().randn(batch_size, feature_dim)
            else:
                X = None
            X = mpi.comm.bcast(X, root=0)
        else:
            # For "simple", only rank 0 generates X
            X = get_rng().randn(batch_size, feature_dim)

        # Create MoE model
        model_class = {
            "simple": SimpleMoE,
            "ep": MoE_EP,
            "tp": MoE_TP
        }.get(moe_type, MoE_TP)
        
        moe = model_class(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            topk=topk
        )

        # Warm up
        _ = moe(X)

        # Measure time and memory
        tracemalloc.reset_peak()
        
        # Use timeit for more accurate timing
        forward_func = partial(run_moe_forward, moe, X)
        
        # Run timeit multiple times and take the best time
        timer = timeit.Timer(forward_func)
        times = timer.repeat(repeat=repeat, number=number)
        
        # Calculate the minimum duration in milliseconds
        # (minimum is standard practice with timeit to reduce noise)
        min_duration_sec = min(times) / number  # Average over 'number' executions
        min_duration_ms = min_duration_sec * 1000
        
        # Get peak memory
        _, peak_memory = tracemalloc.get_traced_memory()
        peak_memory /= 1024 * 1024  # Convert to MB

    # Gather metrics across all ranks
    all_durations = mpi.comm.gather(min_duration_ms, root=0)
    all_memories = mpi.comm.gather(peak_memory, root=0)

    if mpi.get_rank() == 0:
        max_duration = max(all_durations)
        max_memory = max(all_memories)
        return {
            "min_duration_ms": min_duration_ms,  # Best time from this rank
            "max_duration_ms": max_duration,     # Worst time across all ranks
            "max_memory_mb": max_memory
        }
    return None

def benchmark_batch_size(number=3, repeat=3):
    batch_sizes = [8, 16, 32, 64]
    results = []
    for batch_size in batch_sizes:
        for moe_type in ["simple", "tp", "ep"]:
            result = run_moe(moe_type=moe_type, batch_size=batch_size, feature_dim=32, 
                          topk=2, number=number, repeat=repeat)
            if mpi.get_rank() == 0 and result:
                results.append({
                    "moe_type": moe_type,
                    "batch_size": batch_size,
                    "feature_dim": 32,
                    "topk": 2,
                    "min_duration_ms": result["min_duration_ms"],
                    "max_duration_ms": result["max_duration_ms"],
                    "max_memory_mb": result["max_memory_mb"]
                })
    if mpi.get_rank() == 0:
        print("\nBenchmark results for varying batch_size:")
        print("moe_type | batch_size | min_duration_ms | max_duration_ms | max_memory_mb")
        print("-" * 80)
        for res in results:
            print(f"{res['moe_type']:7} | {res['batch_size']:10} | {res['min_duration_ms']:14.2f} | "
                  f"{res['max_duration_ms']:14.2f} | {res['max_memory_mb']:13.2f}")

def benchmark_feature_dim(number=3, repeat=3):
    feature_dims = [32, 64, 128, 256]
    results = []
    for feature_dim in feature_dims:
        for moe_type in ["simple", "tp", "ep"]:
            result = run_moe(moe_type=moe_type, batch_size=8, feature_dim=feature_dim, 
                          topk=2, number=number, repeat=repeat)
            if mpi.get_rank() == 0 and result:
                results.append({
                    "moe_type": moe_type,
                    "batch_size": 8,
                    "feature_dim": feature_dim,
                    "topk": 2,
                    "min_duration_ms": result["min_duration_ms"],
                    "max_duration_ms": result["max_duration_ms"],
                    "max_memory_mb": result["max_memory_mb"]
                })
    if mpi.get_rank() == 0:
        print("\nBenchmark results for varying feature_dim:")
        print("moe_type | feature_dim | min_duration_ms | max_duration_ms | max_memory_mb")
        print("-" * 80)
        for res in results:
            print(f"{res['moe_type']:7} | {res['feature_dim']:11} | {res['min_duration_ms']:14.2f} | "
                  f"{res['max_duration_ms']:14.2f} | {res['max_memory_mb']:13.2f}")

def benchmark_topk(number=3, repeat=3):
    topks = [1, 2, 4]  # Assuming num_experts >= 4 (e.g., mpirun -np 4)
    results = []
    for topk in topks:
        for moe_type in ["simple", "tp", "ep"]:
            result = run_moe(moe_type=moe_type, batch_size=8, feature_dim=32, 
                          topk=topk, number=number, repeat=repeat)
            if mpi.get_rank() == 0 and result:
                results.append({
                    "moe_type": moe_type,
                    "batch_size": 8,
                    "feature_dim": 32,
                    "topk": topk,
                    "min_duration_ms": result["min_duration_ms"],
                    "max_duration_ms": result["max_duration_ms"],
                    "max_memory_mb": result["max_memory_mb"]
                })
    if mpi.get_rank() == 0:
        print("\nBenchmark results for varying topk:")
        print("moe_type | topk | min_duration_ms | max_duration_ms | max_memory_mb")
        print("-" * 80)
        for res in results:
            print(f"{res['moe_type']:7} | {res['topk']:4} | {res['min_duration_ms']:14.2f} | "
                  f"{res['max_duration_ms']:14.2f} | {res['max_memory_mb']:13.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MoE implementations")
    parser.add_argument("--benchmark_type", type=str, choices=["batch_size", "feature_dim", "topk"], 
                        required=True, help="Type of benchmark to run")
    parser.add_argument("--number", type=int, default=3, 
                        help="Number of times to execute the statement for each timing run")
    parser.add_argument("--repeat", type=int, default=3, 
                        help="Number of times to repeat the timer (taking the best time)")
    args = parser.parse_args()

    # Start memory tracing
    tracemalloc.start()

    # Run the specified benchmark
    if args.benchmark_type == "batch_size":
        benchmark_batch_size(number=args.number, repeat=args.repeat)
    elif args.benchmark_type == "feature_dim":
        benchmark_feature_dim(number=args.number, repeat=args.repeat)
    elif args.benchmark_type == "topk":
        benchmark_topk(number=args.number, repeat=args.repeat)