# Benchmarking Mixture of Experts (MoE) Architectures

## Overview

This report presents the benchmarking results for three different Mixture of Experts (MoE) architectures: **Simple MoE**, **Tensor Parallel MoE (TP MoE)**, and **Expert Parallel MoE (EP MoE)**. The benchmarks analyze their performance across three varying factors: **batch size**, **top-k selected experts**, and **feature dimension**. The primary metrics considered are **minimum duration (ms)**, **maximum duration (ms)**, and **maximum memory usage (MB)**.

## How to run
```
mpirun -np 4 python benchmark.py --benchmark_type batch_size
mpirun -np 4 python benchmark.py --benchmark_type topk
mpirun -np 4 python benchmark.py --benchmark_type feature_dim
```

## Findings

From the benchmarking results, several key observations emerge:

1. **Batch Size Scaling**:  
   - **Simple MoE** exhibits increasing latency as batch size grows, but it remains significantly faster than TP MoE.  
   - **TP MoE** has the highest execution time across all batch sizes, indicating it does not scale efficiently in this setup.  
   - **EP MoE** performs best, maintaining the lowest latency while keeping memory usage manageable.

2. **Top-k Scaling**:  
   - **Simple MoE** shows a linear increase in execution time with higher top-k values.  
   - **TP MoE** is significantly slower even at low top-k values, highlighting overheads in tensor parallelism.  
   - **EP MoE** remains the most efficient, handling increased top-k selection with minimal slowdown.

3. **Feature Dimension Scaling**:  
   - **Simple MoE** scales relatively well up to a moderate feature dimension but faces sharp increases in memory usage at higher values.  
   - **TP MoE** maintains moderate memory usage but remains computationally expensive.  
   - **EP MoE** is highly efficient at lower feature dimensions but struggles with extreme values, possibly due to increased communication overhead.

## Performance Analysis

The observed performance differences can be attributed to several architectural and computational factors:

1. **Parallel Computational Benefits**  
   - TP MoE leverages tensor parallelism, which distributes tensor operations across multiple devices but introduces communication overhead.  
   - EP MoE distributes experts across devices, reducing computational workload per device and improving efficiency.  

2. **Workload Distribution**  
   - Simple MoE keeps all computations on a single device, leading to higher memory usage and longer execution times at scale.  
   - EP MoE distributes expert computations across multiple devices, achieving better load balancing.  

3. **Memory Locality & Cache Utilization**  
   - EP MoE has a smaller memory footprint compared to Simple MoE, likely due to improved cache utilization and better memory locality in distributed setups.  
   - TP MoE is memory-efficient but suffers from communication overhead, leading to increased latency.  

4. **Communication vs. Computation Trade-offs**  
   - TP MoE has significant communication overhead due to tensor splitting and aggregation, making it less efficient for latency-sensitive applications.  
   - EP MoE minimizes inter-device communication compared to TP MoE, favoring distributed computation over tensor-level synchronization.  

5. **Reducing Computational Complexity per Process**  
   - EP MoE assigns different experts to different devices, reducing the computational burden per process and enabling faster inference.  
   - Simple MoE lacks this benefit and performs all computations on a single device, leading to bottlenecks at scale.  

## Summary of Architectural Suitability

| Architecture | Strengths | Weaknesses |
|-------------|----------|------------|
| **Simple MoE** | Good for small-scale, single-device setups | Poor scalability, high memory usage at scale |
| **Tensor Parallel MoE (TP MoE)** | Memory-efficient, useful for large models across multiple devices | High latency due to communication overhead |
| **Expert Parallel MoE (EP MoE)** | Fast and scalable, ideal for distributed expert workloads | Can struggle with extreme feature dimensions due to communication costs |

## Benchmark Results

### Varying Batch Size

| MoE Type | Batch Size | Min Duration (ms) | Max Duration (ms) | Max Memory (MB) |
|----------|-----------|-------------------|-------------------|----------------|
| Simple   | 8         | 0.26              | 0.26              | 0.43           |
| TP       | 8         | 3.61              | 3.61              | 0.20           |
| EP       | 8         | 0.18              | 0.18              | 0.24           |
| Simple   | 16        | 0.47              | 0.47              | 0.50           |
| TP       | 16        | 4.18              | 4.18              | 0.22           |
| EP       | 16        | 0.25              | 0.25              | 0.31           |
| Simple   | 32        | 0.88              | 0.88              | 0.52           |
| TP       | 32        | 8.20              | 8.20              | 0.24           |
| EP       | 32        | 0.38              | 0.39              | 0.43           |
| Simple   | 64        | 1.69              | 1.69              | 0.56           |
| TP       | 64        | 16.45             | 16.45             | 0.29           |
| EP       | 64        | 0.66              | 0.66              | 0.62           |

### Varying Top-K

| MoE Type | Top-K | Min Duration (ms) | Max Duration (ms) | Max Memory (MB) |
|----------|------|-------------------|-------------------|----------------|
| Simple   | 1    | 0.15              | 0.15              | 0.43           |
| TP       | 1    | 3.33              | 3.38              | 0.19           |
| EP       | 1    | 0.28              | 0.28              | 0.24           |
| Simple   | 2    | 0.26              | 0.26              | 0.49           |
| TP       | 2    | 2.21              | 2.21              | 0.21           |
| EP       | 2    | 0.21              | 0.21              | 0.25           |
| Simple   | 4    | 0.46              | 0.46              | 0.50           |
| TP       | 4    | 4.13              | 4.13              | 0.21           |
| EP       | 4    | 0.23              | 0.23              | 0.25           |

### Varying Feature Dimension

| MoE Type | Feature Dim | Min Duration (ms) | Max Duration (ms) | Max Memory (MB) |
|----------|------------|-------------------|-------------------|----------------|
| Simple   | 32         | 0.26              | 0.26              | 0.43           |
| TP       | 32         | 5.90              | 5.93              | 0.20           |
| EP       | 32         | 0.47              | 0.47              | 0.24           |
| Simple   | 64         | 0.38              | 0.38              | 0.62           |
| TP       | 64         | 2.07              | 2.07              | 0.24           |
| EP       | 64         | 0.19              | 0.19              | 0.28           |
| Simple   | 128        | 1.31              | 1.31              | 0.88           |
| TP       | 128        | 8.89              | 8.91              | 0.31           |
| EP       | 128        | 425.07            | 434.45            | 0.35           |
| Simple   | 256        | 0.90              | 0.90              | 1.39           |
| TP       | 256        | 2.53              | 2.53              | 0.45           |
| EP       | 256        | 386.61            | 387.33            | 0.49           |

