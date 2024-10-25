import jax
import jax.numpy as jnp
import torch
import time
import pandas as pd
from typing import Callable, Any, Tuple
import numpy as np

# [Previous helper functions remain the same until run_benchmarks]

class ComplexTorchModule(torch.nn.Module):
    """PyTorch module for complex computation."""
    def __init__(self, n_ops=10):
        super().__init__()
        self.n_ops = n_ops
        
    def forward(self, x, y):
        result = torch.zeros_like(x)
        for _ in range(self.n_ops):
            result = result + x * y * torch.sin(x) + torch.exp(y) * torch.cos(x)
        return result

def complex_jax_fn(x, y, n_ops=10):
    """More complex JAX computation."""
    result = jnp.zeros_like(x)
    for _ in range(n_ops):
        result = result + x * y * jnp.sin(x) + jnp.exp(y) * jnp.cos(x)
    return result

def run_benchmarks(sizes: list[int], n_iter: int = 1000, n_ops: int = 10) -> pd.DataFrame:
    """Run complete benchmark suite."""
    results = []
    
    # JIT compile JAX function
    jax_jitted = jax.jit(lambda x, y: complex_jax_fn(x, y, n_ops))

    # Create PyTorch module
    torch_module = ComplexTorchModule(n_ops)

    # Check available devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # If GPU available, move JAX to it
    if device == 'cuda':
        jax.config.update('jax_platform_name', 'gpu')
    
    for size in sizes:
        print(f"\nBenchmarking with input size: {size}")
        
        # Create test data
        x_jax, y_jax, x_torch, y_torch = create_test_data(size, device)
        
        # Create PyTorch JIT functions
        torch_traced = torch.jit.trace(torch_module, (x_torch, y_torch))
        torch_scripted = torch.jit.script(torch_module)
        
        # Calculate estimated FLOPS
        flops = estimate_flops(size, n_ops)
        
        # Run benchmarks
        jax_time = benchmark_jax(jax_jitted, x_jax, y_jax, n_iter)
        trace_time = benchmark_torch(torch_traced, x_torch, y_torch, n_iter, device)
        script_time = benchmark_torch(torch_scripted, x_torch, y_torch, n_iter, device)
        
        # Calculate GFLOPS
        jax_gflops = (flops * 1e-9) / jax_time
        trace_gflops = (flops * 1e-9) / trace_time
        script_gflops = (flops * 1e-9) / script_time
        
        # Store results
        results.extend([
            {'size': size, 'framework': 'JAX (jit)', 
             'time': jax_time, 'GFLOPS': jax_gflops},
            {'size': size, 'framework': 'PyTorch (trace)', 
             'time': trace_time, 'GFLOPS': trace_gflops},
            {'size': size, 'framework': 'PyTorch (script)', 
             'time': script_time, 'GFLOPS': script_gflops}
        ])
        
        # Print immediate results
        print(f"Input size: {size}, Estimated FLOPS per iteration: {flops}")
        print(f"JAX (jit): {jax_time*1e6:.2f} µs ({jax_gflops:.2f} GFLOPS)")
        print(f"PyTorch (trace): {trace_time*1e6:.2f} µs ({trace_gflops:.2f} GFLOPS)")
        print(f"PyTorch (script): {script_time*1e6:.2f} µs ({script_gflops:.2f} GFLOPS)")
    
    return pd.DataFrame(results)

def estimate_flops(size: int, n_ops: int = 1) -> int:
    """
    Estimate FLOPs for our computation.
    For each element we perform:
    - Multiple matrix multiplications (each is 2 FLOPs per element)
    - Multiple trigonometric operations (approx 15 FLOPs each)
    - Multiple additions/subtractions (1 FLOP each)
    - Multiple exponentials (approx 20 FLOPs each)
    """
    flops_per_element = (
        n_ops * (  # Number of repeated operations
            2 +    # Multiplication
            15 +   # sin
            20 +   # exp
            1 +    # addition
            15 +   # cos
            1      # final addition
        )
    )
    return size * flops_per_element

def create_test_data(size: int, device: str) -> Tuple[Any, Any, Any, Any]:
    """Create test data for both frameworks."""
    # JAX data
    key = jax.random.PRNGKey(0)
    x_jax = jax.random.normal(key, (size,))
    y_jax = jax.random.normal(key, (size,))
    
    # PyTorch data
    x_torch = torch.randn(size, device=device)
    y_torch = torch.randn(size, device=device)
    
    return x_jax, y_jax, x_torch, y_torch

def benchmark_jax(fn: Callable, x: Any, y: Any, n_iter: int = 1000) -> float:
    """Benchmark a JAX function."""
    # Warmup
    _ = fn(x, y).block_until_ready()
    
    start = time.perf_counter()
    for _ in range(n_iter):
        result = fn(x, y)
        result.block_until_ready()
    end = time.perf_counter()
    
    return (end - start) / n_iter

def benchmark_torch(fn: Callable, x: torch.Tensor, y: torch.Tensor, 
                   n_iter: int = 1000, device: str = 'cpu') -> float:
    """Benchmark a PyTorch function."""
    # Warmup
    _ = fn(x, y)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = fn(x, y)
        if device == 'cuda':
            torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / n_iter

def plot_results(results_df: pd.DataFrame):
    """Create visualization of results."""
    try:
        import matplotlib.pyplot as plt
        
        # Time plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for framework in results_df['framework'].unique():
            data = results_df[results_df['framework'] == framework]
            plt.plot(data['size'], data['time'] * 1e6, marker='o', label=framework)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Input Size')
        plt.ylabel('Time per iteration (µs)')
        plt.title('Performance Comparison: Time')
        plt.legend()
        plt.grid(True)
        
        # GFLOPS plot
        plt.subplot(1, 2, 2)
        for framework in results_df['framework'].unique():
            data = results_df[results_df['framework'] == framework]
            plt.plot(data['size'], data['GFLOPS'], marker='o', label=framework)
        
        plt.xscale('log')
        plt.xlabel('Input Size')
        plt.ylabel('GFLOPS')
        plt.title('Performance Comparison: GFLOPS')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")

def main():
    # Test with different input sizes and more operations
    sizes = [100, 1000, 10000, 100000, 1000000, 10000000]
    n_iter = 1000
    n_ops = 20  # Increased number of operations
    
    # Run benchmarks
    results_df = run_benchmarks(sizes, n_iter, n_ops)
    
    # Print summary
    print("\nTime summary (seconds):")
    print(results_df.groupby(['size', 'framework'])['time'].mean().unstack())
    print("\nGFLOPS summary:")
    print(results_df.groupby(['size', 'framework'])['GFLOPS'].mean().unstack())
    
    # Plot results
    plot_results(results_df)

if __name__ == "__main__":
    main()
