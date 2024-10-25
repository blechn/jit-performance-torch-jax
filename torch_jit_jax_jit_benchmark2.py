import jax
import jax.numpy as jnp
import torch
import time
import pandas as pd
from typing import Callable, Any, Tuple
import numpy as np

def estimate_matmul_flops(shape_a: tuple, shape_b: tuple) -> int:
    """
    Estimate FLOPs for matrix multiplication.
    For matrix multiplication of (m,n) × (n,p): m*n*p*2 FLOPs
    For batched operation, multiply by batch size
    """
    if len(shape_a) == 2:  # Regular matrix multiplication
        m, n = shape_a
        _, p = shape_b
        return m * n * p * 2
    else:  # Batched tensor multiplication
        batch, m, n = shape_a
        _, _, p = shape_b
        return batch * m * n * p * 2

def jax_complex_op(x, y):
    """Complex operation with proper transpose handling for both batched and non-batched cases."""
    if len(x.shape) == 2:  # Non-batched case
        return jnp.matmul(x, y) + jnp.matmul(y, x.T)
    else:  # Batched case
        return jnp.matmul(x, y) + jnp.matmul(y, jnp.transpose(x, axes=(0, 2, 1)))

class TensorOps(torch.nn.Module):
    """PyTorch module for tensor operations."""
    def __init__(self, op_type='matmul'):
        super().__init__()
        self.op_type = op_type
        
    def forward(self, x, y):
        if self.op_type == 'matmul':
            return torch.matmul(x, y)
        elif self.op_type == 'batch_matmul':
            return torch.matmul(x, y)
        elif self.op_type == 'complex':
            if len(x.shape) == 2:  # Non-batched case
                return torch.matmul(x, y) + torch.matmul(y, x.T)
            else:  # Batched case
                return torch.matmul(x, y) + torch.matmul(y, x.transpose(-2, -1))

def create_matrix_data(shape_a: tuple, shape_b: tuple, device: str) -> Tuple[Any, Any, Any, Any]:
    """Create test data for both frameworks."""
    # JAX data
    key = jax.random.PRNGKey(0)
    x_jax = jax.random.normal(key, shape_a)
    y_jax = jax.random.normal(key, shape_b)
    
    # PyTorch data
    x_torch = torch.randn(*shape_a, device=device)
    y_torch = torch.randn(*shape_b, device=device)
    
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

def run_tensor_benchmarks(shapes: list[tuple], n_iter: int = 1000) -> pd.DataFrame:
    """Run tensor operation benchmarks."""
    results = []
    
    # Check available devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # If GPU available, move JAX to it
    if device == 'cuda':
        jax.config.update('jax_platform_name', 'gpu')
    
    for shape_info in shapes:
        shape_a, shape_b = shape_info
        print(f"\nBenchmarking with shapes: {shape_a} × {shape_b}")
        
        # Create test data
        x_jax, y_jax, x_torch, y_torch = create_matrix_data(shape_a, shape_b, device)
        
        # Define and compile operations
        # Regular matrix multiplication
        jax_matmul = jax.jit(lambda x, y: jnp.matmul(x, y))
        torch_matmul = torch.jit.script(TensorOps('matmul'))
        
        # Complex operation (A@B + B@A^T)
        jax_complex = jax.jit(jax_complex_op)
        torch_complex = torch.jit.script(TensorOps('complex'))
        
        # Calculate estimated FLOPS
        basic_flops = estimate_matmul_flops(shape_a, shape_b)
        complex_flops = basic_flops * 2  # Two matrix multiplications
        
        # Run benchmarks
        # Regular matmul
        jax_time = benchmark_jax(jax_matmul, x_jax, y_jax, n_iter)
        torch_time = benchmark_torch(torch_matmul, x_torch, y_torch, n_iter, device)
        
        # Complex operation
        jax_complex_time = benchmark_jax(jax_complex, x_jax, y_jax, n_iter)
        torch_complex_time = benchmark_torch(torch_complex, x_torch, y_torch, n_iter, device)
        
        # Calculate TFLOPS (Tera FLOPS)
        jax_tflops = (basic_flops * 1e-12) / jax_time
        torch_tflops = (basic_flops * 1e-12) / torch_time
        jax_complex_tflops = (complex_flops * 1e-12) / jax_complex_time
        torch_complex_tflops = (complex_flops * 1e-12) / torch_complex_time
        
        # Store results
        results.extend([
            {
                'operation': 'matmul',
                'shape': f"{shape_a} × {shape_b}",
                'framework': 'JAX (jit)',
                'time': jax_time,
                'TFLOPS': jax_tflops
            },
            {
                'operation': 'matmul',
                'shape': f"{shape_a} × {shape_b}",
                'framework': 'PyTorch (script)',
                'time': torch_time,
                'TFLOPS': torch_tflops
            },
            {
                'operation': 'complex',
                'shape': f"{shape_a} × {shape_b}",
                'framework': 'JAX (jit)',
                'time': jax_complex_time,
                'TFLOPS': jax_complex_tflops
            },
            {
                'operation': 'complex',
                'shape': f"{shape_a} × {shape_b}",
                'framework': 'PyTorch (script)',
                'time': torch_complex_time,
                'TFLOPS': torch_complex_tflops
            }
        ])
        
        # Print immediate results
        print(f"\nMatrix Multiplication Results:")
        print(f"JAX (jit): {jax_time*1e6:.2f} µs ({jax_tflops:.2f} TFLOPS)")
        print(f"PyTorch (script): {torch_time*1e6:.2f} µs ({torch_tflops:.2f} TFLOPS)")
        print(f"\nComplex Operation Results:")
        print(f"JAX (jit): {jax_complex_time*1e6:.2f} µs ({jax_complex_tflops:.2f} TFLOPS)")
        print(f"PyTorch (script): {torch_complex_time*1e6:.2f} µs ({torch_complex_tflops:.2f} TFLOPS)")
    
    return pd.DataFrame(results)

def plot_tensor_results(results_df: pd.DataFrame):
    """Create visualization of tensor operation results with side-by-side bars."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create figure with subplots for each operation
        operations = results_df['operation'].unique()
        fig, axes = plt.subplots(1, len(operations), figsize=(15, 6))
        
        # If only one operation, axes will be a single object instead of array
        if len(operations) == 1:
            axes = [axes]
        
        for idx, op in enumerate(operations):
            op_data = results_df[results_df['operation'] == op]
            
            # Get unique shapes and frameworks
            shapes = op_data['shape'].unique()
            frameworks = op_data['framework'].unique()
            
            # Set width of bars and positions of the bars
            width = 0.35
            x = np.arange(len(shapes))
            
            # Plot bars for each framework
            for i, framework in enumerate(frameworks):
                data = op_data[op_data['framework'] == framework]
                # Align bars side by side using the offset
                offset = (i - len(frameworks)/2 + 0.5) * width
                axes[idx].bar(x + offset, data['TFLOPS'], 
                            width, label=framework)
            
            # Customize the plot
            axes[idx].set_xlabel('Matrix Shapes')
            axes[idx].set_ylabel('TFLOPS')
            axes[idx].set_title(f'{op.capitalize()} Performance')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(shapes, rotation=45)
            axes[idx].legend()
            
            # Add grid for better readability
            axes[idx].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")
def main():
    # Test with different matrix shapes
    shapes = [
        # Regular matrix multiplication
        ((512, 512), (512, 512)),
        ((1024, 1024), (1024, 1024)),
        ((2048, 2048), (2048, 2048)),
        # Batched tensor multiplication
        ((32, 512, 512), (32, 512, 512)),
        ((64, 1024, 1024), (64, 1024, 1024))
    ]
    
    n_iter = 100  # Reduced iterations due to larger computations
    
    # Run benchmarks
    results_df = run_tensor_benchmarks(shapes, n_iter)
    
    # Print summary
    print("\nTime summary (seconds):")
    print(results_df.pivot_table(
        index=['shape', 'operation'], 
        columns='framework', 
        values='time'
    ))
    
    print("\nTFLOPS summary:")
    print(results_df.pivot_table(
        index=['shape', 'operation'], 
        columns='framework', 
        values='TFLOPS'
    ))
    
    # Plot results
    plot_tensor_results(results_df)

if __name__ == "__main__":
    main()
