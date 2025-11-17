# stimojo

A high-performance Mojo port of [Stim](https://github.com/quantumlib/Stim), a fast stabilizer circuit simulator. **stimojo** unlocks SIMD portability across any hardware platforms that support SIMD, leveraging Mojo's compile-time optimization capabilities and cross-platform SIMD abstractions.

## Overview

Stim is a circuit simulator optimized for stabilizer circuits using Pauli frames. stimojo brings this same high-performance approach to Mojo, enabling:

- **Cross-platform SIMD**: Automatic vectorization across CPUs with different SIMD capabilities (AVX-512, AVX2, NEON, etc.)
- **Mojo integration**: Direct integration with Mojo's ecosystem for quantum simulation research
- **High performance**: Compile-time specialization and SIMD-aware algorithms for efficient Pauli operations

## Installation

### Prerequisites

- [Mojo SDK](https://www.modular.com/mojo) (latest version recommended)
- [Pixi](https://pixi.sh/) for dependency management (optional but recommended)

### Setup with Pixi

1. Clone the repository:

```bash
cd /path/to/stabilizer-sim/stimojo
```

2. Initialize the environment:

```bash
pixi install
```

3. Activate the environment:

```bash
pixi shell
```

### Manual Setup

If not using Pixi, ensure you have the Mojo SDK installed and accessible in your `PATH`.

## Quick Start

### Running the Test Suite

Execute all tests to verify the installation:

```bash
mojo test/stimojo/test_pauli_string.mojo
```

### Example: Pauli String Operations

Run the included example to see stimojo in action:

```bash
mojo examples/pauli_string.mojo
```

This demonstrates:

- Creating Pauli strings
- Multiplying Pauli operators with automatic phase tracking
- SIMD-accelerated operations

## Benchmarking

stimojo includes comprehensive benchmarks to evaluate performance:

### Run All Benchmarks

```bash
mojo benchmarks/run_all_benchmarks.mojo
```

### Individual Benchmarks

```bash
mojo benchmarks/pauli_product_benchmark.mojo
```

This benchmark measures:

- Pauli string multiplication performance
- SIMD efficiency across different string lengths
- Comparison with scalar implementations

### Benchmark Output

The benchmarks provide timing results in seconds and can help you:

- Verify SIMD optimizations are working
- Profile performance on your specific hardware
- Compare different implementation strategies

## Architecture

### Core Components

- **`src/stimojo/pauli.mojo`**: Main Pauli string implementation with SIMD-accelerated operations
- **`src/stimojo/ops.mojo`**: Circuit operations and stabilizer frame operations
- **`test/stimojo/test_pauli_string.mojo`**: Comprehensive test suite

### SIMD Strategy

stimojo uses Mojo's `@parameter` decorator to specialize code for the host platform's SIMD width at compile time. This enables:

- Automatic vectorization of Pauli operations
- Hardware-aware optimization without runtime branching
- Portable code across different CPU architectures

## Key Features

### Pauli String Multiplication

Efficiently multiply Pauli strings with automatic phase tracking:

```mojo
var p1 = PauliString("XYZZYX")
var p2 = PauliString("ZYXXYZ")
var result = p1 * p2
# Result includes correct global phase from anticommutations
```

### Phase Tracking

Global phases are tracked in log base _i_ (0=1, 1=_i_, 2=-1, 3=-_i_):

```mojo
# XY = iZ (phase = 1)
var result = PauliString("X") * PauliString("Y")
assert_equal(result.global_phase, 1)
```

### In-Place Operations

For efficiency, use in-place multiplication:

```mojo
var p1 = PauliString("XYZZYX")
var p2 = PauliString("ZYXXYZ")
p1.prod(p2)  # Modifies p1 in-place, more efficient than p1 * p2
```

## Testing

Run tests with detailed output:

```bash
mojo -debug test/stimojo/test_pauli_string.mojo
```


## References

- [Stim GitHub Repository](https://github.com/quantumlib/Stim)
- [Mojo Documentation](https://docs.modular.com/mojo/)
