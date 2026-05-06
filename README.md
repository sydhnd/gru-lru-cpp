# GRU vs LRU: Library-Free C++ Inference Benchmark

A side-by-side performance comparison of GRU and LRU (Linear Recurrent Unit) models running pure bare-metal C++ inference — no PyTorch, no ONNX runtime, no external ML libraries. Weights are trained in Python, exported via a custom interleaved binary format, and loaded directly into memory for the forward pass.

> **Scope:** This is a systems-level benchmark designed as a proof-of-concept for ultra-lightweight edge deployments. The focus is on execution speed, memory layout efficiency, and the mechanical cost of the forward pass without framework bloat. Numerical correctness against the PyTorch reference is out of scope for this demo.

> **Related project:** The full training pipeline (with proper validation, multi-epoch training, and the model architecture itself) lives in [sydhnd/parallel-lru](https://github.com/sydhnd/parallel-lru). This repo strips that down to a single epoch and removes validation on purpose — the goal here is to focus on CPU inference, not training quality.

## Why bare-metal C++?

Frameworks like PyTorch and ONNX Runtime add a lot of machinery you don't need (and don't want) on edge devices or in systems where deterministic latency is critical. Doing inference by hand in C++ gives you:

- A tiny footprint (<1MB binary) with zero runtime dependencies — no Docker container, no Python environment.
- Predictable execution with no graph optimizer surprises.
- A clean apples-to-apples comparison of the recurrent architectures themselves.

## Dataset

Training and evaluation use the [Anomaly Detection in Oil and Gas / Chemical Plants dataset](https://www.kaggle.com/datasets/programmer3/anomaly-detection-in-oil-and-gas-chemical-plants) from Kaggle.

The raw CSV is not included in this repository. Place `plant_sensor_data.csv` under the `data/` directory before running training.

## The Computational Reality: GRU vs LRU

**GRU** relies on update and reset gates driven by non-linear activations. Computing sigmoid and tanh in C++ requires polynomial approximations of `exp()`. This stalls the CPU pipeline, forces frequent memory loads across multiple coupled weight matrices, and prevents the compiler from fully vectorizing the loop.

**LRU** drops the non-linear gates entirely in favor of a linear recurrence using complex numbers (`std::complex<float>`). With no `exp()` calls, the inner loop is just fused multiply-add (FMA) instructions — far fewer memory reads, and very friendly to SIMD auto-vectorization.

### The GRU Mathematical Bottleneck

To compute a single step of a GRU, the CPU has to calculate three separate gates: a reset gate, an update gate, and a candidate hidden state. Each requires a matrix-vector multiplication followed by a non-linear activation:

$$r_t = \sigma(W_{ir} x_t + W_{hr} h_{t-1} + b_r)$$

$$z_t = \sigma(W_{iz} x_t + W_{hz} h_{t-1} + b_z)$$

$$n_t = \tanh(W_{in} x_t + r_t \odot (W_{hn} h_{t-1} + b_{hn}))$$

$$h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$$

The bottleneck is the sigmoid and tanh. Sigmoid is defined as:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

CPUs cannot calculate $e^{-x}$ natively. They have to approximate it with polynomials (Taylor-series-style expansions or specialized hardware approximations). That takes significantly more clock cycles than basic arithmetic and breaks the CPU's ability to keep the instruction pipeline full. For every float in the hidden state, the CPU has to stop, run the polynomial approximation, and then resume the matrix math.

### The LRU Mathematical Advantage

The LRU operates entirely in the complex domain using linear algebra, with no non-linearities at all:

$$h_t = \lambda \odot h_{t-1} + B x_t$$

Here $\lambda$, $h_t$, and $B$ are complex numbers (`std::complex<float>`). The operation $\lambda \odot h_{t-1}$ is element-wise complex multiplication. For two complex numbers $x = a + bi$ and $y = c + di$:

$$(a + bi)(c + di) = (ac - bd) + (ad + bc)i$$

That's 4 multiplications and 2 additions. No `exp()`, no polynomial approximations, no pipeline stalls. The CPU's ALU executes this with Fused Multiply-Add (FMA) instructions in a single hardware step. With `-march=native`, the compiler packs these complex numbers into SIMD registers and runs the entire hidden state update with uninterrupted efficiency.

LRU wins because it replaces expensive, pipeline-stalling calculus with highly parallelizable, hardware-friendly arithmetic.

### One caveat on the LRU speedup

The win comes from doing fewer FLOPs per step, not from parallel scans. Step-by-step inference can't take advantage of the associative-scan property that makes linear RNNs fast during training, since each step still depends on the previous one. The architectural advantage here is purely about doing less work per step.

## Benchmark Results

**Setup:** AMD Ryzen 5900X, single-threaded execution.

**Workload:** 1 inference = processing a full batch of `[batch=32, sequence=300, features=7]`.

The table below compares an unoptimized C++ build (no flags) against a heavily optimized build (`-O3 -march=native -ffast-math`). Showing both lets you see what the architecture contributes and what the compiler contributes.

| Frequency | Model | Latency (Unoptimized) | Latency (Optimized) | CPU (Unoptimized) | CPU (Optimized) |
|-----------|-------|------------------------|----------------------|--------------------|------------------|
| 1 Hz      | GRU   | ~56,190 μs             | ~7,237 μs            | 5.6%               | 0.7%             |
| 1 Hz      | LRU   | ~7,178 μs              | ~256 μs              | 0.7%               | <0.1%            |
| 10 Hz     | GRU   | ~55,617 μs             | ~7,087 μs            | 34.0%              | 6.6%             |
| 10 Hz     | LRU   | ~7,087 μs              | ~254 μs              | 6.6%               | 0.7%             |
| 100 Hz    | GRU   | ~56,265 μs             | ~7,194 μs            | 84.7%              | 41.7%            |
| 100 Hz    | LRU   | ~7,018 μs              | ~254 μs              | 41.7%              | 2.6%             |
| 1000 Hz   | GRU   | *capped*               | ~6,509 μs            | *N/A*              | 84.8%            |
| 1000 Hz   | LRU   | ~6,404 μs              | ~216 μs              | 85.8%              | 18.9%            |

*The unoptimized GRU could not sustain 1000 Hz on a single core, so that row is omitted from the baseline.*

**Key takeaways:**

- **Architecture matters.** Unoptimized, LRU is already ~8x faster than GRU. With the same optimization flags applied to both, the gap widens to ~28x. The linear recurrence simply has more headroom for the compiler to exploit than the gated logic does.
- **Compiler flags matter.** `-O3 -march=native -ffast-math` gave GRU roughly an 8x speedup and LRU roughly a 28x speedup. Same flags, very different impact — because LRU's inner loop is mostly FMA on contiguous floats, which is exactly what AVX2/AVX-512 is designed to accelerate.
- **Throughput headroom.** Optimized GRU nearly saturates a single core at 1000 Hz (84.8% CPU). Optimized LRU runs the same frequency at 18.9% CPU, leaving room to process roughly 4,000 batches per second before saturating one core.

## Memory Layout: Zero-Copy Loading

Traditional loaders require an intermediate buffer to parse data. By using an interleaved binary format, this project achieves a direct memory handshake:

1. **Python side:** Interleaves complex weights into a flat `[Re, Im, Re, Im, ...]` array.
2. **Disk side:** The `.bin` file matches the `std::complex<float>` memory layout exactly.
3. **C++ side:** `file.read()` streams bytes directly into the pre-allocated vector.

Result: zero intermediate buffers, zero reconstruction loops, and near-zero "boot time" for the model.

The C++ standard guarantees `std::complex<float>` is layout-compatible with `float[2]`, which is what makes the `reinterpret_cast` safe:

```cpp
typedef std::complex<float> Complex;

struct LruModel {
    uint32_t feature  = 0;
    uint32_t hidden   = 0;
    uint32_t sequence = 0;
    std::vector<Complex> lam;  // diagonal recurrence
    std::vector<Complex> B;       // input projection
    std::vector<Complex> C;       // output projection
};

// Zero-copy binary read directly into the complex vector
m.lam.resize(m.hidden);
file.read(reinterpret_cast<char*>(m.lam.data()),
          m.hidden * sizeof(Complex));
```

The GRU loader is heavier — it has to correctly offset and map several sets of input-to-hidden and hidden-to-hidden weights and biases, one set per gate. Getting the strides right matters, because a wrong offset produces garbage outputs that still run without crashing.

## Build & Run

```bash
g++ -O3 -march=native -ffast-math *.cpp -o inference_benchmark

./inference_benchmark --model_path weights/gru_model.bin
./inference_benchmark --model_path weights/lru_model.bin
```

Compiler flags do most of the heavy lifting on this workload:

- `-march=native` unlocks AVX2/AVX-512 instructions, so the CPU can compute 8–16 floating-point ops per cycle instead of one.
- `-ffast-math` relaxes strict IEEE 754 compliance, letting the compiler reorder math operations to keep the execution pipeline full. Fine for a perf demo; revisit before any production use since it changes NaN/Inf handling.

## Conclusion

If your use case is "process a stream of sensor data fast and predictably on a CPU," the gates in a GRU are paying for an inductive bias you likely don't need. LRU gives you a massive latency improvement and serious throughput headroom on the same hardware. Deploying a high-speed recurrent model to the edge doesn't require a heavy framework or a discrete GPU — just clean math, hardware-aware memory layout, and a compiled binary.
