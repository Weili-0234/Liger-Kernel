# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Liger Kernel is a collection of Triton kernels for efficient LLM training. It increases training throughput by 20% and reduces memory usage by 60% through kernel fusion, in-place operations, and chunked computation. The kernels are Hugging Face compatible and work with Flash Attention, PyTorch FSDP, and DeepSpeed.

## Common Commands

### Development Setup
```bash
pip install -e .              # Install (auto-detects CUDA/ROCm/XPU/NPU)
pip install -e ".[dev]"       # Install with dev dependencies
prek install                  # Install pre-commit hooks
prek run -a                   # Run pre-commit checks
```

### Testing
```bash
make test                     # Run pytest with coverage (excludes convergence tests)
make checkstyle               # Run ruff linting and formatting
make test-convergence         # Run convergence tests (fp32 and bf16)
python -m pytest test/path/test_file.py::test_function_name  # Run single test
```

### Benchmarks
```bash
make run-benchmarks           # Run all benchmark scripts
make run-benchmarks OVERWRITE=1  # Overwrite existing benchmark data
python benchmark/scripts/benchmark_{kernel_name}.py  # Run individual benchmark
```

### Documentation
```bash
make serve                    # Serve docs locally
make build                    # Build documentation
```

## Architecture

### Source Code Structure (`src/liger_kernel/`)

- **`ops/`**: Core Triton kernels with forward/backward implementations
  - Pure Triton operations (cross_entropy, rms_norm, rope, swiglu, geglu, layer_norm, jsd, etc.)
  - `backends/`: Multi-vendor support (currently Ascend NPU via `_ascend/`)
  - `experimental/`: Experimental ops (embedding, matmul)

- **`transformers/`**: PyTorch `nn.Module` wrappers for HuggingFace compatibility
  - `monkey_patch.py`: Model-specific patching APIs (`apply_liger_kernel_to_*`)
  - `auto_model.py`: `AutoLigerKernelForCausalLM` for automatic patching
  - `model/`: Custom forward functions for fused operations
  - Individual modules: `LigerRMSNorm`, `LigerCrossEntropyLoss`, `LigerSwiGLUMLP`, etc.

- **`chunked_loss/`**: Memory-efficient post-training loss kernels
  - Preference optimization: DPO, ORPO, CPO, SimPO, KTO, GRPO
  - `fused_linear_preference.py`: Base class for preference losses
  - `fused_linear_distillation.py`: Distillation loss base class

### Tests (`test/`)

- **`transformers/`**: Unit tests for nn.Module wrappers
- **`chunked_loss/`**: Tests for post-training losses
- **`convergence/`**: End-to-end training comparisons (fp32/, bf16/)
- **`utils.py`**: Test utilities including `assert_verbose_allclose()`, `set_seed()`

### Key Patterns

**Monkey-patching**: One-liner integration with HF models:
```python
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()  # Patches before model instantiation
```

**AutoModel**: Automatic kernel selection:
```python
from liger_kernel.transformers import AutoLigerKernelForCausalLM
model = AutoLigerKernelForCausalLM.from_pretrained("model_path")
```

**Direct kernel usage**:
```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
loss_fn = LigerFusedLinearCrossEntropyLoss()
```

### Hardware Support

- **CUDA**: torch >= 2.1.2, triton >= 2.3.1
- **ROCm**: torch >= 2.5.0, triton >= 3.0.0
- **Intel XPU**: torch >= 2.6.0
- **Ascend NPU**: torch_npu == 2.7.1, triton-ascend

Platform auto-detected at install time via `setup.py`.

## Code Style

- **Formatter/Linter**: Ruff (line length 120, Python 3.10+)
- **Imports**: Single-line imports, first-party in `known-first-party = ["liger_kernel"]`
- **Pre-commit**: Uses `prek` (Rust-based pre-commit alternative)

## Adding New Kernels

1. Add kernel implementation in `src/liger_kernel/ops/`
2. Add PyTorch wrapper in `src/liger_kernel/transformers/`
3. Create unit tests in `test/transformers/`
4. Add convergence tests in `test/convergence/`
5. Add benchmark script in `benchmark/scripts/benchmark_{kernel_name}.py`

## Adding Model Support

1. Check which kernels in `src/liger_kernel/ops/` can be monkey-patched
2. Add patching function in `src/liger_kernel/transformers/monkey_patch.py`
3. Export in `src/liger_kernel/transformers/__init__.py`
4. Add unit and convergence tests
