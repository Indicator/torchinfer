# torchinfer

torchinfer contains AOT-Tensor, a pipeline that compiles GPU kernels in a
PyTorch model ahead of time, rewrites their call sites in a
`torch.fx.GraphModule`, and exports the model together with its native
libraries. It currently supports Triton kernels.

## Requirements

- Linux with an NVIDIA GPU
- CUDA Toolkit 13.0 with development headers
- Clang 20 or newer
- Python 3.12
- PyTorch 2.10 through 2.14 built for CUDA 13.0, with the Triton version it pins

Exported models target the PyTorch 2.12 stable ABI: load them with PyTorch 2.12
or newer, or with the version that lowered them.

## Installation

```bash
git clone https://github.com/meta-pytorch/torchinfer.git
cd torchinfer
pip install --extra-index-url https://download.pytorch.org/whl/cu130 .
export CXX=/usr/bin/clang++-20
export TORCH_USE_RTLD_GLOBAL=YES
```

Lowering uses the CUDA Toolkit of the `nvcc` on your `PATH`, or else
`/usr/local/cuda`. Set `CUDA_HOME` if that is not CUDA 13.0.

## Quick start

Mark a Triton kernel with `triton_aot`, lower a module that calls it, then
export and load the result:

```python
import tempfile

import torch
import triton
import triton.language as tl
from aot_tensor.api.exporting import export_model
from aot_tensor.api.loading import load_model
from aot_tensor.api.lowering import lower_model
from aot_tensor.types import triton_aot


@triton_aot(annotations={"n_elements": "i32"})
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@torch.fx.wrap
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = ((n_elements + 1023) // 1024,)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


class AddModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return vector_add(x, y)


x = torch.randn(4096, device="cuda")
y = torch.randn(4096, device="cuda")
graph_module = torch.fx.symbolic_trace(AddModel())
lowering = lower_model(graph_module, [(x, y)], work_dir=tempfile.mkdtemp())
model = load_model(export_model(lowering))
assert torch.allclose(model(x, y), x + y)
```

The runnable version is [`examples/vector_add.py`](examples/vector_add.py).

## Documentation

- [Examples](examples/README.md): a DLRM-style ranking model, and the rules for
  writing your own kernels
- [API reference](docs/api.md): the public API and lowering options
- [Loading from C++](docs/cpp-loader.md): running an exported model without
  Python

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

torchinfer is licensed under the [BSD 3-Clause License](LICENSE).

## Contributors

AOT-Tensor is made possible by the following contributors (listed alphabetically by last name):

Chun-Wei Chen, Rui Jian, Yuhang Liu, Runming Lu, Chang Pan, Katie Tseng, Zhiyong Wang, Chenzhi Yu, Yingji Zhang, Zhuoran Zhao, Tim Zheng.

This work builds on an internal prototype created by Sijia Chen and Bert Maher.
