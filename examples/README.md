# Examples

Run each example from the repository root. After the compiler and lowering
logs, it prints where it lowered and exported the model, then checks the loaded
model against PyTorch. The GPU target and temporary directory depend on your
machine.

| Example | What it shows |
| --- | --- |
| [`vector_add.py`](vector_add.py) | One Triton kernel, lowered, exported, and loaded |
| [`recsys_ranker.py`](recsys_ranker.py) | A DLRM-style ranking model from [`recsys/`](recsys) with two Triton kernels |

```text
$ python examples/recsys_ranker.py
Lowered 2 Triton kernels for cuda:sm80 in /tmp/aot_tensor_recsys_ranker_abcd1234
Exported /tmp/aot_tensor_recsys_ranker_abcd1234/model.pt
Loaded model matches the PyTorch reference: True
Loaded model matches it on a batch of 37: True
```

The ranking model combines a PyTorch embedding table and output layer with two
Triton kernels: one sums each user's variable-length item history, and an
autotuned one runs the linear and ReLU of two MLP layers.

## Writing your own kernels

- Give each `triton_aot` kernel and its launcher their own module. Lowering
  accepts one `triton_aot` kernel per module.
- In the module whose model calls a launcher, register the launcher with
  `torch.fx.wrap` so tracing records it as one node.
- In the launcher, assign the grid to a variable and launch the kernel by the
  name of its `@triton.jit` function. A `lambda meta:` grid may read only
  autotuned meta-parameters.
- Keep launchers TorchScript-compatible: exporting compiles them with
  TorchScript.
