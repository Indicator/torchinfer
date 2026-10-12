# Loading from C++

Keep `model.pt`, its sibling `manifest.json`, and all manifest-relative shared
libraries together. Pass the absolute `model.pt` path to the C++ loader:

```cpp
#include <aot_tensor/api/loading.h>

auto module = aot_tensor::loadModel("/path/to/aott_artifact/model.pt");
auto output = module.forward({input}).toTensor();
```

`loadModel()` validates every library path against the model directory, loads
the libraries in manifest order, and then deserializes the TorchScript model.

## Building the loader

Compile `aot_tensor/api/loading.cpp` with your program as C++20 and link
LibTorch. The include path needs the checkout root and the
[nlohmann/json](https://github.com/nlohmann/json) headers; Ubuntu's
`nlohmann-json3-dev` package installs them on the default path. The LibTorch in
the PyTorch wheel works:

```bash
TORCH_DIR="$(python -c 'import os, torch; print(os.path.dirname(torch.__file__))')"
clang++-20 -std=c++20 -I. -I"$TORCH_DIR/include" \
  -I"$TORCH_DIR/include/torch/csrc/api/include" \
  main.cpp aot_tensor/api/loading.cpp \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" -Wl,--no-as-needed \
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -Wl,--as-needed -ldl \
  -o main
```
