# Embedding.cpp

Text embedding tool via `BERT` models upon [ggml](https://github.com/ggerganov/ggml), with critical bug fixes and improvements over the upstream.

## Improvements Over Upstream

This fork includes three critical bug fixes that make the library actually functional:

### 1. Fix SIGILL on tokenizer load (`tokenizer.cpp`)

`bert_tokenizer::load()` declared a `bool` return type but had no `return` statement. The compiler placed a `ud2` (undefined instruction) after the function body, causing an immediate **SIGILL (exit code 132)** on every call. This made the library completely unusable.

**Fix:** Added `return true;` at the end of `bert_tokenizer::load()`.

### 2. Fix SIGSEGV from use-after-free (`bert.cpp`)

In `bert_eval_batch()`, `ggml_free(ctx0)` was called *before* reading `gf->nodes[]` and `ggml_used_mem(ctx0)`. In release builds (`-O2`), the optimizer reuses the freed memory, causing a **SIGSEGV** crash.

**Fix:** Moved `ggml_free(ctx0)` to after all reads from `gf` and `ctx0`.

### 3. Fix garbage embeddings from wrong graph node (`bert.cpp`)

`bert_eval_batch()` read `gf->nodes[n_nodes - 2]` which is an intermediate `ggml_div` node producing the scalar `1.0f / length` — **not** the embedding vector. The actual normalized embedding is `gf->nodes[n_nodes - 1]` (the final `ggml_scale` output). This caused garbage embeddings with magnitude ~6.7e22 and mostly zero values.

**Fix:** Changed to `embeddings_tensor = gf->nodes[gf->n_nodes - 1]`.

---

## Feature (Origin)

* Plain C/C++ implementation without dependencies
* Inherit support for various architectures from ggml (x86 with AVX2, ARM, etc.)
* Choose your model size from 32/16/4 bits per model weight
* all-MiniLM-L6-v2 with 4bit quantization is only 14MB. Inference RAM usage depends on the length of the input
* Sample cpp server over tcp socket and a python test client
* Benchmarks to validate correctness and speed of inference

## Feature (Improve)

* Build tokenizer with [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp).
* Can correctly handle asian writing (CJK, and so on).
* Can process cased/uncased with respect to origin config in `tokenizer.json`.
* Upgrade to use [GGUF](https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md) model file format. So it is easy to expand and keep compatible.
* **Critical bug fixes** listed above — without these, the upstream code does not produce usable embeddings.

> With above, we can run embedding.cpp with more models like [m3e](), [e5]() and so on.

## Limitation

* Only support bert base model for embedding. other architecture like SGPT is not supported.
* Only run on CPU.
* All outputs are mean pooled and normalized.
* Batching support is WIP.
* Lack of real batching means that this library is slower than it could be in usecases where you have multiple sentences.

## Usage

### Checkout submodules

```sh
git submodule update --init --recursive
```

### Build

By default, it build both
- the native binaries, like the example server, with static libraries;
- and the dynamic library for usage from e.g. Python.

```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
cd ..
```

> rust should be installed. see [rust](https://www.rust-lang.org/tools/install) or run `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

### Converting models to gguf format

Converting models is similar to llama.cpp. Use models/convert-to-gguf.py to make hf models into either f32 or f16 gguf models.
Then use ./build/bin/quantize to turn those into Q4_0, 4bit per weight models.

There is also models/run_conversions.sh which creates all 4 versions (f32, f16, Q4_0, Q4_1) at once.

```sh
pip install -r requirements.txt
cd models
# Clone a model from hf
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# Run conversions to 4 ggml formats (f32, f16, Q4_0, Q4_1)
sh run_conversions.sh all-MiniLM-L6-v2
```

## Acknowledgments

This project is a fork of [embedding.cpp](https://github.com/FFengIll/embedding.cpp) by FFengIll, which itself is a fork of [bert.cpp](https://github.com/skeskinen/bert.cpp) by skeskinen. Thank you to the original authors and contributors for the foundational work that made this possible.
