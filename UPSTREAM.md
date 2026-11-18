# clip.cpp (vendored)

This directory is a vendored copy of [clip.cpp](https://github.com/ningpengtao-coder/clip.cpp),
a dependency-free CLIP inference implementation built on [ggml](https://github.com/ggerganov/ggml).

## Upstream

- Repository: https://github.com/ningpengtao-coder/clip.cpp
- Commit: `913458d5d1c9238380a0b0826cd1e71c8828a82d` (`Update README.md`)
- License: MIT (see [LICENSE](LICENSE))
- ggml: vendored in-tree at [ggml/](ggml), from `https://github.com/ggerganov/ggml`
  at commit `c3ae31e5a090a6259c674b18983de53ac4538aa6`. The upstream `.git` and the
  `ggml/.git` submodule metadata were removed after cloning/deps were checked out.

## Local changes

### Build / portability (Windows + MSVC)

- `CMakeLists.txt`: enables `WINDOWS_EXPORT_ALL_SYMBOLS` on the shared `clip` target.
  `clip.h` has no `dllexport` macros, so without this the DLL has no import library
  and cannot be consumed via FFI (Dart / Python ctypes).
- `clip.cpp`: replaced the pthread-based `clip_image_batch_preprocess` implementation
  with `std::thread`; replaced C99 designated initializers and variable-length arrays
  (unsupported by MSVC in C++11 mode) with C++11-compatible code.
- `models/quantize.cpp`: calls `ggml_time_init()`. On Windows `ggml_time_us()` divides
  by `timer_freq`, which is only initialized by `ggml_time_init()`.
- `ggml` vendored here implements GGUF v2. `models/convert_hf_to_gguf.py` emits GGUF v2
  metadata (v2 and v3 have the same on-disk layout) so the generated files load.

### Chinese-CLIP support

- `models/convert_hf_to_gguf.py`
  - Detects `model_type: "chinese_clip"` / `architectures: ["ChineseCLIPModel"]`.
  - Reads the WordPiece `vocab.txt` and writes it as the GGUF token list.
  - Writes layout metadata: `clip.tokenizer` (`bpe` / `wordpiece`),
    `clip.text.arch` (`clip` / `bert`), `clip.text.pooling` (`eot` / `cls`),
    `clip.vision.pooling` (`cls`), `clip.text.use_gelu`, `clip.vision.use_gelu`
    and `clip.text.pad_length`.
  - Maps `text_model.embeddings.*` (word/position/token-type embeddings,
    embeddings LayerNorm), `text_model.encoder.layer.*.attention.self.*`,
    `attention.output.*`, `intermediate.dense`, `output.dense`, plus
    `text_projection.weight` / `visual_projection.weight` onto the runtime tensor names.
  - Uses the actual projection weight shapes for `projection_dim` (the Chinese-CLIP
    vision config can carry a stale default value).
  - Chinese-CLIP output files are named `<name>-f16.gguf` / `<name>-f32.gguf`
    (`--use-f32`). The original CLIP naming scheme is unchanged.
- `clip.cpp` / `clip.h`
  - WordPiece tokenizer for BERT-style text towers: ASCII lowercasing, whitespace /
    CJK / punctuation splitting, greedy longest-match-first WordPiece, `[CLS] ... [SEP]`
    wrapping, padding to a fixed length.
  - BERT text tower: bidirectional attention, padding mask (padded keys excluded),
    token-type embeddings, embedding LayerNorm, post-LN layer order, CLS pooling.
    The original CLIP text tower (causal attention, EOT pooling, pre-LN, final LN)
    is selected by metadata and remains unchanged.
  - Per-tower activation selection (`clip.text.use_gelu` vs `clip.vision.use_gelu`)
    because Chinese-CLIP uses exact `gelu` for text and `quick_gelu` for vision.
  - Vision tower already supports 224/336 and patch14/patch16 through hparams; CLS
    pooling is used for both layouts.
  - `clip_text_encode_raw` / `clip_image_encode_raw` return unnormalized embeddings
    without changing the semantics of `clip_text_encode` / `clip_image_encode`.
  - Compute/scratch buffer sizing has a computed fallback for tensor counts not in
    the original table (e.g. Chinese-CLIP).
  - Load-time validation that the `projection_dim` metadata matches the projection
    weight shape.

## Known limitations / TODO

- WordPiece normalization approximates HF `BertTokenizer`: ASCII-only lowercasing and
  no NFD accent stripping. Chinese input is unaffected; accented Latin text may differ.
- The converter requires a `transformers` release with native `ChineseCLIPModel`
  support (tested with transformers 5.23.0).
- Text sequences are padded to `clip.text.pad_length` (64 by default, clamped to
  `max_position_embeddings`). Padding is masked out, so it does not change the CLS
  embedding, but it does add compute for short inputs.
- Image position-embedding interpolation for resolutions other than the checkpoint's
  `image_size` is not implemented; use the native 224/patch16 or 336/patch14 checkpoint.
- Numeric verification in this workspace used a randomly initialized tiny Chinese-CLIP
  model (no real Chinese-CLIP weights in `assets/models`): max abs diff vs. PyTorch f32
  was ~1.2e-3 for text embeddings and ~7.8e-4 for image embeddings (f16 GGUF weights).
  No parity numbers against the real `chinese-clip-vit-*` checkpoints are claimed.
- `models/quantize.cpp` still only exposes the upstream CLI; `clip_model_quantize` itself
  is unchanged. q8_0 and F16 were exercised for the tiny model.
