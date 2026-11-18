import argparse
import os
import json
import re

import numpy as np
import gguf.gguf_writer
from gguf import *
from transformers import AutoModel, CLIPProcessor

# The GGUF reader vendored in clip.cpp predates GGUF v3. v2 and v3 share the
# same on-disk layout, so emit v2 metadata for compatibility.
gguf.gguf_writer.GGUF_VERSION = 2

TEXT = "clip.text"
VISION = "clip.vision"

# Default CLIP image normalization values, used when preprocessor_config.json is
# not available and no CLIPProcessor can be loaded.
DEFAULT_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
DEFAULT_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)

def is_chinese_clip(config: dict) -> bool:
    model_type = str(config.get("model_type", "")).lower().replace("-", "_")
    if model_type in ("chinese_clip", "chineseclip"):
        return True
    for arch in config.get("architectures") or []:
        if isinstance(arch, str) and "ChineseCLIP" in arch:
            return True
    return False

def should_skip_tensor(name: str, text_only: bool, vision_only: bool) -> bool:
    if name in (
        "logit_scale",
        "text_model.embeddings.position_ids",
        "text_model.embeddings.token_type_ids",
        "vision_model.embeddings.position_ids",
    ):
        return True

    if text_only and name.startswith("v"):
        return True

    if vision_only and name.startswith("t"):
        return True

    return False

def get_bert_text_tensor_name(name: str) -> str:
    """Maps HuggingFace BERT-style Chinese-CLIP text tensors onto clip.cpp names."""
    embeddings = [
        (r"text_model\.embeddings\.word_embeddings\.(weight|bias)", r"t.token_embd.\1"),
        (r"text_model\.embeddings\.position_embeddings\.(weight|bias)", r"t.position_embd.\1"),
        (r"text_model\.embeddings\.token_type_embeddings\.(weight|bias)", r"t.token_type_embd.\1"),
        (r"text_model\.embeddings\.LayerNorm\.(weight|bias)", r"t.pre_ln.\1"),
    ]
    layers = [
        (r"text_model\.encoder\.layer\.(\d+)\.attention\.self\.query\.(weight|bias)", r"t.blk.\1.attn_q.\2"),
        (r"text_model\.encoder\.layer\.(\d+)\.attention\.self\.key\.(weight|bias)", r"t.blk.\1.attn_k.\2"),
        (r"text_model\.encoder\.layer\.(\d+)\.attention\.self\.value\.(weight|bias)", r"t.blk.\1.attn_v.\2"),
        (r"text_model\.encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)", r"t.blk.\1.attn_out.\2"),
        (r"text_model\.encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.(weight|bias)", r"t.blk.\1.ln1.\2"),
        (r"text_model\.encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)", r"t.blk.\1.ffn_down.\2"),
        (r"text_model\.encoder\.layer\.(\d+)\.output\.dense\.(weight|bias)", r"t.blk.\1.ffn_up.\2"),
        (r"text_model\.encoder\.layer\.(\d+)\.output\.LayerNorm\.(weight|bias)", r"t.blk.\1.ln2.\2"),
    ]
    for pattern, repl in embeddings + layers:
        if re.match(pattern, name):
            return re.sub(pattern, repl, name)
    return None

def get_tensor_name(name: str, bert_text: bool = False) -> str:
    if "projection" in name:
        return name

    if bert_text:
        mapped = get_bert_text_tensor_name(name)
        if mapped is not None:
            return mapped

    return name.replace("text_model", "t").replace("vision_model", "v").replace("encoder.layers", "blk").replace("embeddings.", "").replace("_proj", "").replace("self_attn.", "attn_").replace("layer_norm", "ln").replace("layernorm", "ln").replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up").replace("embedding", "embd").replace("final", "post").replace("layrnorm", "ln")


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

ap = argparse.ArgumentParser(prog="convert_hf_to_gguf.py")
ap.add_argument("-m", "--model-dir", help="Path to model directory cloned from HF Hub", required=True)
ap.add_argument("--use-f32", action="store_true", default=False, help="Use f32 instead of f16")
ap.add_argument("--text-only", action="store_true", required=False, help="Save a text-only model. It can't be used to encode images")
ap.add_argument("--vision-only", action="store_true", required=False, help="Save a vision-only model. It can't be used to encode texts")
ap.add_argument("--image-mean", nargs=3, type=float, required=False, help="Override image mean values")
ap.add_argument("--image-std", nargs=3, type=float, required=False, help="Override image std values")
ap.add_argument("-o", "--output-dir", help="Directory to save GGUF files. Default is the original model directory", default=None)

args = ap.parse_args()

if args.text_only and args.vision_only:
    print("--text-only and --image-only arguments cannot be specified at the same time.")
    exit(1)

if args.use_f32:
    print("WARNING: Weights for the convolution op is always saved in f16, as the convolution op in GGML does not support 32-bit kernel weights yet.")

# output in the same directory as the model if output_dir is None
dir_model = args.model_dir

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    v_hparams = config["vision_config"]
    t_hparams = config["text_config"]

chinese_clip = is_chinese_clip(config)
bert_text = chinese_clip

# Chinese-CLIP uses BERT's WordPiece tokenizer (vocab.txt) instead of the
# CLIP BPE vocab.json.
vocab_json_path = os.path.join(dir_model, "vocab.json")
vocab_txt_path = os.path.join(dir_model, "vocab.txt")
wordpiece = chinese_clip or (not os.path.exists(vocab_json_path) and os.path.exists(vocab_txt_path))

if wordpiece:
    if not os.path.exists(vocab_txt_path):
        raise FileNotFoundError(f"expected a WordPiece vocab.txt in {dir_model}")
    # 只按 "\n" 切分：Python 的 str.splitlines() 还会在 U+2028/U+2029 等
    # Unicode 行分隔符处断行，而 WordPiece 的 vocab.txt 里这些字符可能是
    # 合法词条（Chinese-CLIP 的 vocab.txt 就含有 "U+2028" / "##U+2028"），
    # 用 splitlines() 会导致词表条数多于 embedding 行数、分词 id 整体偏移。
    # newline="" 关闭通用换行转换，再手动去掉行尾的 "\r"（兼容 CRLF）。
    with open(vocab_txt_path, "r", encoding="utf-8", newline="") as f:
        raw_vocab = f.read()
    tokens = raw_vocab.split("\n")
    if tokens and tokens[-1] == "":
        tokens.pop()
    tokens = [token[:-1] if token.endswith("\r") else token for token in tokens]
else:
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        tokens = [key for key in vocab]

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if args.use_f32:
    ftype = 0


model = AutoModel.from_pretrained(dir_model)
list_vars = model.state_dict()

if wordpiece:
    # 写入 GGUF 前校验词表条数与文本 embedding 行数一致；不一致说明分词 id
    # 会整体偏移，静默写出的 GGUF 无法与 HF/ONNX 对齐，因此直接中止。
    embd_name = None
    for candidate in ("text_model.embeddings.word_embeddings.weight",
                      "embeddings.word_embeddings.weight"):
        if candidate in list_vars:
            embd_name = candidate
            break
    if embd_name is None:
        raise RuntimeError("未能找到文本 word embedding 张量（text_model.embeddings.word_embeddings.weight），无法校验词表")
    n_embd_rows = int(list_vars[embd_name].shape[0])
    if len(tokens) != n_embd_rows:
        raise ValueError(
            f"词表条数（{len(tokens)}）与文本 embedding 行数（{n_embd_rows}，来自 {embd_name}）不一致，"
            "分词 id 会整体偏移；已中止，未写入 GGUF。请检查 vocab.txt 的换行格式。"
        )

def projection_dim_of(name: str, fallback: int) -> int:
    # Prefer the actual projection weight shape: some configs (e.g. the vision
    # config of Chinese-CLIP) carry a stale/default projection_dim value.
    weight = list_vars.get(name)
    if weight is not None and weight.dim() == 2:
        return int(weight.shape[0])
    return int(fallback)

text_projection_dim = projection_dim_of("text_projection.weight", t_hparams.get("projection_dim", config["projection_dim"]))
vision_projection_dim = projection_dim_of("visual_projection.weight", v_hparams.get("projection_dim", config["projection_dim"]))

fname_middle = None
has_text_encoder = True
has_vision_encoder = True
if args.text_only:
    fname_middle = "text-"
    has_vision_encoder = False
elif args.vision_only:
    fname_middle = "vision-"
    has_text_encoder = False
else:
    fname_middle = ""

output_dir = args.output_dir if args.output_dir is not None else dir_model
os.makedirs(output_dir, exist_ok=True)
output_prefix = os.path.basename(os.path.normpath(output_dir)).replace("ggml_", "")
if chinese_clip:
    # Chinese-CLIP files use <name>-f16.gguf / <name>-f32.gguf.
    fname_out = os.path.join(output_dir, f"{output_prefix}-{fname_middle}{ftype_str[ftype]}.gguf")
else:
    fname_out = os.path.join(output_dir, f"{output_prefix}_ggml-{fname_middle}model-{ftype_str[ftype]}.gguf")
fout = GGUFWriter(path=fname_out, arch="clip")

fout.add_bool("clip.has_text_encoder", has_text_encoder)
fout.add_bool("clip.has_vision_encoder", has_vision_encoder)
fout.add_file_type(ftype)
model_name = config["_name_or_path"] if config.get("_name_or_path") else os.path.basename(os.path.normpath(dir_model))
fout.add_name(model_name)
if args.text_only:
    fout.add_description("text-only CLIP model")
elif args.vision_only:
    fout.add_description("vision-only CLIP model")
else:
    fout.add_description("two-tower CLIP model")

# Metadata that lets clip.cpp distinguish the original CLIP layout from
# BERT-style text towers (Chinese-CLIP).
fout.add_string("clip.tokenizer", "wordpiece" if wordpiece else "bpe")
fout.add_string("clip.text.arch", "bert" if bert_text else "clip")
fout.add_string("clip.text.pooling", "cls" if bert_text else "eot")
fout.add_string("clip.vision.pooling", "cls")
fout.add_bool("clip.text.use_gelu", (t_hparams.get("hidden_act") or "") == "gelu")
fout.add_bool("clip.vision.use_gelu", (v_hparams.get("hidden_act") or "") == "gelu")
if wordpiece:
    # Fixed sequence length used by the runtime WordPiece tokenizer.
    fout.add_uint32("clip.text.pad_length", 64)

if has_text_encoder:
    # text_model hparams
    fout.add_uint32(k(KEY_CONTEXT_LENGTH, TEXT), t_hparams["max_position_embeddings"])
    fout.add_uint32(k(KEY_EMBEDDING_LENGTH, TEXT), t_hparams["hidden_size"])
    fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, TEXT), t_hparams["intermediate_size"])
    fout.add_uint32("clip.text.projection_dim", text_projection_dim)
    fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, TEXT), t_hparams["num_attention_heads"])
    fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, TEXT), t_hparams["layer_norm_eps"])
    fout.add_uint32(k(KEY_BLOCK_COUNT, TEXT), t_hparams["num_hidden_layers"])
    fout.add_token_list(tokens)

if has_vision_encoder:
    # vision_model hparams
    fout.add_uint32("clip.vision.image_size", v_hparams["image_size"])
    fout.add_uint32("clip.vision.patch_size", v_hparams["patch_size"])
    fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), v_hparams["hidden_size"])
    fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), v_hparams["intermediate_size"])
    fout.add_uint32("clip.vision.projection_dim", vision_projection_dim)
    fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), v_hparams["num_attention_heads"])
    fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), v_hparams["layer_norm_eps"])
    fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), v_hparams["num_hidden_layers"])

    image_mean = args.image_mean
    image_std = args.image_std

    preprocessor = None
    preprocessor_path = os.path.join(dir_model, "preprocessor_config.json")
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, "r", encoding="utf-8") as f:
            preprocessor = json.load(f)

    if image_mean is None or image_std is None:
        if preprocessor is not None:
            image_mean = image_mean or preprocessor.get("image_mean", DEFAULT_IMAGE_MEAN)
            image_std = image_std or preprocessor.get("image_std", DEFAULT_IMAGE_STD)
        else:
            try:
                processor = CLIPProcessor.from_pretrained(dir_model)
                image_mean = image_mean or processor.image_processor.image_mean
                image_std = image_std or processor.image_processor.image_std
            except OSError:
                image_mean = image_mean or DEFAULT_IMAGE_MEAN
                image_std = image_std or DEFAULT_IMAGE_STD

    fout.add_array("clip.vision.image_mean", image_mean)
    fout.add_array("clip.vision.image_std", image_std)

    # 记录预处理是否需要中心裁剪。Chinese-CLIP 的 preprocessor_config.json
    # 为 do_center_crop=false（直接 resize 到 image_size×image_size），其它
    # CLIP 模型默认 true（短边等比缩放 + 中心裁剪）。
    do_center_crop = True
    if preprocessor is not None and "do_center_crop" in preprocessor:
        do_center_crop = bool(preprocessor["do_center_crop"])
    fout.add_bool("clip.vision.do_center_crop", do_center_crop)

use_gelu = v_hparams["hidden_act"] == "gelu"
fout.add_bool("clip.use_gelu", use_gelu)


for name, data in list_vars.items():
    if should_skip_tensor(name, args.text_only, args.vision_only):
        # we don't need this
        print(f"skipping parameter: {name}")
        continue

    name = get_tensor_name(name, bert_text)
    data = data.squeeze().numpy()

    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0
    if n_dims == 4:
        print(f"tensor {name} is always saved in f16")
        data = data.astype(np.float16)
        ftype_cur = 1
    elif ftype == 1:
        if name[-7:] == ".weight" and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        if data.dtype != np.float32:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    print(f"{name} - {ftype_str[ftype_cur]} - shape = {data.shape}")
    fout.add_tensor(name, data)


fout.write_header_to_file()
fout.write_kv_data_to_file()
fout.write_tensors_to_file()
fout.close()

print("Done. Output file: " + fname_out)
