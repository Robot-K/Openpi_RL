import dataclasses
import logging
import pathlib
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _convert_gemma3_hf_to_openpi(hf_weights: dict[str, np.ndarray], *, num_layers: int, num_heads: int, num_kv_heads: int, head_dim: int, width: int) -> dict:
    """Convert HuggingFace Gemma 3 safetensors weights to openpi nested dict structure.

    Weight mapping follows HuggingFace Gemma3DecoderLayer → openpi Gemma Block layout.
    All norms use (1 + scale) convention in both HF and openpi, so no conversion needed.
    """
    # Per-layer weights to stack for nn.scan (axis 0 = depth).
    q_einsums, kv_einsums, attn_vec_einsums = [], [], []
    q_norms, k_norms = [], []
    pre_attn_norms, post_attn_norms = [], []
    pre_ffw_norms, post_ffw_norms = [], []
    gating_einsums, linears = [], []

    for i in range(num_layers):
        pfx = f"model.layers.{i}"

        # Q projection: [N*H, D] -> [N, D, H]
        q_w = hf_weights[f"{pfx}.self_attn.q_proj.weight"]
        q_w = q_w.reshape(num_heads, head_dim, width).transpose(0, 2, 1)
        q_einsums.append(q_w)

        # KV projection: each [K*H, D] -> [K, D, H], stacked -> [2, K, D, H]
        k_w = hf_weights[f"{pfx}.self_attn.k_proj.weight"]
        k_w = k_w.reshape(num_kv_heads, head_dim, width).transpose(0, 2, 1)
        v_w = hf_weights[f"{pfx}.self_attn.v_proj.weight"]
        v_w = v_w.reshape(num_kv_heads, head_dim, width).transpose(0, 2, 1)
        kv_einsums.append(np.stack([k_w, v_w], axis=0))

        # O projection: [D, N*H] -> [N, H, D]
        o_w = hf_weights[f"{pfx}.self_attn.o_proj.weight"]
        o_w = o_w.reshape(width, num_heads, head_dim).transpose(1, 2, 0)
        attn_vec_einsums.append(o_w)

        # QK-norm
        q_norms.append(hf_weights[f"{pfx}.self_attn.q_norm.weight"])
        k_norms.append(hf_weights[f"{pfx}.self_attn.k_norm.weight"])

        # Layer norms (all use (1 + weight) convention, load directly).
        pre_attn_norms.append(hf_weights[f"{pfx}.input_layernorm.weight"])
        post_attn_norms.append(hf_weights[f"{pfx}.post_attention_layernorm.weight"])
        pre_ffw_norms.append(hf_weights[f"{pfx}.pre_feedforward_layernorm.weight"])
        post_ffw_norms.append(hf_weights[f"{pfx}.post_feedforward_layernorm.weight"])

        # MLP: gate_proj + up_proj -> [2, D, M]; down_proj -> [M, D]
        gate_w = hf_weights[f"{pfx}.mlp.gate_proj.weight"].T  # [D, M]
        up_w = hf_weights[f"{pfx}.mlp.up_proj.weight"].T  # [D, M]
        gating_einsums.append(np.stack([gate_w, up_w], axis=0))  # [2, D, M]
        linears.append(hf_weights[f"{pfx}.mlp.down_proj.weight"].T)  # [M, D]

    return {
        "llm": {
            "embedder": {
                "input_embedding": hf_weights["model.embed_tokens.weight"],
            },
            "layers": {
                "attn": {
                    "q_einsum": {"w": np.stack(q_einsums)},
                    "kv_einsum": {"w": np.stack(kv_einsums)},
                    "attn_vec_einsum": {"w": np.stack(attn_vec_einsums)},
                    "q_norm": {"scale": np.stack(q_norms)},
                    "k_norm": {"scale": np.stack(k_norms)},
                },
                "pre_attention_norm": {"scale": np.stack(pre_attn_norms)},
                "post_attention_norm": {"scale": np.stack(post_attn_norms)},
                "pre_ffw_norm": {"scale": np.stack(pre_ffw_norms)},
                "post_ffw_norm": {"scale": np.stack(post_ffw_norms)},
                "mlp": {
                    "gating_einsum": np.stack(gating_einsums),
                    "linear": np.stack(linears),
                },
            },
            "final_norm": {
                "scale": hf_weights["model.norm.weight"],
            },
        },
    }


@dataclasses.dataclass(frozen=True)
class ValueFunctionWeightLoader(WeightLoader):
    """Loads Gemma 3 270M weights from HuggingFace safetensors for the value function.

    Optionally loads SigLIP weights from HuggingFace (google/siglip-so400m-patch14-224).

    Args:
        gemma3_dir: Local path to a directory containing HuggingFace Gemma 3 270M
            safetensors files (e.g. from `huggingface-cli download google/gemma-3-270m`).
        load_siglip: If True, also loads SigLIP weights from HuggingFace.
        siglip_hf_repo: HuggingFace repo ID for standalone SigLIP model.
    """

    gemma3_dir: str
    load_siglip: bool = True
    siglip_hf_repo: str = "google/siglip-so400m-patch14-224"

    def load(self, params: at.Params) -> at.Params:
        from safetensors import safe_open

        gemma3_path = pathlib.Path(self.gemma3_dir)
        if not gemma3_path.exists():
            gemma3_path = download.maybe_download(self.gemma3_dir)

        # Load safetensors (single file for 270M model).
        # Use framework='flax' to handle bfloat16 (numpy doesn't support bf16).
        st_file = gemma3_path / "model.safetensors"
        if st_file.exists():
            st_files = [st_file]
        else:
            import glob
            st_files = sorted(gemma3_path.glob("model-*.safetensors"))
            if not st_files:
                raise FileNotFoundError(f"No safetensors files found in {gemma3_path}")

        hf_weights = {}
        for sf in st_files:
            with safe_open(str(sf), framework="flax") as f:
                for key in f.keys():
                    hf_weights[key] = np.array(f.get_tensor(key), dtype=np.float32)

        logger.info(f"Loaded {len(hf_weights)} tensors from Gemma 3 270M checkpoint.")

        # Convert to openpi format.
        # Architecture params for Gemma 3 270M.
        loaded_params = _convert_gemma3_hf_to_openpi(
            hf_weights,
            num_layers=18,
            num_heads=4,
            num_kv_heads=1,
            head_dim=256,
            width=640,
        )

        hf_weights.clear()  # Free memory.

        # Optionally load SigLIP from HuggingFace.
        if self.load_siglip:
            logger.info("Loading SigLIP weights from HuggingFace (%s)...", self.siglip_hf_repo)
            siglip_params = _load_siglip_from_huggingface(
                self.siglip_hf_repo,
                num_layers=27,
                num_heads=16,
                width=1152,
            )
            loaded_params["img"] = siglip_params
            logger.info("SigLIP backbone weights loaded (%d top-level keys).", len(siglip_params))

        # Merge with reference params (fills in value_head and any missing keys).
        return _merge_params(loaded_params, params, missing_regex=".*")


def _load_siglip_from_huggingface(
    repo_id: str,
    *,
    num_layers: int = 27,
    num_heads: int = 16,
    width: int = 1152,
) -> dict:
    """Download SigLIP from HuggingFace and convert to big_vision Flax format (scanned).

    Converts from HuggingFace PyTorch SigLIP format to big_vision Flax format
    compatible with openpi's SigLIP module (scan=True).

    Returns:
        Nested dict of SigLIP params (the 'img' subtree), ready for _merge_params.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    st_path = hf_hub_download(repo_id, "model.safetensors")
    logger.info("SigLIP safetensors downloaded: %s", st_path)

    hf = {}
    with safe_open(st_path, framework="numpy") as f:
        for key in f.keys():
            if key.startswith("vision_model."):
                hf[key] = f.get_tensor(key).astype(np.float32)

    head_dim = width // num_heads

    # --- Patch embedding ---
    # HF Conv2d: [O, I, H, W] -> Flax Conv: [H, W, I, O]
    conv_kernel = hf["vision_model.embeddings.patch_embedding.weight"]
    conv_kernel = conv_kernel.transpose(2, 3, 1, 0)  # [14, 14, 3, 1152]
    conv_bias = hf["vision_model.embeddings.patch_embedding.bias"]

    # --- Position embedding ---
    # HF: [num_patches, width] -> Flax: [1, num_patches, width]
    pos_emb = hf["vision_model.embeddings.position_embedding.weight"][None, :, :]

    # --- Encoder blocks (stacked for nn.scan) ---
    # Collect per-layer weights and stack along axis 0.
    ln0_scale, ln0_bias = [], []
    ln1_scale, ln1_bias = [], []
    q_kernel, q_bias = [], []
    k_kernel, k_bias = [], []
    v_kernel, v_bias = [], []
    out_kernel, out_bias = [], []
    fc1_kernel, fc1_bias = [], []
    fc2_kernel, fc2_bias = [], []

    for i in range(num_layers):
        pfx = f"vision_model.encoder.layers.{i}"

        # LayerNorm 0 (pre-attention)
        ln0_scale.append(hf[f"{pfx}.layer_norm1.weight"])
        ln0_bias.append(hf[f"{pfx}.layer_norm1.bias"])

        # Self-attention: HF Linear [out, in] -> Flax Dense [in, out]
        # Then reshape for multi-head: [in, num_heads, head_dim]
        q_w = hf[f"{pfx}.self_attn.q_proj.weight"].T.reshape(width, num_heads, head_dim)
        q_b = hf[f"{pfx}.self_attn.q_proj.bias"].reshape(num_heads, head_dim)
        q_kernel.append(q_w)
        q_bias.append(q_b)

        k_w = hf[f"{pfx}.self_attn.k_proj.weight"].T.reshape(width, num_heads, head_dim)
        k_b = hf[f"{pfx}.self_attn.k_proj.bias"].reshape(num_heads, head_dim)
        k_kernel.append(k_w)
        k_bias.append(k_b)

        v_w = hf[f"{pfx}.self_attn.v_proj.weight"].T.reshape(width, num_heads, head_dim)
        v_b = hf[f"{pfx}.self_attn.v_proj.bias"].reshape(num_heads, head_dim)
        v_kernel.append(v_w)
        v_bias.append(v_b)

        # Out projection: [out, in] -> [in, out] -> [num_heads, head_dim, width]
        o_w = hf[f"{pfx}.self_attn.out_proj.weight"].T.reshape(num_heads, head_dim, width)
        o_b = hf[f"{pfx}.self_attn.out_proj.bias"]
        out_kernel.append(o_w)
        out_bias.append(o_b)

        # LayerNorm 1 (pre-MLP)
        ln1_scale.append(hf[f"{pfx}.layer_norm2.weight"])
        ln1_bias.append(hf[f"{pfx}.layer_norm2.bias"])

        # MLP: HF Linear [out, in] -> Flax Dense [in, out]
        fc1_kernel.append(hf[f"{pfx}.mlp.fc1.weight"].T)
        fc1_bias.append(hf[f"{pfx}.mlp.fc1.bias"])
        fc2_kernel.append(hf[f"{pfx}.mlp.fc2.weight"].T)
        fc2_bias.append(hf[f"{pfx}.mlp.fc2.bias"])

    # --- Post layernorm ---
    enc_norm_scale = hf["vision_model.post_layernorm.weight"]
    enc_norm_bias = hf["vision_model.post_layernorm.bias"]

    hf.clear()  # Free memory.

    # Build nested dict matching big_vision Flax param structure.
    img_params = {
        "embedding": {
            "kernel": conv_kernel,
            "bias": conv_bias,
        },
        "pos_embedding": pos_emb,
        "Transformer": {
            "encoderblock": {
                "LayerNorm_0": {
                    "scale": np.stack(ln0_scale),
                    "bias": np.stack(ln0_bias),
                },
                "MultiHeadDotProductAttention_0": {
                    "query": {"kernel": np.stack(q_kernel), "bias": np.stack(q_bias)},
                    "key": {"kernel": np.stack(k_kernel), "bias": np.stack(k_bias)},
                    "value": {"kernel": np.stack(v_kernel), "bias": np.stack(v_bias)},
                    "out": {"kernel": np.stack(out_kernel), "bias": np.stack(out_bias)},
                },
                "MlpBlock_0": {
                    "Dense_0": {"kernel": np.stack(fc1_kernel), "bias": np.stack(fc1_bias)},
                    "Dense_1": {"kernel": np.stack(fc2_kernel), "bias": np.stack(fc2_bias)},
                },
                "LayerNorm_1": {
                    "scale": np.stack(ln1_scale),
                    "bias": np.stack(ln1_bias),
                },
            },
            "encoder_norm": {
                "scale": enc_norm_scale,
                "bias": enc_norm_bias,
            },
        },
    }

    return img_params
