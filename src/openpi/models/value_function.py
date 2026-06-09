"""Value function model for pi_0.6* (distributional value prediction).

Architecture: SigLIP So400m/14 (400M) + single-expert Gemma 270M + value_head (width -> 201 bins).
Loss: cross-entropy over discretized return bins (Eq.1).
"""

import dataclasses

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.nnx import bridge as nnx_bridge
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import siglip as _siglip
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class ValueFunctionConfig(_model.BaseValueFunctionModelConfig):
    """Config for distributional value function (pi_0.6* / RECAP)."""

    model_type: _model.ModelType = _model.ModelType.VALUE_FUNCTION
    dtype: str = "bfloat16"
    gemma_variant: _gemma.Variant = "gemma_270m"
    discrete_state_input: bool = True
    num_bins: int = 201
    return_min: float = -1.0
    return_max: float = 0.0
    # Action (or state) space dimension needed by inputs_spec and data transforms.
    action_dim: int = 14
    action_horizon: int = 1 # No need for multi-step value prediction

    # Inherited defaults.
    max_token_len: int = 200  # Following Pi0.5 pattern (needs more tokens for discrete state)

    @override
    def create(self, rng: at.KeyArrayLike) -> "ValueFunction":
        return ValueFunction(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Binned_Value]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        
        value_spec = jax.ShapeDtypeStruct([batch_size], jnp.int32)

        return observation_spec, value_spec


class ValueFunction(_model.BaseValueFunctionModel):
    def __init__(self, config: "ValueFunctionConfig", rngs: nnx.Rngs):
        super().__init__(config.max_token_len, config.num_bins, config.return_min, config.return_max)
        gemma_config = _gemma.get_config(config.gemma_variant)
        self.gemma_config = gemma_config

        # SigLIP image encoder - same as Pi0, So400m/14, ~400M params.
        # Note: num_classes projects to gemma_config.width (640 for gemma_270m).
        # Follow Pi0 pattern: do NOT pass rngs to ToNNX constructor, only to lazy_init.
        self.img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=gemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        self.img.lazy_init(jnp.zeros((1, 224, 224, 3)), train=False, rngs=rngs)

        # Single-expert Gemma (no action expert, no adaRMS).
        # configs is a list with one element - single expert.
        # Follow Pi0 pattern: do NOT pass rngs to ToNNX constructor, only to lazy_init.
        self.llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[gemma_config],
                embed_dtype=config.dtype,
                adarms=False,
            )
        )
        self.llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False],
        )

        # Note: No state_proj layer. Following Pi0.5 pattern, state is tokenized
        # as part of the language tokens during data preprocessing.

        # Value head MLP: paper Eq.1 — V_θ(o) = softmax(MLP(f_θ(o)))
        self.value_head_fc1 = nnx.Linear(gemma_config.width, gemma_config.width, rngs=rngs)
        self.value_head_fc2 = nnx.Linear(gemma_config.width, config.num_bins, rngs=rngs)

        self.num_bins = config.num_bins
        self.return_min = config.return_min
        self.return_max = config.return_max

    def embed_and_forward(self, observation: _model.Observation):
        """Full forward pass: images + language -> Gemma -> value logits."""

        # 1. Encode images with SigLIP.
        image_tokens_list = []
        image_masks_list = []
        # image token
        for key in sorted(observation.images.keys()):
            img = observation.images[key]
            img_mask = observation.image_masks[key]
            tokens, _ = self.img(img, train=False)  # [b, num_patches, width]
            # Mask: expand image_masks [b] -> [b, 1] -> [b, num_patches]
            num_patches = tokens.shape[1]
            mask = jnp.repeat(img_mask[:, None], num_patches, axis=1)  # [b, num_patches]
            image_tokens_list.append(tokens)
            image_masks_list.append(mask)

        # 2. Encode language with Gemma embedder.
        # Note: Following Pi0.5 pattern, state is already tokenized as part of
        # tokenized_prompt during data preprocessing (discrete state input).
        lang_tokens = self.llm(observation.tokenized_prompt, method="embed")  # [b, seq, width]
        lang_mask = observation.tokenized_prompt_mask  # [b, seq]

        # print the squred L2 norm of the language tokens for debugging into [b, seq]
        # jax.debug.print("Language tokens L2 norm squared mean for each token: {}", 
        #         jnp.sum(lang_tokens**4, axis=-1))
        # jax.debug.print("Language mask sum (number of valid tokens) for each example: {}", 
        #         jnp.sum(lang_mask, axis=-1))

        # 3. Concatenate all tokens: [images, language] (state is in language tokens).
        all_tokens = jnp.concatenate(image_tokens_list + [lang_tokens], axis=1)
        all_mask = jnp.concatenate(image_masks_list + [lang_mask], axis=1)

        # 4. Attention mask: fully bidirectional (ar_mask all False).
        ar_mask = jnp.zeros_like(all_mask)  # [b, seq_total]
        attn_mask = make_attn_mask(all_mask, ar_mask)  # [b, 1, seq_total, seq_total]

        # 5. Positions.
        positions = jnp.cumsum(all_mask.astype(jnp.int32), axis=1) - 1
        positions = jnp.maximum(positions, 0)

        # 6. Forward through Gemma (single expert: pass list with one element).
        [outputs], _ = self.llm(
            [all_tokens],
            mask=attn_mask,
            positions=positions,
        )
        # outputs: [b, seq_total, width]

        # 7. Extract the last valid (non-padding) token representation.
        #    The sequence is [img_patches... | lang_tokens... | padding...],
        #    so the last valid token is the final real prompt token ("Value: ").
        seq_len = all_mask.shape[1]
        idxs = jnp.arange(seq_len)
        last_valid_idx = jnp.argmax(jnp.where(all_mask, idxs, -1), axis=1)

        batch_indices = jnp.arange(outputs.shape[0])
        pooled = outputs[batch_indices, last_valid_idx]  # [b, width]

        # Debug: print the squared of outputs and last valid idx
        # image_mask_length = sum(mask.shape[1] for mask in image_masks_list)
        # jax.debug.print("Outputs L2 norm squared mean for each token: {}", 
        #         jnp.sum(outputs[:, image_mask_length:] ** 4, axis=-1))
        # jax.debug.print("Last valid token index for each example: {}", last_valid_idx - image_mask_length)

        # 8. Value head MLP -> logits over bins (paper Eq.1).
        x = jax.nn.gelu(self.value_head_fc1(pooled))
        logits = self.value_head_fc2(x)  # [b, num_bins]
        return logits

    def compute_loss(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        binned_value: _model.Binned_Value,
        *,
        train: bool = True,
    ):
        """Cross-entropy loss over discretized return bins (paper Eq.1)."""
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=tuple(observation.images.keys())
        )
        logits = self.embed_and_forward(observation)  # [b, num_bins]

        # Target: observation.binned_value is int32 [b]
        target_bins = binned_value  # [b]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_bins)  # [b]

        # BaseModel.compute_loss expects shape [*b] - our [b] is compatible.
        return loss

    def infer_value(
        self,
        rng: jax.Array,
        observation: _model.Observation,
    ):
        """Predict continuous value (weighted expectation over bins)."""
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=tuple(observation.images.keys())
        )
        logits = self.embed_and_forward(observation)  # [b, num_bins]
        probs = jax.nn.softmax(logits, axis=-1)  # [b, num_bins]
        bin_centers = jnp.linspace(self.return_min, self.return_max, self.num_bins)  # [num_bins]
        value = jnp.sum(probs * bin_centers[None, :], axis=-1)  # [b]
        return value  # [b]
