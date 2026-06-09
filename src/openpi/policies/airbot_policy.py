import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# Per-arm sign mask for spatial left-right symmetry:
# Joints 1, 4, 5, 6 (1-indexed) are negated; joints 2, 3 and gripper (7) are unchanged.
_AIRBOT_ARM_JOINT_SIGN = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)


def _flip_image_horizontal(image: np.ndarray) -> np.ndarray:
    """Horizontally flip an image in either (H, W, C) or (C, H, W) format."""
    if image.ndim == 3 and image.shape[0] == 3:
        # (C, H, W) format — width is axis 2
        return image[:, :, ::-1].copy()
    # (H, W, C) format — width is axis 1
    return image[:, ::-1, :].copy()


@dataclasses.dataclass(frozen=True)
class AirbotSymmetryAugmentation(transforms.DataTransformFn):
    """Spatial left-right symmetry augmentation for dual-arm Airbot robot.

    With probability ``prob``, performs a mirror-symmetry transform:
    - Swaps left arm and right arm joints in ``state`` and ``actions``.
    - Negates joints 1, 4, 5, 6 (1-indexed) for each arm (joints 2, 3 and
      the gripper joint 7 are left unchanged).
    - Swaps ``left_wrist_0_rgb`` <-> ``right_wrist_0_rgb`` and
      horizontally flips all camera images.

    This transform is placed in ``repack_transforms`` so it is applied
    only during training (dataset loading) and never during inference.
    """

    prob: float = 0.5

    def __call__(self, data: dict) -> dict:
        if np.random.random() >= self.prob:
            return data

        sign = _AIRBOT_ARM_JOINT_SIGN  # shape (7,)
        data = dict(data)  # shallow copy so we can reassign keys

        # --- state ---
        if "state" in data:
            s = np.array(data["state"], dtype=np.float64)
            left_s, right_s = s[..., :7].copy(), s[..., 7:14].copy()
            s = s.copy()
            s[..., :7] = right_s * sign
            s[..., 7:14] = left_s * sign
            data["state"] = s

        # --- actions ---
        if "actions" in data:
            a = np.array(data["actions"], dtype=np.float64)
            left_a, right_a = a[..., :7].copy(), a[..., 7:14].copy()
            a = a.copy()
            a[..., :7] = right_a * sign
            a[..., 7:14] = left_a * sign
            data["actions"] = a

        # --- wrist cameras: swap then horizontally flip both ---
        if "left_wrist_0_rgb" in data and "right_wrist_0_rgb" in data:
            orig_left = _flip_image_horizontal(np.asarray(data["left_wrist_0_rgb"]))
            orig_right = _flip_image_horizontal(np.asarray(data["right_wrist_0_rgb"]))
            data["left_wrist_0_rgb"] = orig_right   # swapped
            data["right_wrist_0_rgb"] = orig_left   # swapped

        # --- base camera: horizontal flip only ---
        if "base_0_rgb" in data:
            data["base_0_rgb"] = _flip_image_horizontal(np.asarray(data["base_0_rgb"]))

        return data


# 5 non-identity permutations of the 3 RGB channels.
_RGB_NON_IDENTITY_PERMS = (
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def _permute_image_channels(image: np.ndarray, perm: tuple[int, int, int]) -> np.ndarray:
    """Permute the color channels of an image in either (H, W, C) or (C, H, W) format."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 3:
        return arr[list(perm), :, :].copy()
    return arr[..., list(perm)].copy()


@dataclasses.dataclass(frozen=True)
class AirbotChannelPermutationAugmentation(transforms.DataTransformFn):
    """Random RGB-channel permutation for color generalization.

    With probability ``prob``, samples a non-identity permutation of the 3
    color channels (5 possibilities, uniform) and applies the SAME permutation
    to all camera streams in the sample (base + wrist). Applied via
    ``repack_transforms`` so it runs only during training, never at inference.
    """

    prob: float = 0.5
    image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    def __call__(self, data: dict) -> dict:
        if np.random.random() >= self.prob:
            return data
        perm = _RGB_NON_IDENTITY_PERMS[np.random.randint(len(_RGB_NON_IDENTITY_PERMS))]
        data = dict(data)
        for key in self.image_keys:
            if key in data:
                data[key] = _permute_image_channels(data[key], perm)
        return data


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AirbotInputs(transforms.DataTransformFn):
    action_dim: int = 32
    num_bins: int = 201

    def __call__(self, data: dict) -> dict:

        state = transforms.pad_to_dim(data["state"], self.action_dim)

        image_dict = {}
        image_mask_dict = {}
        image_name = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        for name in image_name:
            if name in data:
                img = _parse_image(data[name])
                # No need for transformation
                # img = img[..., ::-1]  # BGR -> RGB
                image_dict[name] = img
                image_mask_dict[name] = np.True_

        # # debug
        # import uuid
        # from PIL import Image
        # import os
        # debug_dir = "./debug"
        # os.makedirs(debug_dir, exist_ok=True)
        # uid = uuid.uuid4().hex[:8]
        # for name in image_name:
        #     Image.fromarray(image_dict[name]).save(
        #         f"{debug_dir}/{name}_{uid}.jpg",
        #         quality=95
        #     )

        inputs = {
            "state": state,
            "image": image_dict,
            "image_mask": image_mask_dict,
        }
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "binned_value" in data:
            # For value function, clamp binned_value to valid range [0, num_bins).
            data["binned_value"] = data["binned_value"] % self.num_bins
            inputs["binned_value"] = np.asarray(data["binned_value"], dtype=np.int32)
        if "stage" in data:
            inputs["stage"] = np.asarray(data["stage"], dtype=np.int32)
        if "advantage" in data:
            inputs["advantage"] = data["advantage"]
        if "intervention" in data:
            inputs["intervention"] = data["intervention"]
        return inputs


@dataclasses.dataclass(frozen=True)
class AirbotOutputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            return {"actions": np.asarray(data["actions"])}
        elif "binned_value" in data:
            return {"binned_value": np.asarray(data["binned_value"], dtype=np.int32)}
        else:
            raise ValueError("AirbotOutputs expects either 'actions' or 'binned_value' in the input data.")