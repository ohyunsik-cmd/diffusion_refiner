import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import repeat
from PIL import Image
from torch.utils.data import IterableDataset

from io import BytesIO


@dataclass
class ChunkDatasetCfg:
    roots: list[Path]
    num_context_views: int = 2
    num_target_views: int = 1
    min_distance_between_context_views: int = 5
    max_distance_between_context_views: int = 150  # Increased for more samples
    min_distance_to_context_views: int = 1  # Reduced for more flexibility
    make_baseline_1: bool = False  # Don't normalize baseline - let VGGT handle it
    baseline_min: float = 0.0  # Disable filtering
    baseline_max: float = 1e10  # Disable filtering
    input_image_shape: tuple[int, int] = (352, 352)
    original_image_shape: tuple[int, int] = (360, 640)
    relative_pose: bool = False  # Don't convert to relative poses


class ChunkDataset(IterableDataset):
    def __init__(
        self,
        cfg: ChunkDatasetCfg,
        stage: Literal["train", "val", "test"],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        self.resize = tf.Resize(cfg.input_image_shape)

        self.chunks = []
        for root in cfg.roots:
            # Handle both "val" and "validation" folder names
            if stage == "val" and not (root / "val").exists():
                stage_path = root / "validation"
            else:
                stage_path = root / stage
                
            root_chunks = sorted(
                [path for path in stage_path.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def convert_poses(self, poses, original_size):
        b, _ = poses.shape

        # Get original image size
        orig_h, orig_w = original_size

        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = poses[:, 6:].reshape(b, 3, 4)
        return w2c.inverse(), intrinsics

    def scale_intrinsics(self, intrinsics, original_size, target_size):
        """Scale intrinsics based on image resizing."""
        orig_h, orig_w = original_size
        target_h, target_w = target_size

        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        scaled_intrinsics = intrinsics.clone()
        scaled_intrinsics[:, 0, 0] *= scale_x  # fx
        scaled_intrinsics[:, 1, 1] *= scale_y  # fy
        scaled_intrinsics[:, 0, 2] *= scale_x  # cx
        scaled_intrinsics[:, 1, 2] *= scale_y  # cy

        return scaled_intrinsics

    def convert_images(self, images, return_original_size=False):
        """Convert images with optional original size tracking."""
        torch_images = []
        original_sizes = []
        
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            w, h = image.size
            original_sizes.append((h, w))
            
            image = self.to_tensor(image)
            image = self.resize(image)
            torch_images.append(image)
        
        if return_original_size:
            return torch.stack(torch_images), original_sizes
        return torch.stack(torch_images)

    def camera_normalization(self, reference_extrinsics, extrinsics):
        """Normalize camera pose - make reference the origin."""
        w2c_normalized = reference_extrinsics.inverse() @ extrinsics
        return w2c_normalized

    def sample_views(self, num_views):
        # Always use consistent number of views (not variable in test)
        max_gap = self.cfg.max_distance_between_context_views
        min_gap = self.cfg.min_distance_between_context_views

        max_gap = min(num_views - 1, max_gap)
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        
        if max_gap < min_gap:
            raise ValueError("Not enough frames")

        context_gap = torch.randint(int(min_gap), int(max_gap) + 1, size=tuple()).item()

        index_context_left = torch.randint(
            int(num_views - context_gap),
            size=tuple(),
        ).item()
        index_context_right = index_context_left + context_gap

        # Fixed number of target views (not variable)
        index_target = torch.randint(
            int(index_context_left + self.cfg.min_distance_to_context_views),
            int(index_context_right + 1 - self.cfg.min_distance_to_context_views),
            size=(self.cfg.num_target_views,),
        )

        context_indices = torch.tensor([index_context_left, index_context_right])
        
        return context_indices, index_target

    def __iter__(self):
        if self.stage in ("train", "val"):
            self.chunks = self.shuffle(self.chunks)

        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            chunk = torch.load(chunk_path, weights_only=True)

            if self.stage in ("train", "val"):
                chunk = self.shuffle(chunk)

            for example in chunk:
                # Get original camera poses
                cameras = example["cameras"]
                scene = example["key"]
                num_views = cameras.shape[0]

                # Get original image size (assuming all images are same size)
                first_image = example["images"][0]
                first_image_pil = Image.open(BytesIO(first_image.numpy().tobytes()))
                orig_w, orig_h = first_image_pil.size
                original_size = (orig_h, orig_w)

                # Convert poses
                extrinsics, intrinsics = self.convert_poses(cameras, original_size)

                try:
                    context_indices, target_indices = self.sample_views(num_views)
                except ValueError:
                    continue

                try:
                    # Get context images and original sizes
                    context_images_list = [
                        example["images"][index.item()] for index in context_indices
                    ]
                    context_images, orig_sizes = self.convert_images(
                        context_images_list, return_original_size=True
                    )
                    
                    # Get target images
                    target_images_list = [
                        example["images"][index.item()] for index in target_indices
                    ]
                    target_images = self.convert_images(target_images_list)
                except (IndexError, OSError):
                    continue

                # Scale intrinsics based on resizing
                # Use first context image's original size (they should all be similar)
                target_size = self.cfg.input_image_shape
                scaled_intrinsics = self.scale_intrinsics(
                    intrinsics, original_size, target_size
                )

                # Baseline normalization (optional, disabled by default)
                context_extrinsics = extrinsics[context_indices]
                if self.cfg.make_baseline_1:
                    a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                        continue
                    extrinsics[:, :3, 3] /= scale
                    scale_factor = scale
                else:
                    scale_factor = 1.0

                # Relative pose (optional, disabled by default)
                if self.cfg.relative_pose:
                    extrinsics = self.camera_normalization(extrinsics[context_indices][0:1], extrinsics)

                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": scaled_intrinsics[context_indices],
                        "image": context_images,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": scaled_intrinsics[target_indices],
                        "image": target_images,
                    },
                    "scene": scene,
                    "scale": scale_factor,
                    "original_intrinsics": intrinsics,  # Keep original for reference
                }
                yield example
