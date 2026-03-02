import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import numpy as np


class SPFSplatDataModule:
    """
    Simple datamodule for VGGT training with SPFSplatV2-style chunk datasets.
    Compatible with VGGT's Trainer class.
    """
    
    def __init__(
        self,
        roots: list[str],
        batch_size: int = 4,
        num_workers: int = 4,
        num_context_views: int = 2,
        num_target_views: int = 1,
        max_distance_between_context_views: int = 150,
        min_distance_between_context_views: int = 5,
        min_distance_to_context_views: int = 1,
        input_image_shape: tuple[int, int] = (352, 352),
        seed: int = 123,
    ):
        from pathlib import Path
        from spfsplat_dataset import ChunkDataset, ChunkDatasetCfg
        
        self.roots = [Path(r) for r in roots]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        cfg = ChunkDatasetCfg(
            roots=self.roots,
            num_context_views=num_context_views,
            num_target_views=num_target_views,
            min_distance_between_context_views=min_distance_between_context_views,
            max_distance_between_context_views=max_distance_between_context_views,
            min_distance_to_context_views=min_distance_to_context_views,
            make_baseline_1=False,  # Disabled - VGGT handles this
            baseline_min=0.0,  # Disabled
            baseline_max=1e10,  # Disabled
            input_image_shape=(352, 352),
            original_image_shape=(360, 640),
            relative_pose=False,  # Disabled - VGGT handles this
        )
        
        self.train_dataset = ChunkDataset(cfg, stage="train")
        self.val_dataset = ChunkDataset(cfg, stage="val")
        
    def get_loader(self, epoch: int = 0, shuffle: bool = True):
        """Returns a DataLoader compatible with VGGT's Trainer."""
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + epoch)
        else:
            generator = None
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            generator=generator,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
    
    def get_val_loader(self, epoch: int = 0):
        """Returns a validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
    
    @staticmethod
    def collate_fn(batch):
        """Collate function to create batches compatible with VGGT."""
        if len(batch) == 0:
            return {}
            
        context_extrinsics = []
        context_intrinsics = []
        context_images = []
        target_extrinsics = []
        target_intrinsics = []
        target_images = []
        scenes = []
        
        for item in batch:
            context_extrinsics.append(item["context"]["extrinsics"])
            context_intrinsics.append(item["context"]["intrinsics"])
            context_images.append(item["context"]["image"])
            target_extrinsics.append(item["target"]["extrinsics"])
            target_intrinsics.append(item["target"]["intrinsics"])
            target_images.append(item["target"]["image"])
            scenes.append(item["scene"])
        
        # Stack all context and target images
        # VGGT expects [B, S, 3, H, W]
        context_images = torch.stack(context_images)  # [B, num_context, 3, H, W]
        target_images = torch.stack(target_images)    # [B, num_target, 3, H, W]
        
        # Combine context + target for VGGT input
        # [B, num_context + num_target, 3, H, W]
        all_images = torch.cat([context_images, target_images], dim=1)
        
        return {
            # VGGT format: [B, S, 3, H, W]
            "images": all_images,
            # Also provide context/target separately for downstream use
            "context": {
                "extrinsics": torch.stack(context_extrinsics),
                "intrinsics": torch.stack(context_intrinsics),
                "image": context_images,
            },
            "target": {
                "extrinsics": torch.stack(target_extrinsics),
                "intrinsics": torch.stack(target_intrinsics),
                "image": target_images,
            },
            "scene": scenes,
        }
