import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, Int64
from PIL import Image
from torch import Tensor
from typing import Callable, Literal, TypedDict

Stage = Literal["train", "val", "test"]

class BatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    image: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width
    near: Float[Tensor, "batch _"]  # batch view
    far: Float[Tensor, "batch _"]  # batch view
    index: Int64[Tensor, "batch _"]  # batch view
    overlap: Float[Tensor, "batch _"]  # batch view


class BatchedExample(TypedDict, total=False):
    target: BatchedViews
    context: BatchedViews
    scene: list[str]


class UnbatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "_ 4 4"]
    intrinsics: Float[Tensor, "_ 3 3"]
    image: Float[Tensor, "_ 3 height width"]
    near: Float[Tensor, " _"]
    far: Float[Tensor, " _"]
    index: Int64[Tensor, " _"]


class UnbatchedExample(TypedDict, total=False):
    target: UnbatchedViews
    context: UnbatchedViews
    scene: str


# A data shim modifies the example after it's been returned from the data loader.
DataShim = Callable[[BatchedExample], BatchedExample]

AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(images, intrinsics, shape):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    images = images[..., :, row:row+h_out, col:col+w_out]

    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 2]  = w_out / 2    # cx
    intrinsics[..., 1, 2] -= row   # cy
    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    # print(h_in, w_in, h_out, w_out)
    assert h_out <= h_in and w_out <= w_in

    # normalize intrinsics to original image size
    intrinsics = intrinsics.clone()
    
    intrinsics[...,0,0] *= w_in
    intrinsics[...,1,1] *= h_in
    intrinsics[...,0,2] *= w_in
    intrinsics[...,1,2] *= h_in
    

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    
    # ✅ (1) intrinsics resize 보정 추가
    scale_y = h_scaled / h_in
    scale_x = w_scaled / w_in
    
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= scale_x   # fx
    intrinsics[..., 1, 1] *= scale_y   # fy
    intrinsics[..., 0, 2] *= scale_x   # cx
    intrinsics[..., 1, 2] *= scale_y   # cy
    intrinsics[..., 0, 1] *= scale_x   # skew 있으면 같이 (보통 0)
    intrinsics[..., 1, 0] *= scale_y   # (보통 0)

    assert h_scaled == h_out or w_scaled == w_out
    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape)
    return {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }


def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
    }
