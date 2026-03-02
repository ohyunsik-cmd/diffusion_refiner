import torch
from ..conf import IMAGE_SIZE, RE10K_ROOT, CHUNK_CACHE_SIZE
from ..util.crop_shim import rescale_and_crop
import torchvision.transforms as tf
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import random
import hashlib
from pathlib import Path

def build_chunk_index(root, folder, chunks):
    """Build precomputed index mapping global idx -> (chunk_path, local_idx)."""
    index = []
    for chunk_idx in tqdm(chunks, desc="Building chunk index"):
        chunk_path = root / folder / f"{chunk_idx:06d}.torch"
        if chunk_path.exists():
            chunk = torch.load(chunk_path, weights_only=True)
            for local_idx in range(len(chunk)):
                index.append((chunk_path, local_idx))
    return index

def make_index_cache_path(root: Path, folder: str, chunks: list[int], image_size: int) -> Path:
    """
    Build a unique cache filename for (folder, chunks, image_size).
    This prevents stale index_cache.pt from being reused when TRAIN_CHUNKS/VAL_CHUNKS changes.
    """
    # ex) folder=train, n=4866, min=0, max=4865, sz=336
    sig = f"{folder}|n={len(chunks)}|min={min(chunks)}|max={max(chunks)}|sz={image_size}"
    h = hashlib.md5(sig.encode("utf-8")).hexdigest()[:10]
    return root / f"index_cache_{folder}_{h}.pt"

class ChunkCache:
    """LRU cache for torch chunk files to avoid repeated disk I/O."""
    
    def __init__(self, max_size=CHUNK_CACHE_SIZE):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
    
    def get(self, chunk_path):
        chunk_path_str = str(chunk_path)
        if chunk_path_str in self.cache:
            self.access_order.remove(chunk_path_str)
            self.access_order.append(chunk_path_str)
            return self.cache[chunk_path_str]
        
        chunk = torch.load(chunk_path, map_location="cpu")
        
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[chunk_path_str] = chunk
        self.access_order.append(chunk_path_str)
        return chunk
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()
        

class RE10KDataset(Dataset):
    def __init__(self, chunks, root=RE10K_ROOT, image_size=IMAGE_SIZE, folder="train", 
                 precomputed_index=None, use_cache=True):
        self.chunks = chunks
        self.root = root
        self.folder = folder
        self.image_size = image_size
        self.to_tensor = tf.ToTensor()
        self.resize = tf.Resize((image_size, image_size))
        self.use_cache = use_cache
        self.chunk_cache = ChunkCache(max_size=CHUNK_CACHE_SIZE)
        if precomputed_index is not None:
            self.index = precomputed_index
        else:
            self.index = build_chunk_index(root, folder, chunks)
    
    def __len__(self):
        return len(self.index)
    
    def get_chunk_and_idx(self, idx):
        return self.index[idx]
    
    def __getitem__(self, idx):
        max_chunk_tries = 20
        max_total_tries = 60

        base_idx = idx
        last_err = None

        for _ in range(max_total_tries):
            chunk_path, sample_idx = self.get_chunk_and_idx(idx)

            try:
                chunk = self.chunk_cache.get(chunk_path) if self.use_cache else torch.load(chunk_path, map_location="cpu")
                n = len(chunk)

                cand = [sample_idx]
                if n > 1:
                    near = [(sample_idx + k) % n for k in range(1, min(max_chunk_tries, n))]
                    rand = random.sample(range(n), k=min(max_chunk_tries, n))
                    cand += near + rand
                    # uniq
                    seen = set()
                    cand = [x for x in cand if (x not in seen and not seen.add(x))]

                tried = 0
                for sidx in cand:
                    if tried >= max_chunk_tries:
                        break
                    tried += 1
                    try:
                        return self.load_sample(chunk_path, sidx)
                    except Exception as e:
                        last_err = e
                        continue

                idx = (idx + 1) % len(self)
                continue

            except Exception as e:
                last_err = e
                idx = (idx + 1) % len(self)
                continue

        raise RuntimeError(
            f"Failed to fetch a valid sample after {max_total_tries} tries. "
            f"start_idx={base_idx}, last_err={repr(last_err)}"
        )
    
    def load_sample(self, chunk_path, sample_idx):
        chunk = self.chunk_cache.get(chunk_path) if self.use_cache else torch.load(chunk_path, weights_only=True)
        item = chunk[sample_idx]

        cameras = item["cameras"]
        N = cameras.shape[0]
        if N < 3:
            raise AssertionError(f"Not enough views: {N}")
        
        def convert_poses(poses: torch.Tensor):
            # poses: [V, 18] (fx fy cx cy ... + 3x4)
            b, _ = poses.shape
            intrinsics = torch.eye(3, dtype=torch.float32).repeat(b, 1, 1)
            fx, fy, cx, cy = poses[:, :4].T
            intrinsics[:, 0, 0] = fx
            intrinsics[:, 1, 1] = fy
            intrinsics[:, 0, 2] = cx
            intrinsics[:, 1, 2] = cy

            w2c = torch.eye(4, dtype=torch.float32).repeat(b, 1, 1)
            w2c[:, :3] = poses[:, 6:].reshape(b, 3, 4)
            return w2c, intrinsics

        # --- 1) window=70 선택 ---
        W = 70
        if N <= W:
            start = 0
            end = N
        else:
            start = random.randint(0, N - W)
            end = start + W

        # window 내 후보 인덱스 (원본 기준)
        win = torch.arange(start, end)

        # --- 2) window 안에서 baseline 있게 3개 뽑기 ---
        # target은 window 중앙 근처, context는 양끝 쪽으로 (간단하면서 효과 좋음)
        mid = (start + end) // 2
        t = mid + random.randint(-W//8, W//8)
        t = max(start, min(end - 1, t))

        c0 = start + random.randint(0, W//6)
        c1 = end - 1 - random.randint(0, W//6)

        # 겹치면 조정
        sel = [c0, c1, t]
        sel = list(dict.fromkeys(sel))
        if len(sel) < 3:
            # fallback: window에서 서로 멀게 3개
            sel = [start, (start+end)//2, end-1]

        sel = torch.tensor(sel, dtype=torch.long)
        sel, _ = torch.sort(sel)  # 디버깅 안정

        # --- 3) 카메라 파싱(전체가 아니라 sel만) ---
        poses = cameras[sel]  # [3, ?]
        # convert_poses는 너 코드 그대로 쓰면 됨
        extrinsics, intrinsics = convert_poses(poses)

        # --- 4) 이미지는 sel 3장만 디코딩 ---
        images = []
        for i in sel.tolist():
            img = item["images"][i]
            from io import BytesIO
            img_pil = Image.open(BytesIO(img.numpy().tobytes())).convert("RGB")
            images.append(self.to_tensor(img_pil))
        images = torch.stack(images)  # [3,3,H,W]

        # --- 5) crop/resize는 3장 기준으로만 ---
        H0, W0 = images.shape[-2], images.shape[-1]
        intrinsics_px = intrinsics.clone()
        intrinsics_px[:, 0, 0] *= W0
        intrinsics_px[:, 1, 1] *= H0
        intrinsics_px[:, 0, 2] *= W0
        intrinsics_px[:, 1, 2] *= H0

        images, intrinsics = rescale_and_crop(images, intrinsics_px, (self.image_size, self.image_size))

        return {
            "extrinsics": extrinsics,     # [3,4,4]
            "intrinsics": intrinsics,     # [3,3,3]
            "images": images,             # [3,3,H,W]
            "num_views": 3,               # 고정
            "key": item["key"],
            "sel_idx": sel,               # (옵션) 디버깅용: 원본 프레임 인덱스
            "win_start": start,           # (옵션)
        }