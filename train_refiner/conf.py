import torch
from pathlib import Path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
DTYPE_VAE = torch.float32
DTYPE_UNET = torch.float32
IMAGE_SIZE = 336
RE10K_ROOT = Path("/mnt/hdd1/yunsik/re10k")
TRAIN_CHUNKS = list(range(4866))
VAL_CHUNKS = list(range(542))

GRAM_WARMUP_STEPS = 2000
L_IMG_WEIGHT = 1.0
L_LPIPS_WEIGHT = 1.0
L_GRAM_WEIGHT = 1.0

FIXED_TIMESTEP = 200
REFINEMENT_PROMPT = "high quality, natural, realistic, refined, detailed"

CHUNK_CACHE_SIZE = 4

GAMMA_SCHEDULE_START_STEP = 0
GAMMA_SCHEDULE_RAMP_STEPS = 2000

RUN_NAME = "refiner-train-full"
SEED = 42

NUM_EPOCHS = 3

# logging / eval cadence (step is *batch step*, not optimizer step)
LOG_EVERY = 10
VAL_EVERY = 300
NUM_VAL_STEPS = 4

# checkpoint cadence (batch-step 기준)
SAVE_EVERY = 1000

# optimizer/scheduler
BASE_LR = 5e-6
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_WEIGHT_DECAY = 1e-2
ADAM_EPS = 1e-8
TRAIN_BS = 2
VAL_BS = 1


# stability
MAX_GRAD_NORM = 1.0

# Difix ckpt (difix3D에서 받은 refiner 체크포인트 경로)
DIFIX_PRETRAINED_PKL = None

# 선택 옵션
DIFIX_TIMESTEP = 199
DIFIX_MV_UNET = False
ENABLE_XFORMERS = True
ENABLE_GRAD_CKPT = False
ALLOW_TF32 = True
LORA_RANK = 4