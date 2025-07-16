#!/usr/bin/env python3
"""Utilities for preparing and training a Flux LoRA."""

import shutil
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from imgutils.tagging import get_wd14_tags

# --- 1) USER CONFIGURATION -------------------------------------------------
RAW_PHOTOS = (
    "/content/drive/MyDrive/Mira Arakai/Mira_URPMORXL/dataset/images/10_mira"
)  # noqa: E501

TRIGGER = "mira_arakai woman"  # your new trigger token
RESOLUTION = 768
NETWORK_DIM = 64
MAX_EPOCHS = 6
NUM_REPEATS = 10
LR = "1e-4"
OUTPUT_NAME = "mira-flux-v1"
BATCH_SIZE = 1

OPTIMIZER_TYPE = "adamw8bit"
OPTIMIZER_ARGS = "--relative_step=False --scale_parameter=False"
SAMPLE_PROMPTS = [
    "mira_arakai woman portrait",
    "mira_arakai woman in a sunlit library",
]

SAMPLE_EVERY_N_STEPS = 1000

# --- 2) HELPERS ------------------------------------------------------------


def resize_if_needed(src: Path, dst: Path, short: int = RESOLUTION) -> None:
    """Copy src to dst, resizing so the shorter side equals ``short``."""
    with Image.open(src) as im:
        w, h = im.size
        s = min(w, h)
        if s == short:
            shutil.copy(src, dst)
            return
        scale = short / s
        im.resize(
            (round(w * scale), round(h * scale)),
            Image.LANCZOS,
        ).save(dst)


def write_sample_prompts(path: Path, prompts) -> None:
    """Write each prompt on a new line."""
    path.write_text("\n".join(prompts))

# --- 3) PREP DATASET -------------------------------------------------------


def prepare_dataset() -> None:
    raw = Path(RAW_PHOTOS).expanduser()
    if not raw.is_dir():
        raise FileNotFoundError(f"Raw folder not found: {raw}")

    dst = Path("datasets/mira-flux-lora")
    dst.mkdir(parents=True, exist_ok=True)

    imgs = sorted(raw.glob("*.jpg")) + sorted(raw.glob("*.png"))
    print(f"Found {len(imgs)} images in {raw}")

    for img in tqdm(imgs, desc="Resizing"):
        resize_if_needed(img, dst / img.name)

    tagged = sorted(dst.glob("*.jpg")) + sorted(dst.glob("*.png"))
    tag_with_wd14(tagged)

    write_sample_prompts(Path("sample_prompts.txt"), SAMPLE_PROMPTS)
    print("sample_prompts.txt written")

    toml = Path("dataset.toml")
    toml.write_text(
        f"""[general]
caption_extension = '.txt'
shuffle_caption = false
keep_tokens = 1

[[datasets]]
resolution   = {RESOLUTION}
batch_size   = {BATCH_SIZE}

  [[datasets.subsets]]
  image_dir    = '{dst.resolve()}'
  class_tokens = '{TRIGGER}'
  num_repeats  = {NUM_REPEATS}
"""
    )
    print("dataset.toml written ->", toml)
    print(toml.read_text())


def tag_with_wd14(
    img_paths,
    trigger: str = TRIGGER,
    min_prob: float = 0.35,
) -> None:
    """Tag images with WD-14 Swin-V2 and prepend the trigger."""
    for p in tqdm(img_paths, desc="Tagging"):
        _, gen, chars = get_wd14_tags(
            str(p),
            model_name="SwinV2_v3",
            general_threshold=min_prob,
            character_threshold=0.85,
        )
        tags = list(dict.fromkeys(list(chars) + list(gen)))
        print(f"{p.name} -> {len(tags)} tags: {trigger}, {', '.join(tags)}")
        caption = ", ".join([trigger] + tags)
        p.with_suffix(".txt").write_text(caption)

# --- 4) BUILD TRAIN SCRIPT -------------------------------------------------


def build_train_sh() -> None:
    sh = Path("train.sh")
    sh.write_text(
        f"""#!/usr/bin/env bash
accelerate launch \
  --mixed_precision bf16 \
  ../sd-scripts/flux_train_network.py \
  --pretrained_model_name_or_path models/unet/flux1-dev-fp8.safetensors \
  --clip_l  models/clip/clip_l.safetensors \
  --t5xxl   models/clip/t5xxl_fp8.safetensors \
  --ae      models/vae/ae.sft \
  --cache_latents_to_disk --save_model_as safetensors \
  --network_module networks.lora_flux --network_dim {NETWORK_DIM} \
  --optimizer_type {OPTIMIZER_TYPE} \
  --optimizer_args "{OPTIMIZER_ARGS}" \
  --learning_rate {LR} \
  --max_train_epochs {MAX_EPOCHS} \
  --save_every_n_epochs {MAX_EPOCHS//2} \
  --sample_prompts sample_prompts.txt \
  --sample_every_n_steps {SAMPLE_EVERY_N_STEPS} \
  --dataset_config dataset.toml \
  --output_dir outputs \
  --output_name {OUTPUT_NAME} \
  --timestep_sampling shift \
  --guidance_scale 1 \
  --fp8_base --highvram
"""
    )
    sh.chmod(0o755)
    print("train.sh written ->", sh)
    print(Path("train.sh").read_text())


# --- 5) MAIN --------------------------------------------------------------

if __name__ == "__main__":
    prepare_dataset()
    build_train_sh()
    # subprocess.run(["bash", "train.sh"], check=True)
    print("\nNext step: ./train.sh")
