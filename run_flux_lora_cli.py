#!/usr/bin/env python3
"""Utilities for preparing and training a Flux LoRA."""

import argparse
import shutil
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from imgutils.tagging import get_wd14_tags

DEFAULT_SAMPLE_PROMPTS = [
    "mira_arakai woman portrait",
    "mira_arakai woman in a sunlit library",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset and build Flux LoRA training script",
    )
    parser.add_argument("--raw-photos", required=True, help="Folder with images")
    parser.add_argument("--trigger", required=True, help="Trigger token")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--network-dim", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--num-repeats", type=int, default=10)
    parser.add_argument("--lr", default="1e-4", help="Learning rate")
    parser.add_argument("--output-name", default="mira-flux-v1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--optimizer-type", default="adamw8bit")
    parser.add_argument(
        "--optimizer-args",
        default="--relative_step=False --scale_parameter=False",
    )
    parser.add_argument(
        "--sample-prompt",
        dest="sample_prompts",
        action="append",
        default=[],
        help="Prompt used for sample generation (can be repeated)",
    )
    parser.add_argument("--sample-every-n-steps", type=int, default=1000)
    parser.add_argument(
        "--dataset-dir",
        default="datasets/mira-flux-lora",
        help="Where to store the processed dataset",
    )
    parser.add_argument(
        "--train-output-dir",
        default="outputs",
        help="Where to write training outputs",
    )
    return parser.parse_args()


# --- 2) HELPERS ------------------------------------------------------------


def resize_if_needed(src: Path, dst: Path, short: int) -> None:
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


def prepare_dataset(args: argparse.Namespace) -> None:
    raw = Path(args.raw_photos).expanduser()
    if not raw.is_dir():
        raise FileNotFoundError(f"Raw folder not found: {raw}")

    dst = Path(args.dataset_dir)
    dst.mkdir(parents=True, exist_ok=True)

    imgs = sorted(raw.glob("*.jpg")) + sorted(raw.glob("*.png"))
    print(f"Found {len(imgs)} images in {raw}")

    for img in tqdm(imgs, desc="Resizing"):
        resize_if_needed(img, dst / img.name, args.resolution)

    tagged = sorted(dst.glob("*.jpg")) + sorted(dst.glob("*.png"))
    tag_with_wd14(tagged, args.trigger)

    prompts = args.sample_prompts or DEFAULT_SAMPLE_PROMPTS
    write_sample_prompts(Path("sample_prompts.txt"), prompts)
    print("sample_prompts.txt written")

    toml = Path("dataset.toml")
    toml.write_text(
        f"""[general]
caption_extension = '.txt'
shuffle_caption = false
keep_tokens = 1

[[datasets]]
        resolution   = {args.resolution}
        batch_size   = {args.batch_size}

  [[datasets.subsets]]
  image_dir    = '{dst.resolve()}'
  class_tokens = '{args.trigger}'
  num_repeats  = {args.num_repeats}
"""
    )
    print("dataset.toml written ->", toml)
    print(toml.read_text())


def tag_with_wd14(img_paths, trigger: str, min_prob: float = 0.35) -> None:
    """Tag images with WD-14 Swin-V2 and prepend the trigger."""
    for p in tqdm(img_paths, desc="Tagging"):
        _, gen, chars = get_wd14_tags(
            str(p),
            model_name="SwinV2_v3",
            general_threshold=min_prob,
            character_threshold=0.85,
        )
        tags = list(dict.fromkeys(list(chars) + list(gen)))
        print(f"{p.name} -> {len(tags)} tags: {', '.join(tags)}")
        caption = ", ".join([trigger] + tags)
        p.with_suffix(".txt").write_text(caption)


# --- 4) BUILD TRAIN SCRIPT -------------------------------------------------


def build_train_sh(args: argparse.Namespace) -> None:
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
  --network_module networks.lora_flux --network_dim {args.network_dim} \
  --optimizer_type {args.optimizer_type} \
  --optimizer_args "{args.optimizer_args}" \
  --learning_rate {args.lr} \
  --max_train_epochs {args.max_epochs} \
  --save_every_n_epochs {args.max_epochs//2} \
  --sample_prompts sample_prompts.txt \
  --sample_every_n_steps {args.sample_every_n_steps} \
  --dataset_config dataset.toml \
  --output_dir {args.train_output_dir} \
  --output_name {args.output_name} \
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
    args = parse_args()
    prepare_dataset(args)
    build_train_sh(args)
    # subprocess.run(["bash", "train.sh"], check=True)
    print("\nNext step: ./train.sh")
