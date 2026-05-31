"""Intentional degradation experiments for open-source image models."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_MODES = ("baseline", "low_steps", "low_guidance", "high_guidance", "latent_noise", "latent_dropout")


def slugify(text: str, max_length: int = 64) -> str:
    """Create a filesystem-friendly name from a prompt."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].strip("-") or "prompt"


def choose_device() -> str:
    """Choose the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(model_id: str, device: str):
    """Load a Stable Diffusion pipeline from Hugging Face."""
    from diffusers import StableDiffusionPipeline

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def mode_settings(mode: str) -> dict[str, Any]:
    """Return generation settings for each degradation condition."""
    settings = {
        "baseline": {"num_inference_steps": 30, "guidance_scale": 7.5},
        "low_steps": {"num_inference_steps": 8, "guidance_scale": 7.5},
        "low_guidance": {"num_inference_steps": 30, "guidance_scale": 2.0},
        "high_guidance": {"num_inference_steps": 30, "guidance_scale": 15.0},
        "latent_noise": {"num_inference_steps": 30, "guidance_scale": 7.5},
        "latent_dropout": {"num_inference_steps": 30, "guidance_scale": 7.5},
    }
    if mode not in settings:
        raise ValueError(f"Unknown mode: {mode}")
    return settings[mode]


def degradation_callback(mode: str, noise_scale: float, dropout_rate: float) -> Callable[..., dict[str, Any]] | None:
    """Create a callback that perturbs latents during denoising."""
    if mode not in {"latent_noise", "latent_dropout"}:
        return None

    def callback(pipe, step: int, timestep: int, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        latents = callback_kwargs["latents"]
        total_steps = pipe._num_timesteps
        active_window = total_steps * 0.2 <= step <= total_steps * 0.8

        if active_window and mode == "latent_noise":
            latents = latents + torch.randn_like(latents) * noise_scale

        if active_window and mode == "latent_dropout":
            mask = torch.rand_like(latents) > dropout_rate
            latents = latents * mask

        callback_kwargs["latents"] = latents
        return callback_kwargs

    return callback


def generate_variant(
    pipe,
    prompt: str,
    mode: str,
    seed: int,
    output_path: Path,
    height: int,
    width: int,
    noise_scale: float = 0.08,
    dropout_rate: float = 0.08,
) -> dict[str, Any]:
    """Generate and save one prompt under one degradation condition."""
    settings = mode_settings(mode)
    generator_device = "cpu" if pipe.device.type == "mps" else pipe.device.type
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    callback = degradation_callback(mode, noise_scale=noise_scale, dropout_rate=dropout_rate)

    kwargs = {
        "prompt": prompt,
        "generator": generator,
        "height": height,
        "width": width,
        **settings,
    }
    if callback is not None:
        kwargs["callback_on_step_end"] = callback
        kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    with torch.inference_mode():
        image = pipe(**kwargs).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return {
        "prompt": prompt,
        "mode": mode,
        "seed": seed,
        "output_path": str(output_path),
        "settings": settings,
        "height": height,
        "width": width,
        "noise_scale": noise_scale if mode == "latent_noise" else None,
        "dropout_rate": dropout_rate if mode == "latent_dropout" else None,
    }


def make_comparison_grid(image_paths: list[Path], labels: list[str], output_path: Path) -> None:
    """Create a labeled side-by-side comparison grid."""
    images = [Image.open(path).convert("RGB") for path in image_paths]
    label_height = 36
    width = sum(image.width for image in images)
    height = max(image.height for image in images) + label_height
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)

    x = 0
    for image, label in zip(images, labels):
        grid.paste(image, (x, label_height))
        draw.text((x + 8, 10), label, fill="black")
        x += image.width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def read_prompts(args: argparse.Namespace) -> list[str]:
    """Read prompts from --prompt and/or --prompts-file."""
    prompts = list(args.prompt or [])
    if args.prompts_file:
        path = Path(args.prompts_file)
        prompts.extend(
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not prompts:
        raise ValueError("Provide at least one --prompt or a --prompts-file.")
    return prompts


def run_experiment(args: argparse.Namespace) -> Path:
    """Run the full degradation experiment and return the output directory."""
    device = args.device or choose_device()
    model_id = args.model_id or os.getenv("SD_MODEL_ID", DEFAULT_MODEL_ID)
    modes = args.modes or list(DEFAULT_MODES)
    prompts = read_prompts(args)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / run_id

    pipe = load_pipeline(model_id, device=device)
    print(f"Loaded {model_id} on {device}")
    metadata: dict[str, Any] = {
        "model_id": model_id,
        "device": device,
        "run_id": run_id,
        "modes": modes,
        "prompts": prompts,
        "results": [],
    }

    for prompt_index, prompt in enumerate(prompts, start=1):
        prompt_slug = slugify(prompt)
        prompt_dir = output_dir / f"{prompt_index:02d}-{prompt_slug}"
        image_paths = []
        labels = []

        for mode_index, mode in enumerate(modes):
            seed = args.seed + prompt_index * 100 + mode_index
            image_path = prompt_dir / f"{mode}.png"
            print(f"Generating prompt {prompt_index}/{len(prompts)} mode={mode} seed={seed}")
            result = generate_variant(
                pipe=pipe,
                prompt=prompt,
                mode=mode,
                seed=seed,
                output_path=image_path,
                height=args.height,
                width=args.width,
                noise_scale=args.noise_scale,
                dropout_rate=args.dropout_rate,
            )
            metadata["results"].append(result)
            image_paths.append(image_path)
            labels.append(mode)

        make_comparison_grid(image_paths, labels, prompt_dir / "comparison_grid.png")

    metadata_path = output_dir / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate baseline and intentionally degraded open-model art variants."
    )
    parser.add_argument("--prompt", action="append", help="Prompt to generate. Can be repeated.")
    parser.add_argument("--prompts-file", help="Text file containing one prompt per line.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated images.")
    parser.add_argument("--model-id", default=None, help="Hugging Face model id.")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--seed", type=int, default=153)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--noise-scale", type=float, default=0.08)
    parser.add_argument("--dropout-rate", type=float, default=0.08)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=None)
    args = parser.parse_args()

    output_dir = run_experiment(args)
    print(f"Saved experiment outputs to: {output_dir}")


if __name__ == "__main__":
    main()
