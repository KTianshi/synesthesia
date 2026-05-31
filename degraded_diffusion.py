"""Intentional degradation experiments for open-source image models."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
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
DEFAULT_MODES = (
    "baseline",
    "low_steps",
    "low_guidance",
    "high_guidance",
    "latent_noise",
    "latent_dropout",
    "weight_noise",
    "weight_dropout",
    "activation_roll",
    "activation_shuffle",
    "activation_noise",
)


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
    """Load a compatible text-to-image diffusion pipeline from Hugging Face."""
    from diffusers import AutoPipelineForText2Image

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch_dtype,
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
        "weight_noise": {"num_inference_steps": 30, "guidance_scale": 7.5},
        "weight_dropout": {"num_inference_steps": 30, "guidance_scale": 7.5},
        "activation_roll": {"num_inference_steps": 30, "guidance_scale": 7.5},
        "activation_shuffle": {"num_inference_steps": 30, "guidance_scale": 7.5},
        "activation_noise": {"num_inference_steps": 30, "guidance_scale": 7.5},
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


def get_denoiser(pipe) -> torch.nn.Module:
    """Return the main denoising network for SD, SDXL, or FLUX-style pipelines."""
    if hasattr(pipe, "transformer"):
        return pipe.transformer
    if hasattr(pipe, "unet"):
        return pipe.unet
    raise ValueError("Pipeline does not expose a transformer or unet denoiser.")


def _is_perturbable_weight(name: str, parameter: torch.nn.Parameter) -> bool:
    if parameter.ndim < 2 or not torch.is_floating_point(parameter):
        return False
    target_terms = ("attn", "to_q", "to_k", "to_v", "to_out", "proj", "ff", "mlp")
    return any(term in name.lower() for term in target_terms)


def _is_block_module(name: str, module: torch.nn.Module) -> bool:
    """Select whole denoiser blocks rather than tiny inner layers."""
    del module
    return bool(
        re.search(r"(?:^|\.)transformer_blocks\.\d+$", name)
        or re.search(r"(?:^|\.)single_transformer_blocks\.\d+$", name)
        or re.search(r"(?:^|\.)down_blocks\.\d+$", name)
        or re.search(r"(?:^|\.)mid_block$", name)
        or re.search(r"(?:^|\.)up_blocks\.\d+$", name)
    )


def _perturb_activation_tensor(
    tensor: torch.Tensor,
    mode: str,
    generator: torch.Generator,
    strength: float,
) -> torch.Tensor:
    """Apply a structural activation perturbation while preserving tensor shape."""
    if not torch.is_floating_point(tensor) or tensor.ndim < 3:
        return tensor

    if mode == "activation_noise":
        scale = tensor.detach().float().std().to(tensor.dtype)
        if not torch.isfinite(scale) or float(scale) == 0.0:
            scale = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)
        noise = torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)
        return tensor + noise * scale * strength

    if tensor.ndim == 3:
        # Transformer token sequence: [batch, tokens, channels].
        token_dim = 1
    else:
        # U-Net feature map: perturb width first, preserving channels.
        token_dim = -1

    if mode == "activation_roll":
        shift = max(1, int(tensor.shape[token_dim] * strength))
        rolled = torch.roll(tensor, shifts=shift, dims=token_dim)
        return tensor.lerp(rolled, min(1.0, strength * 2.0))

    if mode == "activation_shuffle":
        index = torch.randperm(tensor.shape[token_dim], generator=generator, device=tensor.device)
        shuffled = tensor.index_select(token_dim, index)
        return tensor.lerp(shuffled, min(1.0, strength))

    return tensor


def _map_activation_output(
    output,
    mode: str,
    generator: torch.Generator,
    strength: float,
):
    if isinstance(output, torch.Tensor):
        return _perturb_activation_tensor(output, mode, generator, strength)
    if isinstance(output, tuple):
        return tuple(
            _perturb_activation_tensor(item, mode, generator, strength)
            if isinstance(item, torch.Tensor)
            else item
            for item in output
        )
    if isinstance(output, list):
        return [
            _perturb_activation_tensor(item, mode, generator, strength)
            if isinstance(item, torch.Tensor)
            else item
            for item in output
        ]
    return output


@contextmanager
def temporary_activation_perturbation(
    pipe,
    mode: str,
    seed: int,
    strength: float,
    max_blocks: int,
):
    """Temporarily perturb hidden activations in selected denoiser blocks."""
    if mode not in {"activation_roll", "activation_shuffle", "activation_noise"}:
        yield []
        return

    denoiser = get_denoiser(pipe)
    blocks = [
        (name, module)
        for name, module in denoiser.named_modules()
        if _is_block_module(name, module)
    ]
    if not blocks:
        yield []
        return

    # Choose middle blocks because they tend to affect scene structure more than texture-only details.
    start = max(0, (len(blocks) - max_blocks) // 2)
    selected = blocks[start : start + max_blocks]
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)
    handles = []

    def hook(_module, _inputs, output):
        return _map_activation_output(output, mode, generator, strength)

    try:
        for _, module in selected:
            handles.append(module.register_forward_hook(hook))
        yield [name for name, _ in selected]
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def temporary_weight_perturbation(
    pipe,
    mode: str,
    seed: int,
    noise_scale: float,
    dropout_rate: float,
    max_tensors: int,
):
    """Temporarily perturb a small subset of denoiser weights, then restore them."""
    if mode not in {"weight_noise", "weight_dropout"}:
        yield []
        return

    denoiser = get_denoiser(pipe)
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)
    candidates = [
        (name, parameter)
        for name, parameter in denoiser.named_parameters()
        if _is_perturbable_weight(name, parameter)
    ]
    if not candidates:
        yield []
        return

    # Spread perturbations across the denoiser instead of only hitting early layers.
    stride = max(1, len(candidates) // max_tensors)
    selected = candidates[::stride][:max_tensors]
    originals = [(parameter, parameter.detach().clone()) for _, parameter in selected]
    changed_names = [name for name, _ in selected]

    try:
        with torch.no_grad():
            for _, parameter in selected:
                if mode == "weight_noise":
                    scale = parameter.detach().float().std().to(parameter.dtype)
                    if not torch.isfinite(scale) or float(scale) == 0.0:
                        scale = torch.tensor(1.0, device=parameter.device, dtype=parameter.dtype)
                    noise = torch.randn(
                        parameter.shape,
                        generator=generator,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    parameter.add_(noise * scale * noise_scale)
                elif mode == "weight_dropout":
                    mask = torch.rand(
                        parameter.shape,
                        generator=generator,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ) > dropout_rate
                    parameter.mul_(mask)
        yield changed_names
    finally:
        with torch.no_grad():
            for parameter, original in originals:
                parameter.copy_(original)


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
    weight_noise_scale: float = 0.02,
    weight_dropout_rate: float = 0.02,
    weight_max_tensors: int = 8,
    activation_strength: float = 0.35,
    activation_max_blocks: int = 8,
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

    perturbation_context = (
        temporary_weight_perturbation(
            pipe=pipe,
            mode=mode,
            seed=seed,
            noise_scale=weight_noise_scale,
            dropout_rate=weight_dropout_rate,
            max_tensors=weight_max_tensors,
        )
        if mode in {"weight_noise", "weight_dropout"}
        else nullcontext([])
    )
    activation_context = (
        temporary_activation_perturbation(
            pipe=pipe,
            mode=mode,
            seed=seed,
            strength=activation_strength,
            max_blocks=activation_max_blocks,
        )
        if mode in {"activation_roll", "activation_shuffle", "activation_noise"}
        else nullcontext([])
    )

    with perturbation_context as perturbed_weights:
        with activation_context as perturbed_activations:
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
        "weight_noise_scale": weight_noise_scale if mode == "weight_noise" else None,
        "weight_dropout_rate": weight_dropout_rate if mode == "weight_dropout" else None,
        "perturbed_weights": perturbed_weights,
        "activation_strength": activation_strength
        if mode in {"activation_roll", "activation_shuffle", "activation_noise"}
        else None,
        "perturbed_activations": perturbed_activations,
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
                weight_noise_scale=args.weight_noise_scale,
                weight_dropout_rate=args.weight_dropout_rate,
                weight_max_tensors=args.weight_max_tensors,
                activation_strength=args.activation_strength,
                activation_max_blocks=args.activation_max_blocks,
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
    parser.add_argument("--weight-noise-scale", type=float, default=0.02)
    parser.add_argument("--weight-dropout-rate", type=float, default=0.02)
    parser.add_argument("--weight-max-tensors", type=int, default=8)
    parser.add_argument("--activation-strength", type=float, default=0.35)
    parser.add_argument("--activation-max-blocks", type=int, default=8)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=None)
    args = parser.parse_args()

    output_dir = run_experiment(args)
    print(f"Saved experiment outputs to: {output_dir}")


if __name__ == "__main__":
    main()
