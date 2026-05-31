# Intentional Imperfection Experiment

This experiment tests whether deliberately degraded open-source diffusion generation can make AI art feel more surprising, strange, or artistically interesting.

## Research Question

Can controlled imperfection in an open image model produce artwork that people prefer over a normal polished baseline?

## Conditions

The script generates the same prompt under several conditions:

- `baseline`: normal generation
- `low_steps`: fewer denoising steps, often rougher and less resolved
- `low_guidance`: weaker prompt guidance, allowing more drift
- `high_guidance`: over-constrained prompt guidance, often harsher or stranger
- `latent_noise`: injects noise into the latent state during denoising
- `latent_dropout`: randomly masks parts of the latent state during denoising

## Setup

Create a `.env` file:

```env
HF_TOKEN=your_huggingface_token_here
```

Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Small smoke test:

```bash
.venv/bin/python degraded_diffusion.py \
  --prompt "a lonely birthday party in an empty mall" \
  --modes baseline low_steps \
  --height 256 \
  --width 256 \
  --output-dir outputs/smoke
```

Full prompt set:

```bash
.venv/bin/python degraded_diffusion.py \
  --prompts-file prompts.txt \
  --output-dir outputs/experiment
```

The script writes each image variant, a labeled `comparison_grid.png`, and a `metadata.json` file.

## GPU Recommendation

Local CPU/MPS generation may be too slow. A GPU machine through DigitalOcean/Paperspace is recommended for the full experiment.

Good first GPU command:

```bash
.venv/bin/python degraded_diffusion.py \
  --prompts-file prompts.txt \
  --modes baseline low_steps low_guidance high_guidance latent_noise latent_dropout \
  --output-dir outputs/experiment
```

## Evaluation

For each prompt, show the comparison grid to reviewers without explaining which condition is which. Ask them to rate or choose:

- Which image is most artistically interesting?
- Which image feels least generic?
- Which image is most surprising or memorable?
- Which image has the strongest emotional effect?
- Which image would you most want to keep looking at?

This gives evidence for whether intentional degradation improves perceived artistic quality.
