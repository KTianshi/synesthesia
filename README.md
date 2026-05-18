# Synesthesia

An AI system that translates music into visuals, inspired by synesthesia, the phenomenon where sensory input in one modality triggers an involuntary experience in another. This project is built around my own personal associations between musical keys and colors.

## Vision

Audio in, generated visual out. The system extracts musical features from an audio file, maps them to visual parameters using a personal key-color schema, constructs a painterly image generation prompt, and produces an image that reflects the emotional and structural character of the music.

## Pipeline

1. Audio feature extraction: `librosa` extracts tempo, RMS energy, dominant key via Krumhansl-Schmuckler key-finding, spectral centroid, and onset strength.
2. Visual parameter mapping: audio features are mapped to color palette, mood, form language, compositional density, and sense of motion using a personal key-color table.
3. Prompt construction: visual parameters are assembled into a painterly image generation prompt.
4. Image generation: the OpenAI image API generates a 1024x1024 image from the prompt.

## Personal Key-Color Schema

The color mappings are drawn from my own synesthetic associations:

```text
C major:   white - pure, clean
Db major:  golden - rich, warm
D major:   golden - sunnier, brighter than Db
Eb major:  golden - heroic, bold
E major:   green
F major:   green
F# major:  green - lighter shade
G major:   blue
Ab major:  pink/pink lemonade - soft, luminous
A major:   pink
Bb major:  dark blue - heavier than G
B major:   indigo - deep, mysterious

C minor:   deep red - intense, dramatic
D minor:   cool grey - somber, austere
Eb minor:  dark blue-grey - stormy, turbulent
E minor:   cool grey - somber, austere
F minor:   dark forest green - shadowy, dense
F# minor:  dark blue-grey - stormy, turbulent
G minor:   dark blue - heavy, brooding
Ab minor:  dark blue-grey - stormy, turbulent
A minor:   cool grey - somber, austere
Bb minor:  very dark purple - ominous, oppressive
B minor:   cool grey - somber, austere
```

## Additional Mapping Rules

- RMS energy maps to saturation and brightness. High energy is vivid and saturated; low energy is muted and diffuse.
- Tempo maps to compositional density. Under 60 BPM is sparse, 60-120 is moderate, and over 120 is dense and busy.
- Spectral centroid maps to form sharpness. Under 2000 Hz produces soft organic forms, while over 4000 Hz produces sharp geometric forms.
- Onset strength maps to sense of motion. High onset strength is dynamic and kinetic, while low onset strength is still and atmospheric.
- Ambiguous key detection blends the top two chroma pitch classes proportionally.

## Stretch Goals

- Real-time p5.js animation layer driven by live microphone input
- Stable Diffusion inference for offline/local image generation
- Multi-segment generation tracking key and mood changes across a full piece

## Usage

Create a `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python3 audio_features.py path/to/audio.wav --output image.png
```

## Stack

- Python
- `librosa`
- OpenAI Python SDK
- OpenAI image generation API
