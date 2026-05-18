"""Audio feature extraction for music-to-visual prompt generation."""

from __future__ import annotations

import json
import os
import base64
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np


PITCH_CLASSES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")

KEY_COLOR_TABLE = {
    "C major": {"color": "white", "description": "pure, clean"},
    "Db major": {"color": "golden", "description": "rich, warm"},
    "D major": {"color": "golden", "description": "sunnier, brighter than Db"},
    "Eb major": {"color": "golden", "description": "heroic, bold"},
    "E major": {"color": "green", "description": ""},
    "F major": {"color": "green", "description": ""},
    "F# major": {"color": "green", "description": "lighter shade than E/F"},
    "G major": {"color": "blue", "description": ""},
    "Ab major": {"color": "pink/pink lemonade", "description": "soft, luminous"},
    "A major": {"color": "pink", "description": "similar to Ab"},
    "Bb major": {"color": "dark blue", "description": "heavier than G major"},
    "B major": {"color": "indigo", "description": "deep, mysterious"},
    "C minor": {"color": "deep red", "description": "intense, dramatic"},
    "D minor": {"color": "cool grey", "description": "somber, austere"},
    "Eb minor": {"color": "dark blue-grey", "description": "stormy, turbulent"},
    "E minor": {"color": "cool grey", "description": "somber, austere"},
    "F minor": {"color": "dark forest green", "description": "shadowy, dense"},
    "F# minor": {"color": "dark blue-grey", "description": "stormy, turbulent"},
    "G minor": {"color": "dark blue", "description": "heavy, brooding"},
    "Ab minor": {"color": "dark blue-grey", "description": "stormy, turbulent"},
    "A minor": {"color": "cool grey", "description": "somber, austere"},
    "Bb minor": {"color": "very dark purple", "description": "ominous, oppressive"},
    "B minor": {"color": "cool grey", "description": "somber, austere"},
}

# Krumhansl-Schmuckler key profiles, ordered from tonic upward by semitone.
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def load_audio(audio_path: str | Path, sr: int | None = 22050) -> tuple[np.ndarray, int]:
    """Load an audio file as mono waveform data."""
    import librosa

    return librosa.load(Path(audio_path), sr=sr, mono=True)


def _safe_mean(values: np.ndarray) -> float:
    """Return a plain Python float for JSON serialization."""
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def summarize_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute a normalized average chroma vector over the full track."""
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    total = float(np.sum(chroma_mean))
    if total <= 0:
        return np.zeros(12, dtype=float)
    return chroma_mean / total


def classify_major_minor(chroma_vector: np.ndarray, tonic_index: int) -> str:
    """Classify the mode for a tonic by comparing major and minor key profiles."""
    major_profile = np.roll(MAJOR_PROFILE, tonic_index)
    minor_profile = np.roll(MINOR_PROFILE, tonic_index)

    major_score = float(np.corrcoef(chroma_vector, major_profile)[0, 1])
    minor_score = float(np.corrcoef(chroma_vector, minor_profile)[0, 1])

    if np.isnan(major_score):
        major_score = -1.0
    if np.isnan(minor_score):
        minor_score = -1.0

    return "major" if major_score >= minor_score else "minor"


def detect_key(chroma_vector: np.ndarray) -> dict[str, Any]:
    """Detect dominant tonic and major/minor mode from an average chroma vector."""
    tonic_index = int(np.argmax(chroma_vector))
    tonic = PITCH_CLASSES[tonic_index]
    mode = classify_major_minor(chroma_vector, tonic_index)

    top_indices = np.argsort(chroma_vector)[-2:][::-1]
    top_pitch_classes = [
        {
            "pitch_class": PITCH_CLASSES[int(index)],
            "weight": float(chroma_vector[int(index)]),
        }
        for index in top_indices
    ]

    return {
        "dominant_key": f"{tonic} {mode}",
        "tonic": tonic,
        "mode": mode,
        "top_pitch_classes": top_pitch_classes,
    }


def extract_audio_features(audio_path: str | Path, sr: int | None = 22050) -> dict[str, Any]:
    """Extract JSON-serializable musical features from an audio file.

    Returns:
        A dict containing tempo, RMS energy, dominant key, mode, spectral
        centroid, onset strength, and chroma details for later color blending.
    """
    import librosa

    y, sample_rate = load_audio(audio_path, sr=sr)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sample_rate)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])

    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sample_rate)
    chroma_vector = summarize_chroma(y, sample_rate)
    key_info = detect_key(chroma_vector)

    return {
        "audio_path": str(Path(audio_path)),
        "sample_rate": int(sample_rate),
        "tempo_bpm": tempo_value,
        "rms_energy": _safe_mean(rms),
        "dominant_key": key_info["dominant_key"],
        "tonic": key_info["tonic"],
        "mode": key_info["mode"],
        "spectral_centroid_hz": _safe_mean(spectral_centroid),
        "onset_strength": _safe_mean(onset_envelope),
        "chroma": {
            "pitch_classes": list(PITCH_CLASSES),
            "mean": [float(value) for value in chroma_vector],
            "top_pitch_classes": key_info["top_pitch_classes"],
        },
    }


def extract_audio_features_json(audio_path: str | Path, sr: int | None = 22050) -> str:
    """Extract audio features and serialize them as pretty-printed JSON."""
    return json.dumps(extract_audio_features(audio_path, sr=sr), indent=2)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _density_from_tempo(tempo_bpm: float) -> str:
    if tempo_bpm < 60:
        return "sparse"
    if tempo_bpm <= 120:
        return "moderate"
    return "dense/busy"


def _form_from_centroid(spectral_centroid_hz: float) -> str:
    if spectral_centroid_hz < 2000:
        return "soft organic forms"
    if spectral_centroid_hz > 4000:
        return "sharp geometric forms"
    return "balanced forms mixing soft curves and crisp edges"


def _movement_from_onset(onset_strength: float) -> str:
    if onset_strength >= 1.5:
        return "dynamic, kinetic"
    if onset_strength <= 0.5:
        return "still, atmospheric"
    return "gently moving"


def _key_descriptor(key_name: str, fallback_key: str | None = None) -> dict[str, str]:
    descriptor = KEY_COLOR_TABLE.get(key_name)
    if descriptor is None and fallback_key is not None:
        descriptor = KEY_COLOR_TABLE.get(fallback_key)
    if descriptor is None:
        return {"color": "unmapped color", "description": "ambiguous"}
    return descriptor


def _blend_pitch_class_colors(features: dict[str, Any]) -> dict[str, Any]:
    mode = str(features.get("mode", "major")).lower()
    dominant_key = str(features.get("dominant_key", ""))
    top_pitch_classes = features.get("chroma", {}).get("top_pitch_classes", [])

    if not top_pitch_classes:
        descriptor = _key_descriptor(dominant_key)
        return {
            "description": descriptor["color"],
            "components": [
                {
                    "key": dominant_key,
                    "color": descriptor["color"],
                    "description": descriptor["description"],
                    "proportion": 1.0,
                }
            ],
        }

    usable_pitch_classes = top_pitch_classes[:2]
    total_weight = sum(float(item.get("weight", 0.0)) for item in usable_pitch_classes)
    if total_weight <= 0:
        total_weight = 1.0

    components = []
    palette_parts = []
    for item in usable_pitch_classes:
        pitch_class = str(item.get("pitch_class", "")).strip()
        key_name = f"{pitch_class} {mode}"
        descriptor = _key_descriptor(key_name, fallback_key=dominant_key)
        proportion = float(item.get("weight", 0.0)) / total_weight
        proportion = _clamp(proportion)

        color_phrase = descriptor["color"]
        if descriptor["description"]:
            color_phrase = f"{color_phrase} ({descriptor['description']})"
        palette_parts.append(f"{round(proportion * 100)}% {color_phrase}")

        components.append(
            {
                "key": key_name,
                "color": descriptor["color"],
                "description": descriptor["description"],
                "chroma_weight": float(item.get("weight", 0.0)),
                "proportion": proportion,
            }
        )

    return {
        "description": " blended with ".join(palette_parts),
        "components": components,
    }


def map_features_to_visual_params(features: dict[str, Any]) -> dict[str, Any]:
    """Map extracted audio features into painterly visual parameters."""
    rms_energy = float(features.get("rms_energy", 0.0))
    energy_normalized = _clamp(rms_energy / 0.1)
    tempo_bpm = float(features.get("tempo_bpm", 0.0))
    spectral_centroid_hz = float(features.get("spectral_centroid_hz", 0.0))
    onset_strength = float(features.get("onset_strength", 0.0))
    mode = str(features.get("mode", "")).lower()

    color_palette = _blend_pitch_class_colors(features)
    mood_adjectives = []
    dominant_descriptor = _key_descriptor(str(features.get("dominant_key", "")))
    if dominant_descriptor["description"]:
        mood_adjectives.append(dominant_descriptor["description"])
    if mode == "minor":
        mood_adjectives.extend(["stormy", "shadowy", "turbulent"])

    return {
        "color_palette": color_palette["description"],
        "color_blend": color_palette["components"],
        "saturation": energy_normalized,
        "brightness": energy_normalized,
        "energy_descriptor": "vivid, saturated" if energy_normalized >= 0.67 else "muted, diffuse"
        if energy_normalized <= 0.33
        else "moderately luminous",
        "compositional_density": _density_from_tempo(tempo_bpm),
        "form_language": _form_from_centroid(spectral_centroid_hz),
        "sense_of_motion": _movement_from_onset(onset_strength),
        "mood_adjectives": ", ".join(dict.fromkeys(mood_adjectives)),
        "source_features": {
            "tempo_bpm": tempo_bpm,
            "rms_energy": rms_energy,
            "spectral_centroid_hz": spectral_centroid_hz,
            "onset_strength": onset_strength,
            "dominant_key": features.get("dominant_key"),
            "mode": features.get("mode"),
        },
    }


def build_dalle_prompt(visual_params: dict[str, Any]) -> str:
    """Build an evocative DALL-E prompt from visual parameters."""
    color_palette = visual_params.get("color_palette", "a resonant color field")
    mood_adjectives = visual_params.get("mood_adjectives") or "expressive, atmospheric"
    form_language = visual_params.get("form_language", "lyrical abstract forms")
    compositional_density = visual_params.get("compositional_density", "balanced composition")
    sense_of_motion = visual_params.get("sense_of_motion", "subtle motion")
    energy_descriptor = visual_params.get("energy_descriptor", "")

    density_phrase = {
        "sparse": "sparse and expansive composition",
        "moderate": "moderately layered composition",
        "dense/busy": "dense, busy composition alive with detail",
    }.get(str(compositional_density), str(compositional_density))

    prompt_parts = [
        f"An abstract painting in {color_palette}",
        str(mood_adjectives),
        f"{energy_descriptor} light and pigment" if energy_descriptor else "",
        str(form_language),
        density_phrase,
        str(sense_of_motion),
    ]
    prompt = ", ".join(part for part in prompt_parts if part).strip()
    return f"{prompt}. Paint it as a vivid, tactile canvas with expressive brushwork."


def generate_image(prompt: str, output_path: str) -> str:
    """Generate a 1024x1024 image, save it, and return the path."""
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    client = OpenAI()
    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    image = response.data[0]
    if image.b64_json:
        destination.write_bytes(base64.b64decode(image.b64_json))
    elif image.url:
        urlretrieve(image.url, destination)
    else:
        raise RuntimeError("Image generation response did not include image data.")

    return str(destination)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Turn an audio file into a synesthesia-inspired DALL-E image."
    )
    parser.add_argument("audio_path", help="Path to the audio file to analyze.")
    parser.add_argument(
        "--output",
        default="generated_image.png",
        help="Path where the generated image should be saved.",
    )
    parser.add_argument("--sr", type=int, default=22050, help="Target sample rate.")
    args = parser.parse_args()

    features = extract_audio_features(args.audio_path, sr=args.sr)
    visual_params = map_features_to_visual_params(features)
    prompt = build_dalle_prompt(visual_params)

    print(f"Detected key: {features['dominant_key']}")
    print(f"Generated prompt: {prompt}")

    saved_path = generate_image(prompt, args.output)
    print(f"Saved image: {saved_path}")


if __name__ == "__main__":
    main()
