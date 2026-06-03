from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install it with:\n"
        "  pip install pyyaml\n"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_capture import list_audio_devices
from src.realtime_pipeline import PipelineConfig, RealtimePipeline


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value

    return out


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data


def build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}

    def set_nested(section: str, key: str, value):
        if value is None:
            return
        updates.setdefault(section, {})[key] = value

    set_nested("audio", "source", args.audio_source)
    set_nested("audio", "device_index", args.audio_device_index)
    set_nested("audio", "chunk_seconds", args.chunk_seconds)
    set_nested("audio", "overlap_seconds", args.overlap_seconds)
    set_nested("audio", "rms_threshold", args.rms_threshold)

    set_nested("language", "direction", args.direction)

    set_nested("asr", "model", args.asr_model)
    set_nested("asr", "device", args.asr_device)
    set_nested("asr", "compute_type", args.asr_compute_type)
    set_nested("asr", "initial_prompt", args.initial_prompt)

    set_nested("nllb", "device", args.nllb_device)
    set_nested("nllb", "dtype", args.nllb_dtype)
    set_nested("nllb", "max_new_tokens", args.max_new_tokens)
    set_nested("nllb", "num_beams", args.num_beams)

    set_nested("sentence_buffer", "max_wait_sec", args.max_wait_sec)
    set_nested("sentence_buffer", "max_chars", args.max_buffer_chars)
    set_nested("sentence_buffer", "min_chars_to_translate", args.min_chars_to_translate)

    if args.no_sentence_buffer:
        set_nested("sentence_buffer", "enabled", False)

    if args.no_partial_asr:
        set_nested("sentence_buffer", "print_partial_asr", False)

    if args.no_quality_filter:
        set_nested("quality_filter", "enabled", False)

    if args.no_glossary:
        set_nested("glossary", "enabled", False)

    if args.no_translation_guard:
        set_nested("translation_guard", "enabled", False)

    return updates


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Japanese/Korean audio translator. Default: Japanese -> Korean."
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "translator.yaml"),
        help="Path to translator.yaml",
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print audio device list and exit.",
    )

    parser.add_argument("--audio-source", choices=["mic", "loopback"], default=None)
    parser.add_argument("--audio-device-index", type=int, default=None)
    parser.add_argument("--chunk-seconds", type=float, default=None)
    parser.add_argument("--overlap-seconds", type=float, default=None)
    parser.add_argument("--rms-threshold", type=float, default=None)

    parser.add_argument("--direction", choices=["ja2ko", "ko2ja"], default=None)

    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--asr-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--asr-compute-type", default=None)
    parser.add_argument("--initial-prompt", default=None)

    parser.add_argument("--nllb-device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--nllb-dtype", choices=["auto", "fp32", "fp16"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)

    parser.add_argument("--max-wait-sec", type=float, default=None)
    parser.add_argument("--max-buffer-chars", type=int, default=None)
    parser.add_argument("--min-chars-to-translate", type=int, default=None)

    parser.add_argument("--no-sentence-buffer", action="store_true")
    parser.add_argument("--no-partial-asr", action="store_true")
    parser.add_argument("--no-quality-filter", action="store_true")
    parser.add_argument("--no-glossary", action="store_true")
    parser.add_argument("--no-translation-guard", action="store_true")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    cfg = load_config(Path(args.config))
    overrides = build_cli_overrides(args)
    cfg = deep_update(cfg, overrides)

    pipeline = RealtimePipeline(
        PipelineConfig(
            raw=cfg,
            project_root=PROJECT_ROOT,
        )
    )

    pipeline.run()


if __name__ == "__main__":
    main()
