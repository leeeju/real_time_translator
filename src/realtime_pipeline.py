from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .audio_capture import AudioCapture, AudioChunk
from .asr_quality_filter import ASRQualityFilter
from .glossary import Glossary
from .nllb_translator import LANG_MAP, NLLBTranslator
from .sentence_buffer import SentenceBuffer, is_near_duplicate, normalize_asr_text
from .translation_guard import TranslationGuard
from .whisper_asr import WhisperASR


@dataclass
class PipelineConfig:
    raw: Dict[str, Any]
    project_root: Path


class RealtimePipeline:
    """
    Realtime terminal pipeline.

    Later GUI can reuse this pipeline by replacing _print_result() with callbacks.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.raw = cfg.raw
        self.project_root = cfg.project_root

        lang_cfg = self.raw.get("language", {})
        self.direction = lang_cfg.get("direction", "ja2ko")

        if self.direction not in LANG_MAP:
            raise ValueError(f"Unsupported direction: {self.direction}")

        self.asr_lang = LANG_MAP[self.direction]["asr_lang"]
        self.src_label = LANG_MAP[self.direction]["src_label"]
        self.tgt_label = LANG_MAP[self.direction]["tgt_label"]

        audio_cfg = self.raw.get("audio", {})
        self.audio_capture = AudioCapture(
            source=audio_cfg.get("source", "loopback"),
            device_index=audio_cfg.get("device_index", None),
            chunk_seconds=float(audio_cfg.get("chunk_seconds", 2.0)),
            overlap_seconds=float(audio_cfg.get("overlap_seconds", 0.3)),
            rms_threshold=float(audio_cfg.get("rms_threshold", 300.0)),
        )

        asr_cfg = self.raw.get("asr", {})
        self.asr = WhisperASR(
            model_name=asr_cfg.get("model", "small"),
            device=asr_cfg.get("device", "cuda"),
            compute_type=asr_cfg.get("compute_type", "int8_float16"),
            beam_size=int(asr_cfg.get("beam_size", 1)),
            vad_filter=bool(asr_cfg.get("vad_filter", True)),
            condition_on_previous_text=bool(asr_cfg.get("condition_on_previous_text", False)),
            no_speech_threshold=float(asr_cfg.get("no_speech_threshold", 0.6)),
            initial_prompt=asr_cfg.get("initial_prompt", None),
        )

        nllb_cfg = self.raw.get("nllb", {})
        cache_dir = self.project_root / nllb_cfg.get("cache_dir", "models")
        self.translator = NLLBTranslator(
            model_name=nllb_cfg.get("model_name", "facebook/nllb-200-distilled-600M"),
            cache_dir=cache_dir,
            device=nllb_cfg.get("device", "cuda"),
            dtype=nllb_cfg.get("dtype", "fp16"),
            max_new_tokens=int(nllb_cfg.get("max_new_tokens", 64)),
            num_beams=int(nllb_cfg.get("num_beams", 2)),
            repetition_penalty=float(nllb_cfg.get("repetition_penalty", 1.18)),
            no_repeat_ngram_size=int(nllb_cfg.get("no_repeat_ngram_size", 3)),
            length_penalty=float(nllb_cfg.get("length_penalty", 0.95)),
        )

        buffer_cfg = self.raw.get("sentence_buffer", {})
        self.buffer_enabled = bool(buffer_cfg.get("enabled", True))
        self.print_partial_asr = bool(buffer_cfg.get("print_partial_asr", True))
        self.suppress_duplicates = bool(buffer_cfg.get("suppress_duplicates", True))
        self.sentence_buffer = SentenceBuffer(
            direction=self.direction,
            max_chars=int(buffer_cfg.get("max_chars", 140)),
            max_wait_sec=float(buffer_cfg.get("max_wait_sec", 4.0)),
            min_chars_to_translate=int(buffer_cfg.get("min_chars_to_translate", 6)),
        )

        quality_cfg = self.raw.get("quality_filter", {})
        self.quality_filter = ASRQualityFilter(
            enabled=bool(quality_cfg.get("enabled", True)),
            min_text_len=int(quality_cfg.get("min_text_len", 2)),
            min_japanese_ratio=float(quality_cfg.get("min_japanese_ratio", 0.30)),
            max_repeated_char_ratio=float(quality_cfg.get("max_repeated_char_ratio", 0.50)),
            max_compression_ratio=float(quality_cfg.get("max_compression_ratio", 3.5)),
            min_avg_logprob=float(quality_cfg.get("min_avg_logprob", -1.50)),
            max_no_speech_prob=float(quality_cfg.get("max_no_speech_prob", 0.85)),
        )

        glossary_cfg = self.raw.get("glossary", {})
        self.glossary = Glossary(
            enabled=bool(glossary_cfg.get("enabled", True)),
            ja_asr_corrections=glossary_cfg.get("ja_asr_corrections", {}),
            ko_asr_corrections=glossary_cfg.get("ko_asr_corrections", {}),
        )

        guard_cfg = self.raw.get("translation_guard", {})
        self.translation_guard = TranslationGuard(
            enabled=bool(guard_cfg.get("enabled", True)),
            quote_continuation_enabled=bool(guard_cfg.get("quote_continuation_enabled", True)),
            statistic_pattern_enabled=bool(guard_cfg.get("statistic_pattern_enabled", True)),
            preserve_percent=bool(guard_cfg.get("preserve_percent", True)),
        )

        output_cfg = self.raw.get("output", {})
        self.show_timing = bool(output_cfg.get("show_timing", True))
        self.show_reason = bool(output_cfg.get("show_reason", True))
        self.show_normalized_text = bool(output_cfg.get("show_normalized_text", True))

        self.audio_queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()

        self.last_asr_text = ""
        self.last_translated_source = ""

    def run(self) -> None:
        print("=== Real-time Translator Pipeline ===")
        print(f"direction       : {self.direction}")
        print("main mode       : Japanese -> Korean" if self.direction == "ja2ko" else "mode            : Korean -> Japanese")
        print(f"sentence buffer : {self.buffer_enabled}")
        print("Stop            : Ctrl+C")
        print()

        t_capture = threading.Thread(
            target=self.audio_capture.run,
            args=(self.audio_queue, self.stop_event),
            daemon=True,
        )

        t_worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )

        t_capture.start()
        t_worker.start()

        try:
            while True:
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n[STOPPING]")
            self.stop_event.set()
            t_capture.join(timeout=2)
            t_worker.join(timeout=2)
            print("[STOPPED]")

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                item: AudioChunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                self._try_timeout_flush()
                continue

            try:
                asr_result = self.asr.transcribe_bytes(
                    raw=item.raw,
                    channels=item.channels,
                    rate=item.rate,
                    language=self.asr_lang,
                )

                original_text = normalize_asr_text(asr_result.text)

                seg_decision = self.quality_filter.accept_segments(asr_result.segments)
                if not seg_decision.accepted:
                    print(f"[DROP][segments:{seg_decision.reason}] {original_text}")
                    continue

                text_decision = self.quality_filter.accept_text(original_text, self.direction)
                if not text_decision.accepted:
                    print(f"[DROP][text:{text_decision.reason}] {original_text}")
                    continue

                if self.suppress_duplicates and is_near_duplicate(self.last_asr_text, original_text):
                    continue

                self.last_asr_text = original_text
                corrected_text = self.glossary.apply_asr_corrections(original_text, self.direction)

                if self.print_partial_asr:
                    print(f"[ASR][{asr_result.elapsed:.2f}s][rms={item.rms:.0f}] {self.src_label}: {original_text}")
                    if self.show_normalized_text and corrected_text != original_text:
                        print(f"[NORM] {self.src_label}: {corrected_text}")

                if self.buffer_enabled:
                    buffer_result = self.sentence_buffer.add(corrected_text)
                    if not buffer_result.ready:
                        continue
                    source_text = self.glossary.apply_asr_corrections(buffer_result.text, self.direction)
                    reason = buffer_result.reason
                else:
                    source_text = corrected_text
                    reason = "chunk"

                self._translate_and_print(
                    source_text=source_text,
                    reason=reason,
                    chunk_start=item.chunk_start,
                    asr_elapsed=asr_result.elapsed,
                    rms=item.rms,
                )

            except Exception as exc:
                print(f"[ERROR] {exc}")

    def _try_timeout_flush(self) -> None:
        if not self.buffer_enabled:
            return

        buffer_result = self.sentence_buffer.force_flush_if_timeout()
        if not buffer_result.ready:
            return

        source_text = self.glossary.apply_asr_corrections(buffer_result.text, self.direction)

        self._translate_and_print(
            source_text=source_text,
            reason=buffer_result.reason,
            chunk_start=None,
            asr_elapsed=None,
            rms=None,
        )

    def _translate_and_print(
        self,
        source_text: str,
        reason: str,
        chunk_start: Optional[float],
        asr_elapsed: Optional[float],
        rms: Optional[float],
    ) -> None:
        if self.suppress_duplicates and is_near_duplicate(self.last_translated_source, source_text):
            return

        guarded = self.translation_guard.try_translate(
            source_text=source_text,
            direction=self.direction,
            translate_fn=self._translate_plain_text,
        )

        if guarded.handled:
            translated_text = guarded.text
            trans_elapsed = 0.0
            reason = f"{reason}+{guarded.reason}"
        else:
            tr_result = self.translator.translate(source_text, self.direction)
            translated_text = self.glossary.apply_translation_postprocess(tr_result.text, self.direction)
            trans_elapsed = tr_result.elapsed

        self.last_translated_source = source_text

        total_elapsed = None
        if chunk_start is not None:
            total_elapsed = time.perf_counter() - chunk_start

        print("=" * 72)

        if self.show_timing:
            total_str = "N/A" if total_elapsed is None else f"{total_elapsed:.2f}s"
            asr_str = "N/A" if asr_elapsed is None else f"{asr_elapsed:.2f}s"
            rms_str = "N/A" if rms is None else f"{rms:.0f}"

            if self.show_reason:
                print(
                    f"[reason={reason} | total={total_str} | "
                    f"asr={asr_str} | trans={trans_elapsed:.2f}s | rms={rms_str}]"
                )
            else:
                print(
                    f"[total={total_str} | asr={asr_str} | "
                    f"trans={trans_elapsed:.2f}s | rms={rms_str}]"
                )

        print(f"{self.src_label}: {source_text}")
        print(f"{self.tgt_label}: {translated_text}")
        print("=" * 72)
        print()

    def _translate_plain_text(self, text: str, direction: str) -> str:
        return self.translator.translate(text, direction).text
