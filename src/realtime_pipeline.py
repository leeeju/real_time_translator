from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .audio_capture import AudioCapture, AudioChunk
from .asr_quality_filter import ASRQualityFilter
from .glossary import Glossary
from .llm_refiner import LLMRefiner
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

    Conversation-mode design:
      - Do not overfit to one radio/news sample.
      - Keep latency acceptable for real conversation.
      - Reduce obvious ASR artifacts before translation.
      - Preserve numbers and units where possible.
      - Use NLLB for draft translation and optional Ollama LLM for Korean-only
        refinement. The LLM is not used as a direct Japanese -> Korean
        translator because qwen2.5:7b-instruct can mix Chinese/English on
        long Japanese input.
      - Later GUI can reuse this pipeline by replacing terminal printing
        with callbacks.
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

        llm_cfg = self.raw.get("llm_refiner", {})
        self.llm_refiner = LLMRefiner.from_config(llm_cfg)

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
        print(
            "main mode       : Japanese -> Korean"
            if self.direction == "ja2ko"
            else "mode            : Korean -> Japanese"
        )
        print(f"sentence buffer : {self.buffer_enabled}")
        print(f"llm refiner     : {self.llm_refiner.enabled} ({self.llm_refiner.model})")
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
        source_text = self._cleanup_source_before_translation(source_text)

        if not source_text:
            return

        if self._should_skip_source_text(source_text):
            print(f"[DROP][source:fragment_or_noise] {source_text}")
            return

        if self.suppress_duplicates and is_near_duplicate(self.last_translated_source, source_text):
            return

        guarded = self.translation_guard.try_translate(
            source_text=source_text,
            direction=self.direction,
            translate_fn=self._translate_plain_text,
        )

        llm_elapsed = 0.0
        llm_reason = "disabled"

        if guarded.handled:
            translated_text = self._postprocess_translation(source_text, guarded.text)
            trans_elapsed = 0.0
            reason = f"{reason}+{guarded.reason}"

            # TranslationGuard가 처리한 특수 패턴은 기본적으로 LLM 보정을 건너뜁니다.
            # config에서 refine_after_guard=true로 켜면 Guard 결과도 자연화합니다.
            if self.llm_refiner.enabled and self.llm_refiner.refine_after_guard:
                llm_result = self.llm_refiner.refine(
                    draft_text=translated_text,
                    direction=self.direction,
                )
                llm_elapsed = llm_result.elapsed
                llm_reason = llm_result.reason
                if llm_result.used:
                    translated_text = llm_result.text
            elif self.llm_refiner.enabled:
                llm_reason = "skip_guard"

        else:
            tr_result = self.translator.translate(source_text, self.direction)
            translated_text = self._postprocess_translation(source_text, tr_result.text)
            trans_elapsed = tr_result.elapsed

            # qwen2.5:7b-instruct는 일본어 직접 번역이 아니라
            # NLLB 초벌 한국어 번역을 자연스럽게 다듬는 refiner로만 사용합니다.
            if self.llm_refiner.enabled:
                llm_result = self.llm_refiner.refine(
                    draft_text=translated_text,
                    direction=self.direction,
                )
                llm_elapsed = llm_result.elapsed
                llm_reason = llm_result.reason
                if llm_result.used:
                    translated_text = llm_result.text

        translated_text = self._postprocess_translation(source_text, translated_text)

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
                    f"asr={asr_str} | nllb={trans_elapsed:.2f}s | "
                    f"llm={llm_elapsed:.2f}s/{llm_reason} | rms={rms_str}]"
                )
            else:
                print(
                    f"[total={total_str} | asr={asr_str} | "
                    f"nllb={trans_elapsed:.2f}s | llm={llm_elapsed:.2f}s | rms={rms_str}]"
                )

        print(f"{self.src_label}: {source_text}")
        print(f"{self.tgt_label}: {translated_text}")
        print("=" * 72)
        print()

    def _translate_plain_text(self, text: str, direction: str) -> str:
        return self.translator.translate(text, direction).text

    def _postprocess_translation(self, source_text: str, translated_text: str) -> str:
        translated_text = self.glossary.apply_translation_postprocess(translated_text, self.direction)
        translated_text = self._preserve_units(source_text, translated_text)
        translated_text = self._cleanup_translated_text(translated_text)
        return translated_text.strip()

    def _cleanup_source_before_translation(self, source_text: str) -> str:
        text = normalize_asr_text(source_text)

        if self.direction == "ja2ko":
            text = self._remove_prompt_leak_phrases(text)
            text = self._remove_noise_tokens(text)

        text = normalize_asr_text(text)
        return text.strip()

    def _should_skip_source_text(self, source_text: str) -> bool:
        if not source_text:
            return True

        if self.direction != "ja2ko":
            return False

        t = source_text.strip().strip("。.!?！？、, 　")

        # Remaining noise-only fragment.
        if self._is_noise_only(t):
            return True

        # Short leftover fragments from previous utterance.
        # Conservative list to avoid dropping valid daily expressions.
        if len(t) <= 28 and re.match(r"^(?:の|を|から|まで|より)", t):
            return True

        # Prompt leak guard at final source stage.
        if self._looks_like_prompt_leak(t):
            return True

        return False

    def _remove_noise_tokens(self, text: str) -> str:
        noise = r"(?:コンッ?|カンッ?|ドンッ?|パンッ?|ガサッ?|ザッ|ノイズ|雑音)"

        # Leading impulse noise.
        text = re.sub(rf"^(?:{noise})[、。,\s　]*", "", text)

        # Noise after sentence boundary or whitespace.
        text = re.sub(rf"([。！？!?、,\s　])(?:{noise})(?=[、。,\s　]|$)", r"\1", text)

        # Repeated punctuation/spacing cleanup.
        text = re.sub(r"\s+", " ", text)
        text = text.replace("。。", "。").replace("、、", "、")

        return text.strip()

    def _remove_prompt_leak_phrases(self, text: str) -> str:
        leak_phrases = (
            "日本語の日常会話です。",
            "日本語の日常会話です",
            "質問、説明、確認、依頼、相づち、短い返事、会議での自然な発言を含みます。",
            "質問、説明、確認、依頼、相づち、短い返事、会議での自然な発言を含みます",
            "質問、説明、確認、依頼、相づち、短い返事、会議での自然な発話を含みます。",
            "質問、説明、確認、依頼、相づち、短い返事、会議での自然な発話を含みます",
        )

        for phrase in leak_phrases:
            text = text.replace(phrase, "")

        return text.strip()

    def _preserve_units(self, source_text: str, translated_text: str) -> str:
        """
        Preserve common numbers/units in ja2ko mode.

        This is not news-specific. It protects real conversation too:
          - 30円 should not become 30달러.
          - 10% should stay 10%.
          - 1kg should stay 1kg or 1kg당.
          - 1割 means about 10%, not 1%.
        """
        if self.direction != "ja2ko":
            return translated_text

        src = source_text
        out = translated_text

        # Japanese yen protection.
        # If source mentions 円 and not dollar, Korean output should not say dollars.
        if re.search(r"\d+(?:\.\d+)?\s*円", src) and not re.search(r"\d+(?:\.\d+)?\s*(?:ドル|＄|\$)", src):
            out = re.sub(r"(\d+(?:\.\d+)?)\s*(?:달러|불)", r"\1엔", out)
            out = out.replace("달러", "엔")

        # Percent preservation.
        if "%" in src or "％" in src:
            out = out.replace("퍼센트", "%")
            out = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1%", out)

        # Japanese wari ratio:
        # 1割 -> 10%, 2割 -> 20%, ...
        for m in re.finditer(r"(\d+)\s*割", src):
            try:
                wari = int(m.group(1))
            except ValueError:
                continue

            percent = wari * 10

            # Correct common mistranslation like 1% when source says 1割.
            out = re.sub(rf"\b{wari}\s*%", f"{percent}%", out)
            out = re.sub(rf"{wari}\s*퍼센트", f"{percent}%", out)

            if "할" in out:
                # Avoid aggressive rewrite when Korean already naturally uses 할/비율.
                continue

        # kg preservation.
        if re.search(r"\d+(?:\.\d+)?\s*(?:kg|KG|Kg|キロ|キログラム)", src):
            out = re.sub(r"(\d+(?:\.\d+)?)\s*킬로그램", r"\1kg", out)
            out = re.sub(r"(\d+(?:\.\d+)?)\s*키로그램", r"\1kg", out)
            out = re.sub(r"(\d+(?:\.\d+)?)\s*키로", r"\1kg", out)

        # Celsius preservation.
        if re.search(r"\d+(?:\.\d+)?\s*(?:℃|度|度C)", src):
            out = re.sub(r"(\d+(?:\.\d+)?)\s*도", r"\1도", out)

        return out

    @staticmethod
    def _cleanup_translated_text(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        # Clean common spacing artifacts.
        text = text.replace(" %", "%")
        text = text.replace(" kg", "kg")

        return text.strip()

    @staticmethod
    def _is_noise_only(text: str) -> bool:
        t = text.strip().strip("。.!?！？、, 　")
        if not t:
            return True

        noise_tokens = {
            "コン",
            "コンッ",
            "カン",
            "カンッ",
            "ドン",
            "ドンッ",
            "パン",
            "パンッ",
            "ガサ",
            "ガサッ",
            "ザッ",
            "ノイズ",
            "雑音",
        }

        if t in noise_tokens:
            return True

        if len(t) <= 4 and re.fullmatch(r"[\u30a0-\u30ffーッ]+", t):
            safe_fillers = {"エット", "アノ"}
            if t not in safe_fillers:
                return True

        return False

    @staticmethod
    def _looks_like_prompt_leak(text: str) -> bool:
        prompt_like_patterns = (
            "日本語の日常会話です",
            "質問、説明、確認、依頼",
            "相づち、短い返事",
            "会議での自然な発言",
            "会議での自然な発話",
        )

        return any(p in text for p in prompt_like_patterns)
