from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # requests가 설치되지 않은 경우에도 파이프라인이 죽지 않게 처리
    requests = None


@dataclass
class LLMRefineResult:
    text: str
    elapsed: float
    used: bool
    reason: str


class LLMRefiner:
    """
    Ollama 기반 한국어 번역문 자연화 후처리기.

    중요:
    - 일본어 원문을 직접 번역시키지 않는다.
    - NLLB가 만든 한국어 초벌 번역만 입력한다.
    - 결과에 일본어/중국어/영어 설명이 섞이면 폐기하고 초벌 번역으로 fallback한다.
    """

    _JA_KANA_RE = re.compile(r"[\u3040-\u30ff]")
    _CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    _HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
    _LATIN_RE = re.compile(r"[A-Za-z]")

    _BAD_MARKERS = [
        "Japanese:",
        "Korean:",
        "English:",
        "Explanation:",
        "Analysis:",
        "The translation",
        "Let me",
        "Based on",
        "Hmm,",
        "Wait,",
        "日本語",
        "韓国語",
        "中国語",
        "中文",
        "설명:",
        "분석:",
        "원문:",
        "번역:",
        "不僅",
        "不仅",
        "而且",
        "毫无意义",
        "强烈批评",
        "伪善",
        "偽善",
    ]

    def __init__(
        self,
        enabled: bool = False,
        provider: str = "ollama",
        model: str = "qwen2.5:7b-instruct",
        url: str = "http://127.0.0.1:11434/api/chat",
        temperature: float = 0.0,
        top_p: float = 0.1,
        num_ctx: int = 1024,
        num_predict: int = 80,
        timeout_sec: float = 3.0,
        min_chars: int = 12,
        max_chars: int = 300,
        min_korean_ratio: float = 0.35,
        max_latin_ratio: float = 0.25,
        reject_cjk: bool = True,
        fallback_on_fail: bool = True,
        refine_after_guard: bool = False,
        stop: Optional[List[str]] = None,
    ):
        self.enabled = bool(enabled)
        self.provider = provider
        self.model = model
        self.url = url
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.num_ctx = int(num_ctx)
        self.num_predict = int(num_predict)
        self.timeout_sec = float(timeout_sec)
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self.min_korean_ratio = float(min_korean_ratio)
        self.max_latin_ratio = float(max_latin_ratio)
        self.reject_cjk = bool(reject_cjk)
        self.fallback_on_fail = bool(fallback_on_fail)
        self.refine_after_guard = bool(refine_after_guard)
        self.stop = stop or [
            "Japanese:",
            "Korean:",
            "English:",
            "Explanation:",
            "Analysis:",
            "日本語:",
            "韓国語:",
            "中国語:",
            "中文:",
            "설명:",
            "분석:",
            "不僅",
            "不仅",
            "而且",
            "毫无意义",
            "强烈批评",
        ]

        if self.provider != "ollama":
            self.enabled = False

        if requests is None:
            self.enabled = False

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LLMRefiner":
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            provider=str(cfg.get("provider", "ollama")),
            model=str(cfg.get("model", "qwen2.5:7b-instruct")),
            url=str(cfg.get("url", "http://127.0.0.1:11434/api/chat")),
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 0.1)),
            num_ctx=int(cfg.get("num_ctx", 1024)),
            num_predict=int(cfg.get("num_predict", 80)),
            timeout_sec=float(cfg.get("timeout_sec", 3.0)),
            min_chars=int(cfg.get("min_chars", 12)),
            max_chars=int(cfg.get("max_chars", 300)),
            min_korean_ratio=float(cfg.get("min_korean_ratio", 0.35)),
            max_latin_ratio=float(cfg.get("max_latin_ratio", 0.25)),
            reject_cjk=bool(cfg.get("reject_cjk", True)),
            fallback_on_fail=bool(cfg.get("fallback_on_fail", True)),
            refine_after_guard=bool(cfg.get("refine_after_guard", False)),
            stop=cfg.get("stop", None),
        )

    def refine(
        self,
        draft_text: str,
        direction: str = "ja2ko",
    ) -> LLMRefineResult:
        start = time.perf_counter()
        draft_text = self._normalize_text(draft_text)

        if not self.enabled:
            return LLMRefineResult(draft_text, 0.0, False, "disabled")

        if direction not in ("ja2ko", "en2ko"):
            return LLMRefineResult(draft_text, 0.0, False, "skip_direction")

        if len(draft_text) < self.min_chars and not self._should_refine_short_draft(draft_text):
            return LLMRefineResult(draft_text, 0.0, False, "skip_short")

        if len(draft_text) > self.max_chars:
            return LLMRefineResult(draft_text, 0.0, False, "skip_long")

        if requests is None:
            return LLMRefineResult(draft_text, 0.0, False, "requests_missing")

        try:
            payload = {
                "model": self.model,
                "messages": self._build_messages(draft_text),
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                    "stop": self.stop,
                },
            }

            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()

            data = response.json()
            raw_text = data.get("message", {}).get("content", "")
            refined = self._cleanup_response(raw_text)

            ok, reason = self._validate(refined)

            elapsed = time.perf_counter() - start

            if not ok:
                return LLMRefineResult(draft_text, elapsed, False, f"invalid_{reason}")

            return LLMRefineResult(refined, elapsed, True, "refined")

        except Exception as exc:
            elapsed = time.perf_counter() - start
            return LLMRefineResult(draft_text, elapsed, False, f"error:{type(exc).__name__}")

    def _build_messages(self, draft_text: str) -> List[Dict[str, str]]:
        system_prompt = (
            "너는 한국어 문장 교정기다. "
            "초벌 번역을 자연스러운 한국어로만 다듬어라. "
            "한국어 한 문장만 출력하고 설명하지 마라."
        )

        user_prompt = f"초벌 번역:\n{draft_text}\n\n최종 한국어:"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


    def _cleanup_response(self, text: str) -> str:
        text = self._normalize_text(text)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # 모델이 라벨을 붙였을 경우 제거
        for prefix in ["최종 한국어:", "한국어:", "Korean:", "韓国語:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # 중간에 설명/외국어 라벨이 시작되면 앞부분만 사용
        for marker in self._BAD_MARKERS:
            idx = text.find(marker)
            if idx > 0:
                text = text[:idx].strip()

        text = text.strip(" \t\r\n\"'`")
        return self._normalize_text(text)

    def _validate(self, text: str) -> Tuple[bool, str]:
        if not text:
            return False, "empty"

        if len(text) < 2:
            return False, "too_short"

        for marker in self._BAD_MARKERS:
            if marker in text:
                return False, "bad_marker"

        if self._JA_KANA_RE.search(text):
            return False, "japanese_kana"

        if self.reject_cjk and self._CJK_RE.search(text):
            return False, "cjk"

        compact = re.sub(r"\s+", "", text)
        if not compact:
            return False, "empty_compact"

        hangul_count = len(self._HANGUL_RE.findall(compact))
        korean_ratio = hangul_count / max(1, len(compact))

        if korean_ratio < self.min_korean_ratio:
            return False, "low_korean_ratio"

        latin_count = len(self._LATIN_RE.findall(compact))
        latin_ratio = latin_count / max(1, len(compact))

        # LiDAR, GNSS, IMU, ROS 같은 짧은 기술 용어는 허용하되,
        # 영어 문장으로 흘러가는 경우는 차단
        if latin_ratio > self.max_latin_ratio:
            return False, "high_latin_ratio"

        return True, "ok"

    @staticmethod
    def _should_refine_short_draft(text: str) -> bool:
        awkward_patterns = (
            "것을 바랍니다",
            "있는 것을 바랍니다",
            "하는 것을 바랍니다",
            "할 것을 바랍니다",
            "저는 여러분",
            "저는 당신",
            "나는 여러분",
        )
        return any(pattern in text for pattern in awkward_patterns)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text or ""
        text = text.replace("\u200b", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()
