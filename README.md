# Real-Time Japanese/Korean Translator

Windows 기반 실시간 음성 번역 파이프라인입니다. 기본 목표는 **Windows에서 재생되는 일본어 오디오(Zoom, Teams, YouTube, 브라우저, 시스템 사운드 등)를 WASAPI loopback으로 캡처하고, faster-whisper로 음성 인식한 뒤 NLLB 모델로 한국어로 번역**하는 것입니다.

현재 기본 방향은 다음과 같습니다.

```text
Japanese audio → faster-whisper ASR → Japanese text cleanup/buffering → NLLB translation → Korean subtitle text
```

---

## 1. 프로젝트 개요

이 프로젝트는 다음 기능을 중심으로 구성됩니다.

| 구분 | 역할 |
|---|---|
| 오디오 캡처 | Windows 마이크 또는 WASAPI loopback 장치에서 오디오 입력 |
| ASR | faster-whisper 기반 일본어/한국어 음성 인식 |
| 품질 필터 | 무음, 반복 문자, 낮은 신뢰도, 언어 비율이 낮은 문장 제거 |
| 문장 버퍼 | 짧은 조각 번역을 줄이고 문장 단위 번역 품질 향상 |
| 용어 보정 | 사용자 정의 일본어/한국어 ASR 보정 사전 적용 |
| 번역 | `facebook/nllb-200-distilled-600M` 기반 일본어↔한국어 번역 |
| 번역 보정 | 퍼센트/통계/설문 문장 등 일부 반복 구조 보정 |

---

## 2. 권장 실행 환경

### 2.1 Windows 환경

권장 환경은 다음과 같습니다.

```text
OS        : Windows 10 / Windows 11 64-bit
Shell     : PowerShell
Python    : Python 3.10 또는 3.11 권장
GPU       : NVIDIA GPU 권장, CPU 실행 가능
CUDA      : PyTorch CUDA 버전과 NVIDIA Driver 호환 필요
Audio     : Windows WASAPI loopback 사용 가능 장치
```

이 프로젝트는 Windows 시스템 오디오를 직접 캡처해야 하므로 `pyaudiowpatch` 기반 WASAPI loopback을 사용합니다. 따라서 Zoom 회의, 브라우저 영상, 시스템 출력 음성을 번역하려면 Windows 환경이 가장 적합합니다.

---

## 3. 권장 폴더 구조

현재 소스는 다음 구조를 기준으로 동작하도록 구성하는 것이 좋습니다.

```text
real_time_translator/
├─ config/
│  └─ translator.yaml
├─ scripts/
│  └─ realtime_audio.py
├─ src/
│  ├─ __init__.py
│  ├─ audio_capture.py
│  ├─ whisper_asr.py
│  ├─ nllb_translator.py
│  ├─ realtime_pipeline.py
│  ├─ sentence_buffer.py
│  ├─ asr_quality_filter.py
│  ├─ glossary.py
│  └─ translation_guard.py
├─ models/
├─ requirements.txt
└─ README.md
```

중요한 점은 `realtime_audio.py`가 프로젝트 루트를 기준으로 `config/translator.yaml`과 `src/` 모듈을 찾는 구조라는 것입니다.

---

## 4. 가상환경 설정

PowerShell에서 프로젝트 루트로 이동한 뒤 아래 명령을 실행합니다.

```powershell
cd C:\Users\<USER>\Desktop\real_time_translator
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

PowerShell 실행 정책 때문에 가상환경 활성화가 막히면 관리자 권한이 아닌 일반 PowerShell에서 다음을 한 번 실행합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

그 후 다시 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 5. 패키지 설치

### 5.1 기본 패키지

```powershell
pip install pyyaml numpy pyaudiowpatch faster-whisper transformers sentencepiece accelerate safetensors protobuf
```

### 5.2 PyTorch 설치

GPU를 사용할 경우 사용 중인 CUDA/PyTorch 조합에 맞는 명령을 사용해야 합니다. 예시는 다음과 같습니다.

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU만 사용할 경우:

```powershell
pip install torch torchvision torchaudio
```

설치 확인:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`True`가 출력되면 CUDA 사용이 가능합니다.

---

## 6. 모델 다운로드

모델은 최초 실행 시 자동으로 다운로드됩니다. 다만 네트워크 상태가 불안정하거나 실행 중 다운로드 지연을 피하고 싶다면 미리 다운로드하는 것이 좋습니다.

### 6.1 faster-whisper ASR 모델 사전 다운로드

기본 모델은 `small`입니다.

```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='int8_float16')"
```

CPU 환경에서는 다음과 같이 확인합니다.

```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"
```

더 높은 인식 품질이 필요하면 `medium` 또는 `large-v3`를 사용할 수 있습니다. 단, 실시간성은 떨어질 수 있습니다.

```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cuda', compute_type='int8_float16')"
```

### 6.2 NLLB 번역 모델 사전 다운로드

기본 번역 모델은 다음입니다.

```text
facebook/nllb-200-distilled-600M
```

프로젝트의 `models/` 폴더에 캐시하려면 다음 명령을 사용할 수 있습니다.

```powershell
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; m='facebook/nllb-200-distilled-600M'; AutoTokenizer.from_pretrained(m, cache_dir='models'); AutoModelForSeq2SeqLM.from_pretrained(m, cache_dir='models')"
```

GPU 메모리가 충분하다면 fp16으로 실행하는 것이 실시간성에 유리합니다.

---

## 7. `translator.yaml` 필수 확인

현재 설정 파일의 주요 구조는 다음과 같습니다.

```yaml
audio:
  source: loopback
  device_index: null
  chunk_seconds: 2.0
  overlap_seconds: 0.3
  rms_threshold: 300.0

language:
  direction: ja2ko

asr:
  model: small
  device: cuda
  compute_type: int8_float16
  beam_size: 1
  vad_filter: true
  condition_on_previous_text: false
  no_speech_threshold: 0.6
  initial_prompt: null

nllb:
  model_name: facebook/nllb-200-distilled-600M
  cache_dir: models
  device: cuda
  dtype: fp16
  max_new_tokens: 64
  num_beams: 1
  repetition_penalty: 1.18
  no_repeat_ngram_size: 3
  length_penalty: 0.95
```

### 중요 수정 사항

현재 업로드된 `translator.yaml`에는 다음 값이 포함되어 있었습니다.

```yaml
num_beams: 1clea
```

이 값은 정수가 아니므로 실행 중 다음과 같은 오류를 유발할 수 있습니다.

```text
ValueError: invalid literal for int() with base 10: '1clea'
```

따라서 반드시 아래처럼 수정해야 합니다.

```yaml
num_beams: 1
```

품질을 조금 더 높이고 지연시간 증가를 허용할 수 있다면 다음도 가능합니다.

```yaml
num_beams: 2
```

---

## 8. 오디오 장치 확인

먼저 Windows에서 인식되는 오디오 장치를 확인합니다.

```powershell
python .\scripts\realtime_audio.py --list-devices
```

출력 예시는 다음과 비슷합니다.

```text
=== Audio Devices ===
[0] in=2, out=0, loopback=False, rate=48000, name=Microphone (...)
[5] in=2, out=0, loopback=True,  rate=48000, name=Speakers (...) [Loopback]
```

시스템 사운드, Zoom, 브라우저 영상을 번역하려면 `loopback=True` 장치를 선택해야 합니다. 기본 출력 장치의 loopback이 자동으로 잡히지 않으면 `--audio-device-index`로 직접 지정합니다.

---

## 9. 기본 실행

### 9.1 설정 파일 기준 실행

```powershell
python .\scripts\realtime_audio.py
```

### 9.2 PowerShell 단일 명령 실행

아래 명령은 Windows loopback 오디오를 사용하여 일본어 음성을 한국어로 번역합니다.

```powershell
python .\scripts\realtime_audio.py --audio-source loopback --direction ja2ko --chunk-seconds 2.0 --overlap-seconds 0.3 --asr-model small --asr-device cuda --asr-compute-type int8_float16 --nllb-device cuda --nllb-dtype fp16 --max-wait-sec 3.0 --num-beams 1 --initial-prompt ""
```

PowerShell에서 명령을 여러 줄로 나눌 경우 줄 끝에 백틱(`` ` ``)을 사용해야 합니다.

```powershell
python .\scripts\realtime_audio.py `
  --audio-source loopback `
  --direction ja2ko `
  --chunk-seconds 2.0 `
  --overlap-seconds 0.3 `
  --asr-model small `
  --asr-device cuda `
  --asr-compute-type int8_float16 `
  --nllb-device cuda `
  --nllb-dtype fp16 `
  --max-wait-sec 3.0 `
  --num-beams 1 `
  --initial-prompt ""
```

명령을 줄바꿈한 뒤 백틱 없이 다음 줄에 `--num-beams ...`를 입력하면 PowerShell이 별도 명령으로 해석할 수 있으므로 주의해야 합니다.

---

## 10. 마이크 입력 실행

마이크 음성을 번역하려면 다음처럼 실행합니다.

```powershell
python .\scripts\realtime_audio.py --audio-source mic --direction ja2ko
```

특정 마이크를 지정하려면 먼저 장치를 확인합니다.

```powershell
python .\scripts\realtime_audio.py --list-devices
```

그 후 인덱스를 지정합니다.

```powershell
python .\scripts\realtime_audio.py --audio-source mic --audio-device-index 2 --direction ja2ko
```

---

## 11. 한국어 → 일본어 번역

기본은 일본어 → 한국어입니다. 한국어 음성을 일본어로 번역하려면 다음과 같이 실행합니다.

```powershell
python .\scripts\realtime_audio.py --direction ko2ja --audio-source mic
```

또는 설정 파일에서 다음 항목을 수정합니다.

```yaml
language:
  direction: ko2ja
```

---

## 12. 실시간 번역 품질 구조

이 프로젝트의 번역 품질은 단순히 번역 모델만으로 결정되지 않습니다. 실제 품질은 아래 순서의 전체 파이프라인에 의해 결정됩니다.

```text
오디오 품질
  ↓
ASR 인식 품질
  ↓
ASR 조각 필터링
  ↓
문장 버퍼링
  ↓
일본어/한국어 정규화
  ↓
사용자 용어집 보정
  ↓
NLLB 번역
  ↓
후처리/통계 문장 보정
```

따라서 번역 결과가 나쁘면 먼저 번역 모델보다 **오디오 입력과 ASR 결과**를 확인해야 합니다.

실행 출력에서 다음 항목을 확인합니다.

```text
[ASR][0.45s][rms=812] JA: ...
[reason=sentence_end | total=2.71s | asr=0.45s | trans=0.23s | rms=812]
JA: ...
KO: ...
```

| 출력 항목 | 의미 |
|---|---|
| ASR | 음성 인식 결과 |
| rms | 입력 오디오 크기 |
| reason | 번역이 실행된 이유: 문장 종료, timeout, max_chars 등 |
| total | 오디오 chunk 시작부터 번역 출력까지 전체 시간 |
| asr | 음성 인식 소요 시간 |
| trans | 번역 모델 소요 시간 |

---

## 13. 품질 튜닝 기준

### 13.1 빠른 응답이 우선일 때

```yaml
audio:
  chunk_seconds: 1.5
  overlap_seconds: 0.2

sentence_buffer:
  max_wait_sec: 2.0

asr:
  model: small
  beam_size: 1

nllb:
  num_beams: 1
```

장점:

- 자막 출력이 빠름
- Zoom/회의 중 즉각적인 이해에 유리

단점:

- 문장이 잘리거나 조사/어미 번역이 불안정할 수 있음
- 긴 일본어 문장에서는 의미 누락 가능성 증가

### 13.2 번역 정확도가 우선일 때

```yaml
audio:
  chunk_seconds: 2.5
  overlap_seconds: 0.4

sentence_buffer:
  max_wait_sec: 4.0

asr:
  model: medium
  beam_size: 1

nllb:
  num_beams: 2
```

장점:

- 문장 단위 번역 품질 향상
- 일본어 긴 문장, 통계/뉴스형 문장에 유리

단점:

- 지연시간 증가
- GPU 메모리 사용량 증가

### 13.3 잡음/무음 오인식이 많을 때

```yaml
audio:
  rms_threshold: 500.0

quality_filter:
  max_no_speech_prob: 0.75
  min_avg_logprob: -1.20
```

`rms_threshold`를 높이면 작은 잡음이나 배경음을 무시하기 쉬워집니다. 다만 실제 음성이 작은 경우 번역 자체가 나오지 않을 수 있습니다.

### 13.4 말이 잘리는 경우

```yaml
audio:
  overlap_seconds: 0.4

sentence_buffer:
  max_wait_sec: 4.0
```

`overlap_seconds`는 이전 오디오 tail을 다음 chunk에 붙여 Whisper가 경계 부분을 더 안정적으로 인식하게 합니다.

### 13.5 반복 출력이 많을 때

```yaml
sentence_buffer:
  suppress_duplicates: true
```

동일하거나 거의 동일한 ASR 결과가 반복될 때 중복 번역을 줄입니다.

---

## 14. 용어집 관리

일본어 ASR이 특정 단어를 자주 잘못 인식하면 `translator.yaml`의 `glossary.ja_asr_corrections`에 추가합니다.

예시:

```yaml
glossary:
  enabled: true
  ja_asr_corrections:
    "かぶか": "株価"
    "にっけいへいきん": "日経平均"
    "かわせ": "為替"
```

한국어 → 일본어 방향에서는 `ko_asr_corrections`를 사용합니다.

```yaml
glossary:
  ko_asr_corrections:
    "라이다": "LiDAR"
    "지엔에스에스": "GNSS"
    "아이엠유": "IMU"
```

권장 방식:

- 테스트 문장 전체를 넣지 않습니다.
- 전문 용어, 고유명사, 자주 틀리는 단어만 넣습니다.
- 회의/뉴스/기술 발표 등 도메인별로 용어집을 분리해도 좋습니다.

---

## 15. 번역 품질 평가 방법

실시간 번역 품질은 다음 기준으로 평가하는 것이 좋습니다.

### 15.1 ASR 품질

| 지표 | 확인 방법 |
|---|---|
| 누락률 | 실제 발화 대비 ASR 문장 누락 여부 |
| 오인식률 | 일본어 원문과 ASR 결과 비교 |
| 반복률 | 같은 문장이 여러 번 출력되는지 확인 |
| 무음 오검출 | 조용한 상황에서 ASR이 생성되는지 확인 |

### 15.2 번역 품질

| 지표 | 확인 방법 |
|---|---|
| 의미 보존 | 원문 의미가 한국어에 유지되는지 확인 |
| 숫자/퍼센트 보존 | `26%`, 날짜, 금액, 단위가 유지되는지 확인 |
| 용어 일관성 | 전문 용어가 매번 같은 표현으로 번역되는지 확인 |
| 문장 완성도 | 조사, 어미, 부정 표현이 자연스러운지 확인 |

### 15.3 실시간성

| 지표 | 권장 기준 |
|---|---|
| total | 회의용이면 2~4초 이하 권장 |
| asr | GPU 기준 1초 이하가 이상적 |
| trans | NLLB 600M fp16 기준 수백 ms~1초 내외 목표 |
| timeout 번역 비율 | 너무 높으면 문장 종결 판단이 부족한 상태 |

---

## 16. 권장 테스트 시나리오

### 16.1 시스템 오디오 테스트

1. YouTube 또는 일본어 뉴스 영상을 재생합니다.
2. `--audio-source loopback`으로 실행합니다.
3. `[ASR] JA:` 출력이 실제 일본어 발화와 유사한지 확인합니다.
4. `KO:` 번역에서 숫자, 주어, 부정 표현이 유지되는지 확인합니다.

### 16.2 마이크 테스트

1. 마이크 입력 장치를 확인합니다.
2. 조용한 환경에서 일본어 문장을 짧게 말합니다.
3. `rms`가 너무 낮으면 마이크 입력 볼륨을 높입니다.
4. 잡음이 많으면 `rms_threshold`를 높입니다.

### 16.3 지연시간 테스트

다음 항목을 기록합니다.

```text
total=...
asr=...
trans=...
reason=...
rms=...
```

비교 실험 예시:

| 실험 | chunk_seconds | overlap_seconds | max_wait_sec | ASR model | NLLB beams |
|---|---:|---:|---:|---|---:|
| 빠른 모드 | 1.5 | 0.2 | 2.0 | small | 1 |
| 기본 모드 | 2.0 | 0.3 | 3.0 | small | 1 |
| 품질 모드 | 2.5 | 0.4 | 4.0 | medium | 2 |

---

## 17. 자주 발생하는 오류와 해결

### 17.1 `PyYAML is required`

```text
PyYAML is required. Install it with:
  pip install pyyaml
```

해결:

```powershell
pip install pyyaml
```

### 17.2 `WASAPI is unavailable`

가능 원인:

- Windows가 아닌 환경에서 loopback 실행
- 오디오 장치 드라이버 문제
- 기본 출력 장치가 비활성화됨

해결:

```powershell
python .\scripts\realtime_audio.py --list-devices
```

loopback 장치가 보이는지 확인합니다. 보이지 않으면 Windows 사운드 설정에서 출력 장치를 활성화합니다.

### 17.3 `Could not find loopback device for default output device`

기본 출력 장치와 loopback 장치 매칭이 실패한 경우입니다.

해결:

1. 장치 목록 확인
2. loopback 장치 index 직접 지정

```powershell
python .\scripts\realtime_audio.py --list-devices
python .\scripts\realtime_audio.py --audio-source loopback --audio-device-index 5
```

### 17.4 CUDA를 사용할 수 없을 때

확인:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

`False`면 CPU 모드로 실행합니다.

```powershell
python .\scripts\realtime_audio.py --asr-device cpu --asr-compute-type int8 --nllb-device cpu --nllb-dtype fp32
```

단, CPU 모드는 실시간성이 크게 떨어질 수 있습니다.

### 17.5 `invalid literal for int() with base 10: '1clea'`

`translator.yaml`에서 아래 값을 수정합니다.

```yaml
num_beams: 1
```

### 17.6 번역이 너무 늦게 나올 때

다음 값을 줄입니다.

```yaml
audio:
  chunk_seconds: 1.5

sentence_buffer:
  max_wait_sec: 2.0

nllb:
  num_beams: 1
```

### 17.7 문장이 너무 잘릴 때

다음 값을 늘립니다.

```yaml
audio:
  chunk_seconds: 2.5
  overlap_seconds: 0.4

sentence_buffer:
  max_wait_sec: 4.0
```

### 17.8 잡음이 번역될 때

```yaml
audio:
  rms_threshold: 500.0
```

또는 quality filter를 더 엄격하게 조정합니다.

```yaml
quality_filter:
  min_avg_logprob: -1.20
  max_no_speech_prob: 0.75
```

---

## 18. 실행 전 체크리스트

```text
[ ] Windows PowerShell에서 가상환경이 활성화되었는가?
[ ] Python 3.10/3.11 환경인가?
[ ] PyTorch CUDA 사용 가능 여부를 확인했는가?
[ ] faster-whisper 모델이 정상 다운로드되었는가?
[ ] NLLB 모델이 정상 다운로드되었는가?
[ ] translator.yaml의 num_beams가 정수인가?
[ ] --list-devices로 loopback 장치를 확인했는가?
[ ] Zoom/브라우저/시스템 출력 장치가 Windows 기본 출력 장치인가?
[ ] 실행 명령을 PowerShell에서 한 줄 또는 백틱으로 입력했는가?
```

---

## 19. 추천 기본 설정

실시간 Zoom/회의 번역 기준 권장값입니다.

```yaml
audio:
  source: loopback
  chunk_seconds: 2.0
  overlap_seconds: 0.3
  rms_threshold: 300.0

asr:
  model: small
  device: cuda
  compute_type: int8_float16
  beam_size: 1
  vad_filter: true
  condition_on_previous_text: false
  no_speech_threshold: 0.6

nllb:
  model_name: facebook/nllb-200-distilled-600M
  device: cuda
  dtype: fp16
  max_new_tokens: 64
  num_beams: 1

sentence_buffer:
  enabled: true
  max_wait_sec: 3.0
  min_chars_to_translate: 6
  suppress_duplicates: true
```

---

## 20. 향후 개선 방향

1. PyQt 또는 웹 기반 자막 GUI 추가
2. 번역 결과를 `.srt`, `.txt`, `.jsonl`로 저장
3. 회의별 사용자 용어집 분리
4. ASR 결과와 번역 결과를 동시에 로깅하여 품질 평가 자동화
5. `medium` ASR 모델과 `small` 모델의 지연시간/정확도 비교 실험
6. OBS 자막 오버레이 연동
7. Zoom 회의 자막창 형태의 Always-on-top overlay 구현

---

## 21. 최소 실행 요약

처음 실행할 때 가장 짧은 절차는 다음과 같습니다.

```powershell
cd C:\Users\<USER>\Desktop\real_time_translator
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install pyyaml numpy pyaudiowpatch faster-whisper transformers sentencepiece accelerate safetensors protobuf
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python .\scripts\realtime_audio.py --list-devices
python .\scripts\realtime_audio.py --audio-source loopback --direction ja2ko --chunk-seconds 2.0 --overlap-seconds 0.3 --asr-model small --asr-device cuda --asr-compute-type int8_float16 --nllb-device cuda --nllb-dtype fp16 --max-wait-sec 3.0 --num-beams 1 --initial-prompt ""
```

CPU 모드 최소 실행:

```powershell
python .\scripts\realtime_audio.py --audio-source loopback --direction ja2ko --asr-device cpu --asr-compute-type int8 --nllb-device cpu --nllb-dtype fp32 --num-beams 1
```

---

## 22. 결론

이 프로젝트의 핵심은 단순히 일본어를 한국어로 번역하는 것이 아니라, **실시간 환경에서 깨끗한 오디오 입력을 확보하고, ASR 조각을 적절히 문장화한 뒤, 용어집과 번역 보정 규칙을 통해 일관된 자막 품질을 얻는 것**입니다.

가장 먼저 확인해야 할 것은 다음 세 가지입니다.

1. loopback 장치가 정상적으로 잡히는가?
2. `[ASR] JA:` 결과가 실제 음성과 유사한가?
3. `total`, `asr`, `trans` 시간이 목적에 맞는가?

이 세 항목이 안정화되면 `glossary`, `sentence_buffer`, `quality_filter`, `num_beams`를 조정하면서 실시간성과 번역 품질의 균형을 맞추면 됩니다.
