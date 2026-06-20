# Real-Time Speech Translator

Windows에서 재생 중인 시스템 오디오 또는 마이크 입력을 실시간으로 캡처해 자막처럼 번역하는 Python 파이프라인입니다.

현재 지원 방향:

```text
Japanese -> Korean
Korean   -> Japanese
English  -> Korean
Korean   -> English
```

핵심 흐름은 다음과 같습니다.

```text
audio capture -> faster-whisper ASR -> text cleanup/buffering -> NLLB translation -> optional Korean LLM refinement -> terminal output
```

## Features

- Windows WASAPI loopback으로 Zoom, Teams, YouTube, 브라우저, 시스템 사운드 캡처
- 마이크 입력 지원
- faster-whisper 기반 일본어/한국어/영어 음성 인식
- NLLB `facebook/nllb-200-distilled-600M` 기반 다국어 번역
- 짧은 ASR 조각을 문장 단위로 묶는 sentence buffer
- ASR 품질 필터, 중복 억제, 용어 보정
- 한국어 출력 방향(`ja2ko`, `en2ko`)에서 선택적 Ollama LLM 자연화
- 실행 목적이 바로 보이는 PowerShell 실행 파일 제공

## Requirements

권장 환경:

```text
OS      : Windows 10 / Windows 11
Shell   : PowerShell
Python  : 3.10 또는 3.11
GPU     : NVIDIA GPU 권장
CUDA    : PyTorch CUDA 빌드와 드라이버 호환 필요
Audio   : WASAPI loopback 지원 장치
```

CPU 실행도 가능하지만 ASR/번역 속도가 크게 느려질 수 있습니다.

## Project Structure

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
│  ├─ translation_guard.py
│  └─ llm_refiner.py
├─ models/
├─ requirements.txt
├─ run_ja_to_ko.ps1
├─ run_ko_to_ja.ps1
├─ run_en_to_ko.ps1
├─ run_ko_to_en.ps1
└─ README.md
```

## Setup

프로젝트 루트로 이동합니다.

```powershell
cd D:\real_time_translator
```

가상환경을 만들고 활성화합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

PowerShell 실행 정책 때문에 활성화가 막히면 한 번만 실행합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

기본 패키지를 설치합니다.

```powershell
pip install -r requirements.txt
```

CUDA GPU를 사용할 경우 PyTorch CUDA 빌드를 설치합니다. CUDA 12.1 예시는 다음과 같습니다.

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU만 사용할 경우:

```powershell
pip install torch torchvision torchaudio
```

CUDA 인식 확인:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`True`가 나오면 CUDA 사용이 가능합니다.

## Models

모델은 최초 실행 시 자동으로 다운로드됩니다.

기본 모델:

- ASR: faster-whisper `medium`
- Translation: `facebook/nllb-200-distilled-600M`

모델 캐시는 기본적으로 프로젝트의 `models/` 폴더를 사용합니다.

네트워크가 느리면 최초 실행 때 시간이 걸릴 수 있습니다. Hugging Face rate limit이 거슬리면 `HF_TOKEN` 설정을 고려하세요.

## Audio Devices

사용 가능한 오디오 장치를 확인합니다.

```powershell
python .\scripts\realtime_audio.py --list-devices
```

시스템 사운드, 브라우저, Zoom/Teams 음성을 번역하려면 `loopback=True` 장치를 사용합니다. 기본 장치 자동 선택이 안 되면 `--audio-device-index`로 직접 지정할 수 있습니다.

## Run

실행 전 PowerShell에서 가상환경이 활성화되어 있어야 합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

### Japanese -> Korean

시스템 오디오 loopback 기준입니다.

```powershell
.\run_ja_to_ko.ps1
```

이 실행 파일은 Ollama API가 꺼져 있으면 `ollama serve`를 백그라운드로 자동 시작한 뒤 실행합니다.

내부적으로 다음 방향을 사용합니다.

```text
--direction ja2ko
```

### Korean -> Japanese

마이크 입력 기준입니다.

```powershell
.\run_ko_to_ja.ps1
```

내부적으로 다음 방향을 사용합니다.

```text
--direction ko2ja
```

### English -> Korean

시스템 오디오 loopback 기준입니다.

```powershell
.\run_en_to_ko.ps1
```

이 실행 파일은 Ollama API가 꺼져 있으면 `ollama serve`를 백그라운드로 자동 시작한 뒤 실행합니다.

내부적으로 다음 방향을 사용합니다.

```text
--direction en2ko
```

### Korean -> English

마이크 입력 기준입니다.

```powershell
.\run_ko_to_en.ps1
```

내부적으로 다음 방향을 사용합니다.

```text
--direction ko2en
```

## Manual Run

실행 스크립트 대신 직접 실행할 수도 있습니다.

```powershell
python .\scripts\realtime_audio.py `
  --audio-source loopback `
  --direction ja2ko `
  --chunk-seconds 3.0 `
  --overlap-seconds 0.8 `
  --asr-model medium `
  --asr-device cuda `
  --asr-compute-type int8_float16 `
  --nllb-device cuda `
  --nllb-dtype fp16 `
  --max-wait-sec 1.5 `
  --max-buffer-chars 90 `
  --num-beams 1
```

지원되는 `--direction` 값:

```text
ja2ko
ko2ja
en2ko
ko2en
```

지원되는 `--audio-source` 값:

```text
loopback
mic
```

## Configuration

기본 설정은 `config/translator.yaml`에 있습니다.

중요 항목:

```yaml
audio:
  source: loopback
  chunk_seconds: 2.8
  overlap_seconds: 0.5
  rms_threshold: 220.0

language:
  direction: ja2ko

asr:
  model: medium
  device: cuda
  compute_type: int8_float16

nllb:
  model_name: facebook/nllb-200-distilled-600M
  device: cuda
  dtype: fp16
  num_beams: 1

sentence_buffer:
  enabled: true
  max_chars: 90
  max_wait_sec: 1.5
  min_chars_to_translate: 4
```

실행 파일의 CLI 옵션이 `translator.yaml` 값을 덮어씁니다.

## Output

실행 중 출력은 다음 형태입니다.

```text
[ASR][0.43s][rms=1731] JA: 備えをしておいていただきたいと思います。
========================================================================
[reason=sentence_end | total=4.10s | asr=0.42s | nllb=0.11s | llm=0.00s/skip_short | rms=2497]
JA: 備えをしておいていただきたいと思います。
KO: 준비해 두시기 바랍니다.
========================================================================
```

항목 의미:

| 항목 | 의미 |
|---|---|
| ASR | 음성 인식 결과 |
| rms | 입력 오디오 크기 |
| reason | 번역 출력 이유: sentence_end, timeout, max_chars 등 |
| total | 오디오 chunk 시작부터 출력까지의 전체 시간 |
| asr | 음성 인식 소요 시간 |
| nllb | 번역 모델 소요 시간 |
| llm | LLM refiner 소요 시간과 사용 여부 |

## Tuning

빠른 반응이 중요하면:

```yaml
audio:
  chunk_seconds: 2.0
  overlap_seconds: 0.3

sentence_buffer:
  max_wait_sec: 1.0
```

문장이 너무 잘리면:

```yaml
audio:
  chunk_seconds: 3.0
  overlap_seconds: 0.8

sentence_buffer:
  max_wait_sec: 2.0
```

잡음이 번역되면:

```yaml
audio:
  rms_threshold: 400.0
```

정확도를 조금 더 우선하면:

```yaml
nllb:
  num_beams: 2
```

단, `num_beams`를 올리면 번역 지연이 증가합니다.

## Glossary

자주 틀리는 ASR 단어는 `config/translator.yaml`의 glossary에 추가합니다.

```yaml
glossary:
  ja_asr_corrections:
    "すいません": "すみません"

  ko_asr_corrections:
    "라이다": "LiDAR"
```

권장 원칙:

- 테스트 영상 하나에만 맞는 문장 보정은 넣지 않습니다.
- 평상시 대화/화상회의에 반복적으로 나오는 단어만 넣습니다.
- 고유명사, 제품명, 기술 용어처럼 명확한 항목만 추가합니다.

## Troubleshooting

### PowerShell에서 글자가 깨질 때

실행 스크립트에는 UTF-8 설정이 포함되어 있습니다. 직접 실행할 때는 아래를 먼저 실행하세요.

```powershell
chcp 65001
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

### Loopback 장치가 안 잡힐 때

```powershell
python .\scripts\realtime_audio.py --list-devices
```

loopback 장치 index를 확인한 뒤 직접 지정합니다.

```powershell
python .\scripts\realtime_audio.py --audio-source loopback --audio-device-index 20 --direction ja2ko
```

### CUDA를 사용할 수 없을 때

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

`False`면 CPU 옵션으로 실행합니다.

```powershell
python .\scripts\realtime_audio.py --asr-device cpu --asr-compute-type int8 --nllb-device cpu --nllb-dtype fp32
```

### Ollama LLM 보정이 느리거나 timeout일 때

LLM refiner는 선택 기능입니다. 끄고 실행할 수 있습니다.

```powershell
python .\scripts\realtime_audio.py --direction ja2ko --no-llm-refiner
```

`ko2ja`, `ko2en` 실행 파일은 기본적으로 LLM refiner를 끕니다.

`run_ja_to_ko.ps1`, `run_en_to_ko.ps1`은 Ollama가 설치되어 있고 PATH에 잡혀 있으면 `ollama serve`를 자동으로 시작합니다. 이미 Ollama가 실행 중이면 추가로 시작하지 않습니다.

## Quick Start

가장 짧은 실행 순서:

```powershell
cd D:\real_time_translator
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python .\scripts\realtime_audio.py --list-devices
.\run_ja_to_ko.ps1
```

영어 영상을 한국어로 보려면:

```powershell
.\run_en_to_ko.ps1
```
