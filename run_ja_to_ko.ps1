chcp 65001 | Out-Null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Test-OllamaReady {
  try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -TimeoutSec 1 | Out-Null
    return $true
  } catch {
    return $false
  }
}

if (-not (Test-OllamaReady)) {
  $ollama = Get-Command ollama -ErrorAction SilentlyContinue
  if ($null -eq $ollama) {
    Write-Warning "Ollama is not installed or not in PATH. Continuing without a running refiner service."
  } else {
    Write-Host "Starting Ollama service..."
    Start-Process -FilePath $ollama.Source -ArgumentList "serve" -WindowStyle Hidden | Out-Null

    for ($i = 0; $i -lt 10; $i++) {
      Start-Sleep -Milliseconds 500
      if (Test-OllamaReady) {
        Write-Host "Ollama service is ready."
        break
      }
    }

    if (-not (Test-OllamaReady)) {
      Write-Warning "Ollama did not become ready in time. Translation will continue, and LLM refinement may fall back."
    }
  }
}

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
