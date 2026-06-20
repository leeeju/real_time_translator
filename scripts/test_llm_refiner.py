from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_refiner import LLMRefiner

cfg = yaml.safe_load(open(PROJECT_ROOT / "config" / "translator.yaml", encoding="utf-8"))
refiner = LLMRefiner.from_config(cfg.get("llm_refiner", {}))

print("enabled:", refiner.enabled)
print("model:", refiner.model)
print("url:", refiner.url)

draft = "발언 철회에 대해 거론하지 않았다는 점을 지적하며, 이러한 방식은 허위뿐 아니라 무의미하다고 강력히 비판하고 있습니다."

result = refiner.refine(draft_text=draft, direction="ja2ko")

print("used:", result.used)
print("reason:", result.reason)
print("elapsed:", f"{result.elapsed:.3f}s")
print("text:", result.text)
