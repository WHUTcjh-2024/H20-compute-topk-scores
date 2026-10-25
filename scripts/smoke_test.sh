#!/usr/bin/env bash
# Shortest H20 bring-up: environment check -> import test -> correctness -> benchmark.
# GPU stages are skipped (exit 0) when CUDA is unavailable. Set REQUIRE_GPU=1 to fail instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in \
    "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
    "$ROOT/.venv/bin/python" \
    "$ROOT/venv/bin/python" \
    python3; do
    [ -n "$candidate" ] || continue
    if [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [ -z "$PYTHON" ]; then
  echo "[smoke] no python interpreter found; set PYTHON=/path/to/python" >&2
  exit 1
fi
echo "[smoke] python: $PYTHON"

echo "[smoke] stage 1/4: environment check"
"$PYTHON" scripts/check_env.py --strict

echo "[smoke] stage 2/4: import test"
"$PYTHON" - <<'PY'
import torch
import triton
import vllm

from h20_topk import compute_topk_scores

print(
    f"[smoke] torch={torch.__version__} triton={triton.__version__} "
    f"vllm={vllm.__version__}"
)
print(
    f"[smoke] torch.version.cuda={torch.version.cuda} "
    f"cuda_available={torch.cuda.is_available()}"
)
print(f"[smoke] candidate import OK: {compute_topk_scores.__module__}")
PY

GPU_OK="$("$PYTHON" -c 'import torch; print(int(torch.cuda.is_available()))')"
if [ "$GPU_OK" != "1" ]; then
  echo "[smoke] CUDA unavailable: skipping correctness and benchmark stages."
  if [ "${REQUIRE_GPU:-0}" = "1" ]; then
    echo "[smoke] REQUIRE_GPU=1 set; failing." >&2
    exit 3
  fi
  exit 0
fi

echo "[smoke] stage 3/4: quick correctness"
QUICK_MODE=1 "$PYTHON" -m pytest -q tests/test_compute_topk_scores.py -m compute_topk_scores

echo "[smoke] stage 4/4: benchmark smoke"
OUT="$ROOT/artifacts/smoke"
mkdir -p "$OUT"
COMMON=(--batches 1,8 --vocabs 32768 --ks 0,10 --dtypes bfloat16 --warmup 3 --repeats 10)
"$PYTHON" scripts/bench_matrix.py --target vllm "${COMMON[@]}" --output "$OUT/vllm.json"
"$PYTHON" scripts/bench_matrix.py \
  --target candidate \
  --op "$ROOT/src/h20_topk/ops/compute_topk_scores.py" \
  "${COMMON[@]}" \
  --output "$OUT/candidate.json"

"$PYTHON" - "$OUT/vllm.json" "$OUT/candidate.json" <<'PY'
import json
import sys

baseline = json.load(open(sys.argv[1], encoding="utf-8"))
candidate = json.load(open(sys.argv[2], encoding="utf-8"))
index = {
    (case["dtype"], case["B"], case["V"], case["K"]): case
    for case in candidate["cases"]
}
print("[smoke] vLLM median / candidate median")
for case in baseline["cases"]:
    key = (case["dtype"], case["B"], case["V"], case["K"])
    other = index.get(key)
    if not other:
        continue
    ratio = case["device_ms"]["median"] / other["device_ms"]["median"]
    print(
        f"[smoke] {key[0]} B={key[1]} V={key[2]} K={key[3]} "
        f"vllm={case['device_ms']['median']:.4f}ms "
        f"candidate={other['device_ms']['median']:.4f}ms ratio={ratio:.3f}x"
    )
PY

echo "[smoke] done"
