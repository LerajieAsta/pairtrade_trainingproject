#!/usr/bin/env bash
# 逐支渲染 notebooks/ 的投影片，並**逐支驗證產物**。
#
# 為什麼不用 `quarto render`（不帶參數）：一次傳多支時，其中一支失敗會讓先前
# 幾支靜默不落檔，而終端仍顯示那幾支的處理訊息、結束碼也可能是 0
# （2026-08-04 實測，見 thesis/README.md）。故逐支跑，且比對檔案時間。
set -u
cd "$(dirname "$0")/../notebooks" || exit 1
START=$(date +%s)
FAIL=0
LIST=$(python - <<'PY'
import re, io
y = io.open("_quarto.yml", encoding="utf-8").read()
m = re.search(r"render:\n((?:\s*-\s*\S+\n)+)", y)
print("\n".join(re.findall(r"-\s*(\S+)", m.group(1))))
PY
)
for nb in $LIST; do
    out="../docs/slides/${nb%.ipynb}.html"
    printf '%-46s ' "$nb"
    if quarto render "$nb" >/dev/null 2>"/tmp/q.$$"; then
        if [ -f "$out" ] && [ "$(stat -c %Y "$out")" -ge "$START" ]; then
            printf 'OK   %s KB\n' "$(( $(stat -c %s "$out") / 1024 ))"
        else
            printf 'X 結束碼 0 但產物未更新\n'; FAIL=$((FAIL+1))
        fi
    else
        printf 'X 渲染失敗\n'; sed 's/^/      /' "/tmp/q.$$" | tail -5; FAIL=$((FAIL+1))
    fi
    rm -f "/tmp/q.$$"
done
echo
echo "失敗 $FAIL 支；耗時 $(( ($(date +%s) - START) / 60 )) 分"
exit $FAIL
