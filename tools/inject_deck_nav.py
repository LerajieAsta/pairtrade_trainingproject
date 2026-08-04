"""把 notebooks/_deck_nav.html 的導覽列補進已渲染的投影片 HTML。

`_quarto.yml` 的 include-after-body 只在重新渲染時生效，而 docs/slides/ 下
的檔案（單檔 3.5 MB、embed-resources）平常不會全部重跑。此腳本直接把同一段
片段寫進 </body> 之前，效果與重新渲染一致。

腳本可重複執行：以 deck-nav:start／end 為標記，既有的區塊會被新版取代。

    python tools/inject_deck_nav.py            # 寫入
    python tools/inject_deck_nav.py --check    # 只回報哪些檔案還沒有導覽列
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNIPPET = ROOT / "notebooks" / "_deck_nav.html"
SLIDES = ROOT / "docs" / "slides"

BLOCK = re.compile(r"<!-- deck-nav:start.*?<!-- deck-nav:end -->\s*", re.S)


def main() -> int:
    check_only = "--check" in sys.argv
    snippet = io.open(SNIPPET, encoding="utf-8").read().strip()

    missing, written = [], []
    for path in sorted(SLIDES.rglob("*.html")):
        html = io.open(path, encoding="utf-8").read()
        stripped = BLOCK.sub("", html)
        if stripped == html:
            missing.append(path)
        if check_only:
            continue
        if "</body>" not in stripped:
            print(f"跳過（找不到 </body>）：{path}")
            continue
        head, sep, tail = stripped.rpartition("</body>")
        io.open(path, "w", encoding="utf-8", newline="").write(
            head + snippet + "\n" + sep + tail
        )
        written.append(path)

    label = "缺少導覽列" if check_only else "已寫入"
    targets = missing if check_only else written
    for path in targets:
        print(f"{label}：{path.relative_to(ROOT)}")
    print(f"{label} {len(targets)} 個檔案（共掃描 {len(list(SLIDES.rglob('*.html')))} 個）")
    return 1 if (check_only and missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
