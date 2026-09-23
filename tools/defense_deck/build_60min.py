"""產生約 60 分鐘版的口試簡報。

用法（於 repo 根目錄）：
    Project/Scripts/python.exe tools/defense_deck/build_60min.py
輸出：thesis/1150922_60min.pptx（以 thesis/1150922.pptx 的母片為底，不改動原檔）。
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import common  # noqa: E402
from common import REGISTRY, fmt_min  # noqa: E402
from lib import Deck, notes  # noqa: E402

PART_MODULES = ["part0_2", "part3", "part4", "part5_6", "backup"]

TEMPLATE = os.path.join(ROOT, "thesis", "1150922.pptx")
OUT = os.path.join(ROOT, "thesis", "1150922_60min.pptx")


def main(out=OUT, only=None):
    for name in PART_MODULES:
        if only and name not in only:
            continue
        if os.path.exists(os.path.join(HERE, name + ".py")):
            importlib.import_module(name)

    for _, minutes, part in REGISTRY:
        if part is not None:
            common.PART_MIN[part] = common.PART_MIN.get(part, 0) + minutes

    deck = Deck(TEMPLATE)
    elapsed = 0.0
    for fn, minutes, part in REGISTRY:
        body = fn(deck)
        slide = deck.prs.slides[-1]
        if minutes > 0:
            elapsed += minutes
            secs = int(round(minutes * 60))
            cue = (f"【建議 {secs // 60} 分 {secs % 60:02d} 秒｜講完時約 {fmt_min(elapsed)}】"
                   if secs >= 60 else f"【建議 {secs} 秒｜講完時約 {fmt_min(elapsed)}】")
        else:
            cue = "【備用投影片：問答時使用】"
        notes(slide, cue + "\n\n" + (body or "").strip())
        deck.timeline.append((deck.n, fn.__name__, minutes, elapsed))

    deck.save(out)
    main_slides = sum(1 for t in deck.timeline if t[2] > 0)
    print(f"saved {out}")
    print(f"slides: {deck.n}（正片 {main_slides}，備用 {deck.n - main_slides}）")
    print(f"total talk time: {fmt_min(elapsed)}")
    for k in sorted(common.PART_MIN):
        print(f"  part {k}: {common.PART_MIN[k]:.2f} min")
    return deck


if __name__ == "__main__":
    only = sys.argv[1:] or None
    main(only=only)
