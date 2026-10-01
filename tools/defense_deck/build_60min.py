"""產生約 60 分鐘版的口試簡報。

用法（於 repo 根目錄）：
    Project/Scripts/python.exe tools/defense_deck/build_60min.py
輸出：thesis/1150922_60min.pptx（以 thesis/1150922.pptx 的母片為底，不改動原檔）。

章節順序與標題同論文（摘要 → 第一章至第六章 → 參考文獻 → 附錄）；
投影片順序由 outline.py 決定，頁首的章名與節名由 common.py 讀論文標題產生。
"""
import importlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import common  # noqa: E402
from common import REGISTRY, fmt_min  # noqa: E402
from lib import Deck, notes  # noqa: E402

PART_MODULES = ["part0_2", "part3", "part4", "part5_6", "extra", "backup"]

TEMPLATE = os.path.join(ROOT, "thesis", "1150922.pptx")
OUT = os.path.join(ROOT, "thesis", "1150922_60min.pptx")

CUE_REFS = "【參考文獻：不講述，供查閱】"
CUE_APPENDIX = "【附錄：不在正式報告時間內，問答時依問題翻到這一頁】"

# 講稿中以此開頭的段落是補充說明：建議時間不含這些段落，時間不夠時跳過。
OPTIONAL = "〔可略〕"


def core_chars(body):
    """主線講稿字數（含標點、不含空白，扣掉〔可略〕段落）。"""
    paras = [p for p in (body or "").strip().split("\n\n")
             if not p.lstrip().startswith(OPTIONAL)]
    return len(re.sub(r"\s", "", "".join(paras)))


def plan():
    """回傳 [(函式名稱, 章索引或 None, 區段)]；區段 ∈ main／refs／appendix。"""
    import extra
    import outline
    seq = []
    for part, names in outline.MAIN:
        seq += [(n, part, "main") for n in names]
    seq += [(n, None, "refs") for n in extra.REF_SLIDES]
    seq += [(n, None, "appendix") for n in outline.APPENDIX]
    return seq


def main(out=OUT):
    for name in PART_MODULES:
        importlib.import_module(name)

    seq = plan()
    names = [n for n, _, _ in seq]
    missing = [n for n in names if n not in REGISTRY]
    if missing:
        raise KeyError(f"outline 列了未定義的投影片：{missing}")
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"outline 重複列出：{dup}")
    unused = [n for n in REGISTRY if n not in names]
    if unused:
        print(f"注意：已定義但未列入 outline 的投影片：{unused}")

    common.PART_MIN.clear()
    common.PAGE.clear()
    for i, (n, part, kind) in enumerate(seq):
        common.PAGE[n] = i + 1
        if kind == "main" and part is not None:
            common.PART_MIN[part] = common.PART_MIN.get(part, 0) + REGISTRY[n][1]

    deck = Deck(TEMPLATE)
    elapsed = 0.0
    total_core = 0
    for n, part, kind in seq:
        fn, minutes = REGISTRY[n]
        before = deck.n
        body = fn(deck)
        if deck.n != before + 1:
            raise RuntimeError(f"{n} 應該恰好產生一張投影片")
        slide = deck.prs.slides[-1]
        core = 0
        if kind == "main":
            elapsed += minutes
            core = core_chars(body)
            total_core += core
            secs = int(round(minutes * 60))
            cue = (f"建議 {secs // 60} 分 {secs % 60:02d} 秒｜講完時約 {fmt_min(elapsed)}"
                   if secs >= 60 else f"建議 {secs} 秒｜講完時約 {fmt_min(elapsed)}")
            if OPTIONAL in (body or ""):
                cue += f"｜{OPTIONAL}段落不計入時間，不夠時跳過"
            cue = f"【{cue}】"
        else:
            minutes = 0
            cue = CUE_REFS if kind == "refs" else CUE_APPENDIX
        notes(slide, cue + "\n\n" + (body or "").strip())
        deck.timeline.append((deck.n, n, minutes, elapsed, kind, core))

    deck.save(out)
    count = {k: sum(1 for t in deck.timeline if t[4] == k)
             for k in ("main", "refs", "appendix")}
    print(f"saved {out}")
    print(f"slides: {deck.n}（正文 {count['main']}，參考文獻 {count['refs']}，"
          f"附錄 {count['appendix']}）")
    print(f"total talk time: {fmt_min(elapsed)}")
    print(f"主線講稿 {total_core} 字（不含{OPTIONAL}段落），約每分鐘 {total_core / elapsed:.0f} 字")
    for k in sorted(common.PART_MIN):
        print(f"  {common.chapter_label(k)}: {common.PART_MIN[k]:.2f} min")
    return deck


if __name__ == "__main__":
    main()
