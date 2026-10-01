"""各部分共用的投影片骨架、章節分隔頁、時間登記，以及由論文標題產生的頁首。

章節順序與標題一律同論文（`thesis/NN_*.md`）：摘要、第一章至第六章、參考文獻、附錄。
頁首的章名與節名由論文的 Markdown 標題讀取（`sec()`／`app()`），
論文標題改動後重建簡報即同步；節號在論文中找不到時直接報錯。
"""
import glob
import os
import re

from lib import (ACCENT, GREY, LEFT, LIGHT, NAVY, WHITE, CONTENT_W, box,
                 header, page_number, text, title)

HERE = os.path.dirname(os.path.abspath(__file__))
THESIS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "thesis")

_CN = "零一二三四五六七八九"


def _load_headings():
    """回傳 (節號→標題, 章號→章名, 附錄字母→附錄名)。"""
    secs, chaps, apps = {}, {}, {}
    for path in sorted(glob.glob(os.path.join(THESIS, "[0-9][0-9]_*.md"))):
        fence = False
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.lstrip().startswith("```"):
                    fence = not fence
                if fence:
                    continue
                m = re.match(r"#{2,3} ((?:\d+|[A-G])(?:\.\d+)+)\s+(.+?)\s*$", line)
                if m:
                    secs[m.group(1)] = m.group(2).strip()
                    continue
                m = re.match(r"# 第(.)章\s*(.+?)\s*$", line)
                if m:
                    chaps[_CN.index(m.group(1))] = m.group(2).strip()
                    continue
                m = re.match(r"# 附錄 ([A-G])\s*(.+?)\s*$", line)
                if m:
                    apps[m.group(1)] = m.group(2).strip()
    return secs, chaps, apps


SECS, CHAPS, APPS = _load_headings()

# 章節＝論文的章。索引 0 為摘要，1–6 為第一章至第六章。
PARTS = [("摘要", "摘要", "")] + [
    (f"第{_CN[i]}章", CHAPS[i], "") for i in range(1, 7)]


def chapter_label(i):
    return "摘要" if i == 0 else f"第{_CN[i]}章　{CHAPS[i]}"


def sec(*nums, tag=None, note=None):
    """頁首：第 N 章 章名 ｜ 節號 節名（可多節）。tag 為證據等級等附註。"""
    for n in nums:
        if n not in SECS:
            raise KeyError(f"論文中找不到節 {n}（請核對 thesis/*.md 的標題）")
    chap = int(nums[0].split(".")[0])
    body = "、".join(f"{n} {SECS[n]}" for n in nums)
    if note:
        body += f"（{note}）"
    label = f"{chapter_label(chap)}　｜　{body}"
    return label + (f"　〔{tag}〕" if tag else "")


def chap(i, what, tag=None):
    """頁首：第 N 章 章名 ｜ 自訂文字（不對應單一節時使用）。"""
    label = f"{chapter_label(i)}　｜　{what}"
    return label + (f"　〔{tag}〕" if tag else "")


def app(letter, *nums, tag=None, short=False):
    """頁首：附錄 X 附錄名 ｜ 節號 節名。short=True 時只列節號（節名太長時用）。"""
    if letter not in APPS:
        raise KeyError(f"論文中找不到附錄 {letter}")
    label = f"附錄 {letter}　{APPS[letter]}"
    for n in nums:
        if n not in SECS:
            raise KeyError(f"論文中找不到節 {n}")
    if nums and short:
        label += "　｜　" + "、".join(nums) + " 節"
    elif nums:
        label += "　｜　" + "、".join(f"{n} {SECS[n]}" for n in nums)
    return label + (f"　〔{tag}〕" if tag else "")


def supp(*nums):
    """頁首：附錄（正文補充）｜ 對應的正文節。"""
    for n in nums:
        if n not in SECS:
            raise KeyError(f"論文中找不到節 {n}")
    return "附錄（正文補充）　｜　" + "、".join(f"{n} {SECS[n]}" for n in nums)


# 由 build 腳本在建構前填入：{章索引: 分鐘}、{函式名稱: 頁碼}
PART_MIN = {}
PAGE = {}

# 函式名稱 → (函式, 預設分鐘)。順序與所屬章由 outline.OUTLINE 決定。
REGISTRY = {}


def reg(minutes, part=None):
    """登記一張投影片：minutes 為建議講述時間。part 參數保留相容，實際歸屬見 outline.py。"""
    def deco(fn):
        REGISTRY[fn.__name__] = (fn, minutes)
        return fn
    return deco


def header_size(label, full=12, room=68):
    """頁首只有一行：論文節名較長時縮小字級，不換行。room 為 12 pt 下可容納的全形字數。"""
    n = sum(1.0 if ord(c) > 0x2E7F else 0.55 for c in label)
    return full if n <= room else max(9.0, int(full * room / n * 2) / 2)


def std(d, label, ttl, dark=False, title_size=30):
    s = d.slide(dark=dark)
    header(s, label, dark=dark, size=header_size(label))
    title(s, ttl, dark=dark, size=title_size)
    page_number(s, d.n, dark=dark)
    return s


def fmt_min(m):
    total = int(round(m * 60))
    return f"{total // 60:02d}:{total % 60:02d}"


def agenda_rows(s, current=None, x=LEFT, y=1.95, dark=True, row_h=0.56,
                show_min=True):
    for i, (num, name, _ref) in enumerate(PARTS):
        yy = y + i * row_h
        on = (current is None) or (i == current)
        numcol = ACCENT if (current is not None and i == current) else (
            WHITE if dark else NAVY)
        txtcol = (WHITE if dark else NAVY) if on else ("7F93B5" if dark else GREY)
        text(s, x, yy + 0.04, 1.6, row_h, num, size=18, bold=True,
             color=numcol if on else txtcol)
        text(s, x + 1.7, yy + 0.04, 7.6, row_h, "" if i == 0 else name, size=19,
             bold=on, color=txtcol)
        if show_min and i in PART_MIN:
            text(s, x + 10.1, yy + 0.08, 1.6, row_h,
                 f"約 {PART_MIN[i]:.0f} 分鐘", size=13, color=txtcol, align="r")


def divider(d, part_idx, lead):
    num, name, _ref = PARTS[part_idx]
    s = d.slide(dark=True)
    text(s, LEFT, 0.55, CONTENT_W, 0.4, num, size=14, bold=True, color=LIGHT)
    text(s, LEFT, 0.95, CONTENT_W, 0.9, name, size=36, bold=True, color=WHITE)
    text(s, LEFT, 1.85, CONTENT_W, 0.6, lead, size=17, color=LIGHT)
    agenda_rows(s, current=part_idx, y=2.70, row_h=0.52)
    page_number(s, d.n, dark=True)
    return s
