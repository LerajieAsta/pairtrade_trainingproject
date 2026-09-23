"""各部分共用的投影片骨架、章節分隔頁與時間登記。"""
from lib import (ACCENT, GREY, LEFT, LIGHT, NAVY, WHITE, CONTENT_W, box,
                 header, page_number, text, title)

PARTS = [
    ("一", "背景：配對交易是什麼", "§1.1、第二章"),
    ("二", "我想問的問題", "§1.2–1.3、§2.3.5"),
    ("三", "我怎麼做", "第三章"),
    ("四", "結果：三段證據", "第四章"),
    ("五", "所以，可以拿來賺錢嗎？", "第五章"),
    ("六", "結論", "第六章"),
]

# 由 build 腳本在建構前填入：{部分索引: 分鐘}
PART_MIN = {}

REGISTRY = []


def reg(minutes, part=None):
    """登記一張投影片：minutes 為建議講述時間，part 為所屬部分（0–5，None 為開場／備用）。"""
    def deco(fn):
        REGISTRY.append((fn, minutes, part))
        return fn
    return deco


def std(d, label, ttl, dark=False, title_size=30):
    s = d.slide(dark=dark)
    header(s, label, dark=dark)
    title(s, ttl, dark=dark, size=title_size)
    page_number(s, d.n, dark=dark)
    return s


def fmt_min(m):
    total = int(round(m * 60))
    return f"{total // 60:02d}:{total % 60:02d}"


def agenda_rows(s, current=None, x=LEFT, y=1.95, dark=True, row_h=0.62,
                show_min=True):
    for i, (num, name, ref) in enumerate(PARTS):
        yy = y + i * row_h
        on = (current is None) or (i == current)
        numcol = ACCENT if (current is not None and i == current) else (
            WHITE if dark else NAVY)
        txtcol = (WHITE if dark else NAVY) if on else ("7F93B5" if dark else GREY)
        text(s, x, yy, 0.8, row_h, f"{i + 1}", size=22, bold=True,
             color=numcol if on else txtcol)
        text(s, x + 0.9, yy + 0.04, 6.6, row_h, name, size=20, bold=on,
             color=txtcol)
        text(s, x + 7.6, yy + 0.10, 2.4, row_h, ref, size=13, color=txtcol)
        if show_min and i in PART_MIN:
            text(s, x + 10.1, yy + 0.10, 1.6, row_h,
                 f"約 {PART_MIN[i]:.0f} 分鐘", size=13, color=txtcol, align="r")


def divider(d, part_idx, lead):
    num, name, ref = PARTS[part_idx]
    s = d.slide(dark=True)
    text(s, LEFT, 0.55, CONTENT_W, 0.4, f"第{num}部分　｜　{ref}", size=14,
         bold=True, color=LIGHT)
    text(s, LEFT, 0.95, CONTENT_W, 0.9, name, size=36, bold=True, color=WHITE)
    text(s, LEFT, 1.85, CONTENT_W, 0.6, lead, size=17, color=LIGHT)
    agenda_rows(s, current=part_idx, y=2.85, row_h=0.56)
    page_number(s, d.n, dark=True)
    return s
