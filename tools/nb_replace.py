# -*- coding: utf-8 -*-
"""
在 notebook 的 markdown 儲存格內做**精確字串取代**，並強制每筆恰好命中一次。

為什麼要有這支：`notebooks/thesis/ch*.ipynb` 是 `thesis/*.md` 的投影片版，
兩者各自維護同一批數字。手改 JSON 很容易改到一半或改錯儲存格，
而 notebook 是二進位般的一整包 JSON，改壞了 diff 也看不出來。
本工具把「改哪一段、改成什麼、命中幾次」變成可檢查的清單。

用法（規格檔為 JSON：{"檔案路徑": [[舊字串, 新字串], ...]}）：
    python -m tools.nb_replace spec.json
任一筆命中次數不是 1 就整支中止、不寫檔——寧可停下也不要寫出半套數字。
"""
import io
import json
import sys


def apply(spec: dict) -> None:
    plans = []
    for path, reps in spec.items():
        nb = json.load(io.open(path, encoding="utf-8"))
        hits = {i: 0 for i in range(len(reps))}
        for cell in nb["cells"]:
            if cell["cell_type"] not in ("markdown", "raw"):
                continue
            src = "".join(cell["source"])
            for i, (old, new) in enumerate(reps):
                n = src.count(old)
                if n:
                    hits[i] += n
                    src = src.replace(old, new)
            cell["source"] = src.splitlines(keepends=True)
        bad = [(i, hits[i], reps[i][0][:60]) for i in hits if hits[i] != 1]
        if bad:
            for i, n, frag in bad:
                print(f"  ✗ {path} 第 {i} 筆命中 {n} 次：{frag}")
            raise SystemExit(f"{path}：{len(bad)} 筆未恰好命中一次，全檔未寫入")
        plans.append((path, nb, len(reps)))

    for path, nb, n in plans:
        with io.open(path, "w", encoding="utf-8") as fh:
            json.dump(nb, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        print(f"  ✔ {path}：{n} 筆")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    apply(json.load(io.open(sys.argv[1], encoding="utf-8")))
