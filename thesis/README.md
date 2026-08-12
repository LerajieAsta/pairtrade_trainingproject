# 論文章節草稿

本目錄存放**散文體**的論文章節草稿，供正式論文寫作參考。

## 三軌：散文稿、投影片、已發佈網站

同一章節存在**三份**，前兩份文體刻意不同，第三份是第二份的產物：

| 軌 | 位置 | 文體／性質 | 用途 |
| :--- | :--- | :--- | :--- |
| ① | `thesis/*.md` | **散文** —— 完整段落、連貫論證 | 論文正文寫作的參考底稿 |
| ② | `notebooks/thesis/*.ipynb` | **條列** —— 表格、要點、callout | 口試投影片原始檔 |
| ③ | `docs/slides/thesis/*.html` | ②的 quarto 產物 | **GitHub Pages 實際對外顯示的內容** |

①②實質內容一致，呈現形式各自最佳化；③是②渲染後的靜態檔。

> **改動任一章時，三軌都要處理**：同步①②的內容，再重新渲染③。
> 三者皆無自動同步，這是已知的維護成本。

**最容易漏的是③。** 它不隨 `.md`／`.ipynb` 變動而更新，且是唯一對外可見的一份——
2026-08-04 就發生過只改①②、網站仍顯示七月舊版的情況（舊特徵數、無組合系統一節）。
Git 上是新的，網頁上是舊的，兩邊都「沒錯」。

## 章節對照

| 章 | ① 散文稿 | ② 投影片原始檔 | ③ 已發佈頁面 |
| :--- | :--- | :--- | :--- |
| 一　緒論 | `01_緒論.md` | `notebooks/thesis/ch1_introduction.ipynb` | `docs/slides/thesis/ch1_introduction.html` |
| 二　文獻探討 | `02_文獻探討.md` | `…/ch2_literature.ipynb` | `…/ch2_literature.html` |
| 三　研究方法 | `03_研究方法.md` | `…/ch3_methodology.ipynb` | `…/ch3_methodology.html` |
| 四　實證結果 | `04_實證結果.md` | `…/ch4_results.ipynb` | `…/ch4_results.html` |
| 五　結論 | `05_結論.md` | `…/ch5_conclusion.ipynb` | `…/ch5_conclusion.html` |
| 附錄 A　前行研究差異定位 | `06_附錄A_前行研究差異定位.md` | — | — |

> 附錄 A **只有散文稿**，未做投影片。它是 §5.3 其九的展開（與許鈞翔 2025 的
> 八項差異、其中進場時點一項的受控隔離結果），BH 校正後 0/3 顯著，
> 強度不足以進主軸；但其三條策略已計入第四章 DSR 的試驗宇宙（$N$ = 110），
> 故**必須揭露**。若日後補做投影片，記得同時加 `docs/index.html`／`docs/appendix.html`
> 的卡片與 `notebooks/_deck_nav.html` 的清單。

## 渲染到已發佈網站

```bash
cd notebooks
quarto render thesis/ch4_results.ipynb          # 逐一渲染（建議）
```

輸出至 `docs/slides/thesis/`。`_quarto.yml` 的 render 清單已含 `thesis/*.ipynb`，
故不帶參數的 `quarto render` 亦會一併處理。

### 務必逐一渲染並比對檔案時間

一次傳多個檔給 `quarto render` 時，**其中一個失敗會讓先前幾個靜默不落檔，
而終端仍顯示那幾個的處理訊息、結束碼也可能是 0**。
2026-08-04 實測：三章一起渲染，ch4 因路徑問題失敗，ch3 看似成功但 HTML 停在七月。

所以渲染後一定要驗證產物本身，不能只看結束碼：

```bash
ls -la docs/slides/thesis/           # 檔案時間應為剛才
grep -c "某個新加的關鍵詞" docs/slides/thesis/ch4_results.html
grep -c "李伯修" docs/slides/thesis/*.html    # 必須為 0
```

### 新增章節時要手動加入口

著陸頁分兩層，皆為**手寫**、不會自動列出新產生的頁面：

| 頁面 | 內容 |
| :--- | :--- |
| `docs/index.html` | 只放論文正文六項：第一～五章 + 主要結果總覽 |
| `docs/appendix.html` | 其餘一切：研究架構、形成期／交易期各策略投影片、績效比較、指標定義、主要結論、現役策略清單 |

新增投影片後須自行加卡片，否則該頁只能靠打路徑進入——
論文五章從 2026-07-31 發佈到 08-04 都處於這個狀態，首頁完全沒有連結。

改完可用雙向檢查（斷鏈 + 有產出卻沒被連到的孤兒頁）驗證：

```python
import io, os, re
links = set()
for page in ("index.html", "appendix.html"):
    html = io.open("docs/" + page, encoding="utf-8").read()
    links |= {l for l in re.findall(r'href="([^"]+)"', html)
              if not l.startswith(("http", "#"))}
print("斷鏈：", [l for l in sorted(links) if not os.path.exists(os.path.join("docs", l))])
have = {os.path.relpath(os.path.join(r, f), "docs").replace(os.sep, "/")
        for r, _, fs in os.walk("docs/slides") for f in fs if f.endswith(".html")}
print("孤兒頁：", sorted(have - links))
```

### 投影片內的導覽列

每份投影片右上角有「⌂ 首頁／附錄／上一頁／下一頁」浮動列，來源是
`notebooks/_deck_nav.html`（由 `_quarto.yml` 的 `include-after-body` 注入）。
頁面順序寫死在該檔的 `THESIS`／`APPENDIX` 兩份清單裡，**新增投影片時要一併加進去**，
否則該頁不會顯示導覽列（不在清單內就整條隱藏，不會壞掉、只是沒有出口）。

改完 `_deck_nav.html` 後，已渲染的 3.5 MB HTML 不會自己更新。不想全部重跑 quarto 時：

```bash
python tools/inject_deck_nav.py            # 直接改寫 docs/slides/ 下所有 HTML（可重複執行）
python tools/inject_deck_nav.py --check    # 只回報哪些頁面還沒有導覽列
```
