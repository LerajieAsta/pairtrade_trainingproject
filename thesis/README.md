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

**③ 其實有兩道關卡**，兩道都會靜默失效：渲染（`.ipynb` → `docs/`）
與**發佈**（`docs/` → 網站，來源是 **`main`**）。在功能分支上渲染完並 commit，
第一道過了、第二道沒過，網站照樣是舊的——2026-08-12 就這樣落後了 12 個提交。
驗證方式見下方「但檔案時間查不出『網站落後』」。

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

### 但檔案時間查不出「網站落後」——要抓線上頁面

上面那三行只證明**本地**產物是新的。Pages 的來源是 **`main` + `/docs`**
（repo 設定的 Deploy from a branch，倉庫裡沒有 `.github/workflows/`），
所以在功能分支上渲染完、甚至 commit 並 push 了分支，**網站仍然不會動**。

2026-08-12 實測：`main` 落後 12 個提交，公開頁面掛著三條當時已被推翻的宣稱
（「選哪些配對，不決定報酬」、regime「改善 19 格」、成本餘裕「6.5–12.8 bps」），
而本地 `ls -la` 一切正常、git 也乾淨。**兩邊都「沒錯」，只是不同步。**
同次還發現 `docs/slides/trading/rl_threshold_trading.html` 線上根本不存在，
而第四章 §4.2.3「增益來源④」整節都靠它。

所以驗證的最後一步是**抓線上頁面**，而不是看本地檔案：

```bash
# 1. 確認遠端 main 真的含新內容（合併前後都可查，不需切分支）
git fetch origin
git show origin/main:docs/slides/thesis/ch4_results.html | grep -c "某個新加的關鍵詞"

# 2. 確認 Pages 已重建（?cb= 是為了繞開 CDN 與工具的快取）
curl -sL "https://lerajieasta.github.io/pairtrade_trainingproject/index.html?cb=$RANDOM" \
  | grep -o "最後更新 [0-9-]*"
curl -sL "https://lerajieasta.github.io/pairtrade_trainingproject/slides/thesis/ch4_results.html?cb=$RANDOM" \
  | grep -c "某個新加的關鍵詞"

# 3. 新增的投影片要確認真的取得到（不是 404）
curl -sLo /dev/null -w "%{http_code} %{size_download}\n" \
  "https://lerajieasta.github.io/pairtrade_trainingproject/slides/trading/rl_threshold_trading.html?cb=$RANDOM"
```

推 `main` 後 Pages 約一分鐘內重建。查的時候記得**同時 grep 一個新加的關鍵詞
和一個已刪除的舊字串**——只查新的，會漏掉「新舊並存」這種渲染不完全的情況。

> ⚠️ 帶快取的抓取工具（含本專案常用的 WebFetch）對同一 URL 有數分鐘快取，
> 剛推完去查很可能拿到舊版而誤判為「Pages 沒更新」。加 `?cb=` 隨機參數即可。

### 合併到 main 時，別為了切分支動到 `.db`

`formation_data/*.db` 有追蹤但**長期帶著未提交的修改**（見專案慣例：不 stage、
更**絕不可** `git checkout --` 還原，git 版本是舊快照）。要把功能分支併進 `main`，
最安全的是**先推遠端 ref、再移動本地指標**，全程不碰工作區：

```bash
git push origin <branch>:main        # 遠端快進，working tree 完全沒被觸碰
git fetch origin && git branch -f main origin/main
git checkout main                    # 此時兩者同樹，checkout 不會更新任何檔案
```

事後用 `md5sum` 對一次 `.db` 確認沒被動到。

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
