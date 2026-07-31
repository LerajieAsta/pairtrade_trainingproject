# 論文章節草稿

本目錄存放**散文體**的論文章節草稿，供正式論文寫作參考。

## 與 `notebooks/thesis/` 的關係

同一章節有兩份，**文體刻意不同**：

| 位置 | 文體 | 用途 |
| :--- | :--- | :--- |
| `thesis/*.md` | **散文** —— 完整段落、連貫論證 | 論文正文寫作的參考底稿 |
| `notebooks/thesis/*.ipynb` | **條列** —— 表格、要點、callout | 口試投影片（quarto → revealjs） |

兩者實質內容一致，但呈現形式為各自用途最佳化。**修改其一時須同步另一份**——
目前無自動同步機制，這是已知的維護成本。

## 章節對照

| 章 | 散文稿 | 投影片 |
| :--- | :--- | :--- |
| 一　緒論 | `01_緒論.md` | `notebooks/thesis/ch1_introduction.ipynb` |
| 二　文獻探討 | `02_文獻探討.md` | `notebooks/thesis/ch2_literature.ipynb` |
| 三　研究方法 | `03_研究方法.md` | `notebooks/thesis/ch3_methodology.ipynb` |
| 四　實證結果 | `04_實證結果.md` | `notebooks/thesis/ch4_results.ipynb` |
| 五　結論 | `05_結論.md` | `notebooks/thesis/ch5_conclusion.ipynb` |

## 渲染投影片

```bash
cd notebooks && quarto render thesis/ch1_introduction.ipynb
```

輸出至 `docs/slides/thesis/`。`_quarto.yml` 的 render 清單已含 `thesis/*.ipynb`，
故 `quarto render` 亦會一併處理。
