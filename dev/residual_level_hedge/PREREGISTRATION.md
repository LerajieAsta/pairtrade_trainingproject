# 預先登記：殘差報酬訊號＋水準價差部位（交易端）

寫定日期：2026-10-21。**本檔在跑之前定稿。**

---

## 一、問題

`dev/residual_spread/RESULTS.md`：以殘差報酬累積（Avellaneda & Lee 2010）取代水準價差後，字面上 14/15 臂 Sharpe 改善
（中位 +0.178），但部位依報酬迴歸的 $1:\lvert\beta_{\text{ret}}\rvert$ 配置，而 $\beta_{\text{ret}}$ 受衰減偏誤（中位 0.61–0.78 vs
水準 0.98），B 腳避險不足，組合留下約 +0.04 的市場 β；扣市場後四臂中三臂的改善大多消失。該案結論第 4 點建議：
**以殘差報酬產生訊號、部位仍以水準 β 配置，隔離「價差定義」本身的效果。**

本案即此檢驗：**部位口徑與現行完全相同，只把進出場的時點改由殘差報酬 z 決定，績效是否改善？**

> ⚠ 本案動機來自看過殘差報酬研究結果之後；為此判準沿用該案的同一組門檻，並加入市場 β 的效度條件。

## 二、規則（寫死）

- **訊號**：與 `_RR` 完全相同——形成期日報酬 OLS $r_A = \alpha + \beta_{\text{ret}} r_B + \varepsilon$，
  $X(t) = \sum (r_A - \alpha - \beta_{\text{ret}} r_B)$，$Z = (X - \bar X_{\text{form}})/\operatorname{sd}(X_{\text{form}})$；
  進場 $|Z| > 2$、出場穿越 0、停損與期末強平不變。
- **部位**：與現行水準價差**逐字相同**——避險比率用形成期水準 β（formation DB 的 `Hedge_Ratio`），
  兩腳權重用 `_leg_scales`（signal 避險 $1/\sigma_A : \lvert\beta\rvert/\sigma_B$；OLS 路徑為 $1:\lvert\beta\rvert$），總名目額不變。
- 實作：`zscore_trading` 新增 `residual_hedge="return"|"level"`（預設 `"return"`，`_RR` 逐位元不變），本案用 `"level"`，檔名後綴 `_RRL`。

## 三、範圍與唯一變因

| 項目 | 規定 |
|:---|:---|
| 臂 | 主軸 15 支 `Grid {NOGRP,GICS,HDB,AGG,KM}-{SSD,DTW,SDP}`（與殘差報酬研究相同） |
| 網格 | Top {1,3,5,10,20} × 停損 {0,5,15}% = 每臂 15 格，共 **225 格** |
| 對照 | 主要：result.db 現行同格（只差訊號）；次要：residual_spread.db 的 `_RR` 同格（只差部位） |
| 寫入 | `results/residual_level_hedge.db`（約 30 GB；目前剩 39.2 GB，磁碟低於 5 GB 時中止） |

## 四、判準（跑前寫死）

主要指標：逐臂 ΔSharpe = (15 格 Sharpe_Raw 平均)_RRL − (15 格 Sharpe_Raw 平均)_現行，n = 15。

**判定「殘差報酬訊號較佳」須同時滿足：**
1. 15 臂中位 ΔSharpe ≥ +0.05；
2. ≥ 10/15 臂為正；
3. 後半期（2014-01-02 起）15 臂中位 ΔSharpe > 0；
4. **效度條件**：RRL 組合（15 格等權逐日報酬）對成分股等權市場報酬的 β，15 臂中位 ≤ +0.02。
   若 1–3 成立但 4 不成立 → 判定為「曝險混淆，不成立」。

**判定「現行價差較佳」須同時滿足：** 中位 ≤ −0.05、≤ 5/15 為正、後半期中位 < 0。其餘 → 無差異。

另報（不入判準）：15 格等權逐日損益差分（4.3 同程序，15 臂 BH）；扣市場後 Sharpe（RRL、現行、`_RR`）；
RRL − `_RR` 的 ΔSharpe（部位口徑的效果）；進場次數、勝率、停損。

**績效不佳的處置**（同 2026-10-21 使用者指示）：若「殘差報酬訊號較佳」不成立，`residual_level_hedge.db` 只留摘要、刪除逐日紀錄。

## 五、試驗宇宙

既有 15 支 METHOD 下的配置變體：method N 不變（88），config **+225**（庫外）。跑完後重算 var_sr 與 SR₀。

## 六、接線驗證（跑網格前必須全過）

- **V1**：`residual_hedge` 未設時，`_RR` 的 GICS-SSD、NOGRP-DTW Top1/SL0 摘要與 residual_spread.db 逐位元相同；
  `spread_mode` 未設時與 result.db 逐位元相同。
- **V2**：RRL 試跑格的 `ZScore` 欄與 `_RR` 同格逐列相同（訊號未變）。
- **V3**：RRL 進場時兩腳名目額比與現行引擎同一配對以 `_leg_scales` 與水準 β 算出的比例相同（容差 1e-9），總名目額 = 每對資金；
  且 HDB（OLS 路徑）臂的比例為 $1:\lvert\beta_{\text{level}}\rvert$。
- **V4**：RRL 某配對期若訊號時點恰與現行相同（進出場日皆同），其交易損益與現行相同——驗證部位真的沿用現行口徑。

---

## 附記（2026-10-21，接線驗證之後補記；規則與判準未改動）

- **V1 通過**：現行 GICS-SSD 與 result.db、`_RR` 的 GICS-SSD／NOGRP-DTW 與 residual_spread.db，摘要逐位元相同。
- **V2 通過**：GICS-SSD、HDB-SSD 的 RRL 與 `_RR` 各 37,170 列，ZScore 逐列相同。
- **V3 通過**：894 次進場，傳入的避險比率與形成期水準 β 完全相同、名目額比差 4.4e-16、總名目額相對差 2.4e-16。
  HDB-SSD 亦走 signal 權重（主軸臂皆 `ignore_ols_alpha=True`，無 OLS 路徑），§六「HDB 為 1:|β|」的預期據此更正。
- **V4 改寫後通過**：原寫法比對交易損益是否相同，但每對資金 = 當下權益 ÷ 槽位（隨權益複利），兩邊先前損益不同，資金即不同。
  改為：部位序列與現行完全相同的配對期，逐日損益比 RRL／現行在期內為常數（只差資金倍數）。GICS-SSD 9 期、HDB-SSD 13 期，
  最大相對偏離 1.7e-05（4 位捨入誤差）。
