"""60 分鐘版的投影片順序。

章節順序與標題同論文：摘要 → 第一章至第六章 → 參考文獻 → 附錄。
每一項是投影片函式的名稱（定義於 part*.py、extra.py、backup.py）；
要調整順序或把某一頁移到附錄，只改這個檔案。
"""

# (章索引, [函式名稱])；章索引 0 為摘要、1–6 為第一章至第六章、None 為開場
MAIN = [
    (None, ["cover", "roadmap"]),
    (0, ["abstract"]),
    (1, ["div_ch1", "what_is_pairs", "two_routes", "the_gap",
         "research_questions", "scope"]),
    (2, ["div_ch2", "why_profit", "ggr_rules", "decay", "pair_selection",
         "trading_lit", "three_spaces", "tension", "eval_conditions",
         "lit_gap"]),
    (3, ["div_ch3", "architecture", "data", "rolling", "four_layers",
         "features_sources", "ranking", "cointegration", "drl_inputs",
         "drl_v2_v3", "ablation_design", "hypotheses", "dlthr_menu", "not_rl",
         "stats_design", "measures_grades", "params"]),
    (4, ["div_ch4", "h1_turnover", "h2_perf", "h3_cost", "seg2", "argmax",
         "seg3", "equity", "fake_gains", "robustness", "three_answers"]),
    (5, ["div_ch5", "rf_excess", "dsr", "three_lines", "pnl_flows",
         "forced_close", "limitations"]),
    (6, ["div_ch6", "one_liner", "contributions", "recommendations",
         "five_lessons", "thanks"]),
]

# 參考文獻頁由 extra.py 依 thesis/14_參考文獻.md 動態產生（extra.REF_SLIDES）。

# 附錄：先依論文附錄 A–G，再放正文各節的補充（依節次）。
APPENDIX = [
    "app_index",
    "b7_appendix_a",                    # 附錄 A
    "b4_fixes",                         # 附錄 B
    "b6_appendix_c", "b5_regime",       # 附錄 C
    "b1_checks",                        # 附錄 D
    "appendix_e", "b11_appendix_e",     # 附錄 E
    "b13_fw504", "b13b_ablation_fw504", # 附錄 F
    "b14_allocation",                   # 附錄 G
    "b9_ari",                           # 3.2.4
    "params_learning",                  # 3.3.2、3.4
    "rlthr", "counterfactual",          # 3.5、4.5（RL-THR 受控對照）
    "b10_fdr",                          # 3.6.5
    "b8_sensitivity",                   # 3.7
    "level_tension",                    # 4.4（門檻水準的替代解釋）
    "b2_retrain",                       # 4.6.1
    "b3_measures", "b12_cells",         # 5.1.1
    "sources_descriptive",              # 5.3
]
