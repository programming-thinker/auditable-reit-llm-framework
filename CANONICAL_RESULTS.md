# CANONICAL RESULTS — 唯一权威数据表（Single Source of Truth）

> **本文件是论文所有数字的唯一来源。** 任何与本表冲突的旧文档数字一律以此为准。
> 每个数字都标注了可追溯的源文件。最后更新：2026-06-26（DeepSeek test 完成后回填终稿数字）。

---

## 0. 状态标记

| 标记 | 含义 |
|------|------|
| ✅ FINAL | 已锁定、可追溯、可写入论文 |
| ⏳ PROVISIONAL | 基于部分数据（DeepSeek test 未跑完 575），跑完后回填 |

---

## 1. 数据与特征（✅ FINAL）

| 项 | 权威值 | 来源 |
|----|--------|------|
| REIT 数量 | **25** | `data/processed/splits/*.csv` |
| 特征数（用于分类） | **13 numeric + sector**（≈18 维 one-hot 后） | `outputs/tables/v6_selected_features.csv` |
| ~~"47 features"~~ | **错误，弃用** | 90 列 enriched panel 中 70 个宏观变量横截面 std=0，对分类无用 |
| 唯一保留的 enriched 特征 | `dividend_yield_lag1` | — |
| Train 行数 | **2014**（OLS dropna 后 1739） | `enriched_train_2015_2021.csv` |
| Validation 行数 | **600** | `enriched_validation_2022_2023.csv` |
| Test 行数 | **575** | `enriched_test_2024_2025.csv` |
| 标签定义 | `future_ret_1m > +2%`→increase，`< −2%`→reduce，其余 hold | `src/06c_build_reit_monthly_panel.py:44` |

### Test 集标签分布（✅ FINAL）
| 类 | 数量 | 占比 |
|----|------|------|
| increase | 228 | 39.7% |
| hold | 182 | 31.7% |
| **reduce** | **165** | **28.7%** |

---

## 2. 结构化基线（✅ FINAL，reproduce_v6 PASSED 6dp）

| 方法 | 指标 | 值 | 来源 |
|------|------|-----|------|
| OLS（enriched） | Test R² | **−4.108** | `outputs/tables/ols_metrics_enriched.csv` |
| OLS（enriched） | Val R² / Train R² | −8.185 / 0.397 | 同上（过拟合：拟合 train，泛化崩） |
| Multinomial Logistic（`class_weight=balanced`） | **Test reduce recall** | **0.0%（0/165）** | `quant_only_confusion_matrix_test.csv` |
| Multinomial Logistic | Test hold recall | 74%（135/182） | 同上 |
| Multinomial Logistic | Test reduce 预测次数 | **0 次**（argmax 产物，非真地板） | 同上 |
| Fama-MacBeth | 显著项 | 仅 sector（2/66） | `fama_macbeth_test_enriched.csv` |

> ⚠️ **重要措辞**：基线"reduce recall=0%"是 **argmax 假象**（模型用了 balanced 权重仍从不把 reduce 当 argmax）。它**不是**真正的对比地板。真正的地板见 §3 的 trivial baselines。

---

## 3. LLM vs 平凡基线（同期同样本，⏳ PROVISIONAL — 268/575）

> **这取代旧的、无效的"0%→26.2%"跨期对比。** 全部在同一 Test 2024–2025、同一批样本上。
> 来源：`outputs/llm_deepseek_test/headline_comparison.csv`、`significance_bootstrap.csv`。

**✅ FINAL — v2 5-agent 框架（全 575，2024–2025，修复 disclosure + Fundamentals agent）**
| 方法 | reduce recall | reduce precision | hold recall | accuracy |
|------|--------------|------------------|-------------|----------|
| **LLM v2 (5-agent)** | **0.206** | 0.291 | 0.41 | 0.334 |
| Logistic-argmax（稻草人） | 0.000 | 0.000 | 0.74 | 0.372 |
| **Random-at-budget**（同开火率随机） | **0.204** | 0.287 | — | — |
| Threshold-logistic（调阈值） | 0.188 | 0.265 | — | — |

**Bootstrap（REIT-block，95% CI）**：LLM − Random reduce recall 差 = **+0.001，CI [−0.12, +0.13]**（跨 0）。

> ✅ **定论**：v2 框架（修好文本 + 加基本面）reduce recall 0.206 ≈ 随机 0.204，**不显著超越平凡基线**。
> 预测受系统性天花板封顶（§经济学支柱）。**贡献不在预测**，在 §4 的可分解理由接地度。

---

## 4. 污染 / 接地审计（✅ FINAL — v2, 575 条）

来源：`outputs/llm_deepseek_test/contamination_summary.csv`

| 项 | v1 (XBRL) | **v2 (修复+5agent)** | 含义 |
|----|-----------|----------------------|------|
| 输入 look-ahead 违规 | 0 | **0** | filing-date 防线完好 |
| reduce 预测有具体 filing 事实支撑 | 57% | **88%** | 高度接地（题目"可分解理由"的可测量证据） |
| reduce 平均引用事实数 | 2.84 | **3.58** | — |
| reduce 平均风险术语 | 0.48 | **1.23** | — |
| 含具体金额/百分比 | 53% | **82%** | — |
| 全样本接地率 | 68% | **83%** | — |

> **这是题目 "Decomposable Rationales Beyond" 的核心证据**：v2 框架的理由 88% 引用具体 filing 事实/数值，
> 可审计、可验证——超越不透明基线的系数输出。预测受系统性封顶；价值在可审计诊断，非预测。
> 权重层污染无法完全排除（局限）；但抽取-验证设计 + 88% 接地大幅缓解。

---

## 5. 漂移数字修正清单（交人工确认后改）

> 不直接改 Zone 1 / 已提交文档，仅列出，由作者批量替换。

### 5.1 "47 features" → "13 numeric + sector"
出现于：`THESIS_DRAFT_OUTLINE.md`、`THESIS_DRAFT_OUTLINE_FINAL.md`、`README.md`、`RESUME_SESSION.md`、`LLM_PHASE_PROGRESS_2026-06-24.md`、`THESIS_FEATURE_CORRECTION.md`（后者是修正说明，保留）。

### 5.2 "0% → 26.2%" 跨期招牌句 → §3 同期对比；**千问全部剔除**
出现于：`PROJECT_MASTER_STATUS.md`、`THESIS_DRAFT_OUTLINE.md`、`THESIS_DRAFT_OUTLINE_FINAL.md`、`RESUME_SESSION.md`、`TEST_RUN_STATUS.md`、`LLM_PHASE_PROGRESS_2026-06-24.md`。
**改法（2026-06-26 决定）**：**千问（Qwen）整体从论文剔除**——26.2% mini、396 partial 一律不进论文、不作跨模型 robustness。
唯一报告的 LLM 系统 = DeepSeek V4-Flash（§3 同期 575）。Qwen 的 jsonl 仅本地归档备查。

### 5.3 完成度数字（76% vs 85%）
出现于：`PROJECT_MASTER_STATUS.md`(85%)、`RESUME_SESSION.md`(76%)、两份 OUTLINE。
**改法**：统一为一个口径（建议按章节加权，写定后只在本表维护）。

---

## 6. 待回填（575 跑完后）
- [x] §3 全部数字（headline_comparison.csv 全量）
- [x] §3 bootstrap CI（全量）
- [x] §4 污染审计（全量 575）
- [x] §3 最终判定措辞

---

## 7. 论文统计表格（权威源，2026-07-02 期刊化重构后）

> **2026-07-02 起论文改为期刊结构（8 节，Table 1–8 + A1）**。排版表格的唯一生成源是
> **`analysis/make_tables_tex.py`** → **`paper/_tables/table{1..8,A1}.tex`**（booktabs 三线表，
> 纯读取已有 CSV，无重算）；markdown 源里的简化表仅供阅读，构建时被 `.tex` 版替换。
> 旧 make_tables.py → outputs/tables_v2/ 的 9 表版本对应重构前的稿子（备份于
> `_legacy/docs/THESIS_DRAFT_pre_journal_restructure_2026-07-02.md`）。

| 新表号 | 内容（Panel 结构） | 生成源文件 |
|---------|------|-----------|
| Table 1 | A 标签分布 / B 13 分类特征描述统计 / C 9 基本面 | `splits/*.csv` + panel + `firm_fundamentals_panel.csv` |
| Table 2 | A 混淆矩阵 / B 调参与模型族稳健性 | `quant_only_confusion_matrix_test.csv` + `tuned_robustness.csv` |
| Table 3 | A OLS OOS R² / B Fama–MacBeth（经典 t + NW t） | `ols_metrics_enriched.csv` + `extended_ols.csv` + `extended_fama_macbeth.csv` + `fama_macbeth_nw.csv` |
| Table 4 | 主结果对比 + bootstrap CI + MDE 注 | `headline_comparison.csv` + `significance_bootstrap.csv` + `inference_robustness.csv` + `power_mde.csv` |
| Table 5 | 通道消融（Δ hits 列，无 yes/no） | `ablation_channels.csv` |
| Table 6 | 系统性分解：raw vs market-adjusted | `systematic_vs_idiosyncratic.csv` + `market_adjusted_labels.csv` |
| Table 7 | 接地/事实性/judge/人机一致审计 | `contamination_summary.csv` + `factuality_audit.csv` + `rationale_quality_audit_summary.csv` + `spotcheck_agreement.csv` |
| Table 8 | 理由质量：judge 分数 + 人机校准 κ | `rationale_quality_audit_summary.csv` + `spotcheck_agreement.csv` |
| Table A1 | 变量定义（构造/时点/来源） | 生成器内定义，对照 §3.2 |

> 文献库：`REFERENCES.md` 已重建为 57 条（按三辩论组织，全部 DOI/arXiv 于 2026-06-27 联网核实），
> 旧 V5 版归档于 `_legacy/docs/REFERENCES_legacy.md`。架构图 `outputs/figures_v2/fig_architecture.png` 已作为
> 论文 Figure 5.1 嵌入。`reproduce_v6` 在本轮改动后仍 PASS（6dp），确认未触碰 Zone 1。

---

## 8. 外审回应审计（2026-06-27 新增，全部来自新增只读脚本，可重跑追溯）

> 针对外审六条意见新增三个 `analysis/` 只读脚本，数字全部可由脚本重跑复现，已写入论文。
> 未触碰 Zone 1；`reproduce_v6` 仍 PASS。

### 8.1 理由质量审计（② entailment / relevance / actionability）
| 项 | 权威值 | 来源 |
|----|--------|------|
| judge 模型 | **DeepSeek V4-Pro**（独立于 V4-Flash 生成器，降自评偏差） | `analysis/rationale_quality_audit.py` |
| 评分 reduce 决策数 | **114 / 117**（3 条不可解析） | `outputs/llm_deepseek_test/rationale_quality_audit.csv` |
| mean entailment | **1.22 / 2**（100% ≥1；22% ==2，即 25/114 完全蕴含） | 同上 |
| mean relevance | **1.97 / 2**（100% ≥1） | 同上 |
| mean actionability | **1.61 / 2** | 同上 |
| 成本 | ~**USD 0.087**（prompt 115,918 / completion 129,836 tok） | `audit_log/cost_ledger.jsonl` |
| 人工抽查表 | **45 行**（无自动分，防锚定） | `audit_log/rationale_spotcheck_sheet.csv` |

> ✅ 解读：高 relevance + 仅部分 entailment，与"系统性风险→firm 层信号混合"一致。

**人工校准（45 条盲评，作者打分 vs 机器 V4-Pro）** — `spotcheck_agreement.csv` / `audit_log/rationale_spotcheck_human.csv`
| 维度 | 人工均值 | 机器均值 | 完全一致 | ±1 内 | quad-κ |
|------|---------|---------|---------|-------|--------|
| **entailment**（核心维度） | 1.18 | 1.16 | **93%** | 100% | **0.76**（substantial） |
| relevance | 1.78 | 1.93 | 80% | 100% | 0.23（κ 悖论：margin 饱和，非分歧） |
| actionability | 1.91 | 1.44 | 51% | 98% | 0.06（人工更宽松，唯一明显分歧，从不差>1） |
| 总体（135 配对） | 1.62 | 1.51 | 75% | 99% | 0.47 |

> ✅ 核心维度 entailment 人机高度一致（κ=0.76）；人工独立复现"部分蕴含"模式。actionability 分歧如实报告。
> 多评审 κ 研究仍为 future work。45 条全部匹配到机器分（45/45）。

### 8.2 重述 / Point-in-Time 审计（⑤）
| 项 | 权威值 | 来源 |
|----|--------|------|
| 原始 XBRL 重述率 | **10.1%**（318/3,136 firm-period-concepts 被后续修订） | `restatement_audit_summary.csv` |
| test 决策实际取值中重述占比 | **0 / 4,232 = 0.0%** | 同上 |

> ✅ 解读：filed≤t 无穿越（**无条件**）；test 窗口 **0 重述回灌**——`asof()` 取最新财年，
> 其 t 时点值即原始申报。结论：是有意且合规的 PIT 选择，**非 leakage bug**。来源 `analysis/restatement_audit.py`。

### 8.3 Fama–MacBeth Newey–West 稳健性（⑥，Extended 面板 = Table 5）
| 特征（test 窗口，T=23，maxlags=2） | 经典 t | NW t | NW 下 5% 显著 |
|----|------|------|------|
| `leverage` | -2.72 | **-2.95** | 是 |
| `debt_to_equity` | 3.11 | **2.73** | 是 |
| `ret_6m` | -2.29 | **-3.04** | 是 |
| `leverage`（full，T=118，maxlags=4） | -3.11 | **-3.00** | 是 |

> ✅ 复现 vs 已发布经典 t **最大差 0.000**。NW 下三个 test 显著 pricing 项**全部存活**；
> HAC 仅放大 SE，**预测**结论不受影响（pricing≠prediction）。来源 `analysis/fama_macbeth_nw.py` → `fama_macbeth_nw.csv`。

### 8.4 措辞修正（①③④，无新数字）
- ① "opaque baseline" → 精确对照"global parametric coefficients vs. per-decision input-grounded provenance-tracked rationales"（baseline 本身可解释，贡献是**可分解证据**非"治黑箱"）。
- ③ "非 model-capacity gap" → "no evidence interpretable model families recover firm-level signal"，重锚 model-free 的 firm R²=0.007 vs month 0.43 + V4-Pro≈Flash。
- ④ 宏观"useless" → "uninformative for within-month cross-sectional discrimination" + 指向 §6.7（弱、不可时序择时）。

---

## 9. 修订轮新增审计（2026-07-02，全部 analysis/ 只读脚本可重跑）

### 9.1 市场调整标签稳健性（堵"标签构造机械产物"质疑）
来源：`analysis/market_adjusted_labels.py` → `outputs/fundamentals_robustness/market_adjusted_labels.csv`

| 项 | 原始标签（raw return ±2%） | 市场调整标签（LOO 超额收益 ±2%） |
|----|--------------------------|--------------------------------|
| 时间聚集（月度 reduce 率 SD / 二项零假设） | **3.26×** | **0.90×**（聚集消失） |
| 月度固定效应 R²（谁 reduce） | **0.433** | **0.032** |
| firm 特征 R² | 0.007 | 0.011 |
| tuned logit reduce recall（test） | 0.000 | **0.000**（两个特征集均为 0，fire 0 次） |
| test reduce 事件数 | 165 | 185 |

> ✅ 定论：去掉共同因子后时间聚集如系统性解释所预测般消失，但 firm-level 不可预测性**不变**——
> null 不是 raw-return 标签的机械产物，在市场/特异分解两侧都成立。已写入论文 §6.6/摘要/§7.1/结论。

### 9.2 统计功效 / 最小可检测效应（MDE）
来源：`analysis/power_mde.py` → `outputs/fundamentals_robustness/power_mde.csv`

| 方案 | bootstrap SE | MDE（80% 功效，模拟法） |
|------|-------------|------------------------|
| REIT-block | 0.0624 | **12.5 pp** |
| month-block | 0.0379 | **10.0 pp** |

> ✅ 措辞：设计可排除"经济上大的优势"（≥10–12.5pp recall 提升），但 <10pp 的小效应不可检测——
> 报告为 "no evidence of an edge" 而非 "evidence of no edge"。已写入论文 §6.2/摘要。

### 9.3 聚合稳健性重跑（⚠️ 修正旧数字）
来源：`analysis/aggregation_robustness.py`（已改为落盘）→ `outputs/llm_deepseek_test/aggregation_robustness.csv`

- ~~"LLM 聚合器与确定性均值 85% 一致"~~ **错误，弃用**（旧 CSV 是全量跑完前的 partial 数据）。
- **正确值：77.4%（445/575）**；LLM 聚合器 reduce 开火 117 次 vs 均值规则 57 次（约 2×）。
- 两条规则各自都不超过同预算随机基线 → 聚合方式不影响预测结论。已写入论文 §5.4/Appendix B.5。

### 9.4 Masked-identity 污染探针（✅ 完成 2026-07-02，70 决策子集，同 reasoner_subset 分层种子）
来源：`analysis/masked_identity_subset.py` → `audit_log/masked_probe/` + `outputs/llm_deepseek_test/masked_probe_comparison.csv`

| 项 | 权威值 |
|----|--------|
| 完成决策 | **69 / 70**（1 条解析失败，剔除） |
| 掩码验证 | **700 条渲染消息全扫描，0 泄漏**（命中即调用前抛 LeakError） |
| reduce recall（masked vs unmasked，同批决策） | **7/39 = 0.179 vs 6/40 = 0.150**（一个命中量级，方向与"背记假说"相反） |
| 接地率（all / reduce） | masked 0.94 / 0.95 vs unmasked 0.97 / 0.97 |
| 单条标签一致率 | 65.2%（n=69；温度 0 下输入扰动的正常敏感度） |
| 成本 | **USD 0.52**（独立 ledger：`audit_log/masked_probe/cost_ledger.jsonl`） |

> ✅ 定论：隐去公司身份后预测性能与接地率均无退化 → **与"模型靠背记公司层面 2024–25 结果"不一致**，
> firm-level 污染通道被直接检验并关闭。残余通道如实声明：决策日期仍可见，不排除对**市场整体**
> 历史的记忆——但论文评估的单位是 firm-level 预测，该层面现在是"已检验"而非"仅论证"。
> 已写入论文 Section 6.3（新结构）；旧第 8 章"首要局限"相应改写。

## 9.5 文献扩充 + 管线审计轮（2026-07-05）

### 文献（57 → 70 条，全部两轮联网核实）
新增 13 条：Lopez-Lira & Tang（JFE 已接收）、Chen–Kelly–Xiu、Sarkar & Vafa、Gu–Kelly–Xiu (RFS 2020)、
Leow & Lindenthal (REE 53(3))、Li (2026, C-REITs 多智能体)、Lütticke et al. (ERES 2025)、
Campbell et al. (2014 RAST)、Kim–Wang–Zhang (2019 CAR)、Tetlock (2007 JF)、Doran–Peterson–Price (2012 JREFE)、
Ludwig–Mullainathan–Rambachan (ARE 2026)、Wu et al. (2025 匿名化)。正文 8 处编织；build.py SPEC 已同步。

### 两个披露通道 bug（已修复 + 暴露审计，论文 §3.3 如实披露）
| Bug | 影响面（575 条 as-run） | 来源 |
|----|----|----|
| 8-K 取最旧五条（`[:5]` 于升序表） | **67.7%（389/575）** 决策丢最新 8-K，均 3.3 条；组间 recall 0.154 vs 0.295（firm/period 混杂，不作因果读） | `audit_8k_recency.csv` |
| Item 1A 锚点误命中交叉引用 | 真正命中风险因子章节仅 **23.8%（137/575）**；46 条（AMT/VICI）清洗文本中无可定位章节 | `audit_item1a_anchor.csv` |

> 消融显示去掉披露通道 recall 0.212 ≥ 0.206 → 缺陷不可能制造整体 null；§5.3 措辞已收窄为
> "on the inputs as supplied"；修正后 rerun 列入 future work。修复：orchestrator [-5:] + 强化锚点正则（12 个新单测）。

### 零成本证据（全部写入论文，untabulated）
| 项 | 权威值 | 来源 |
|----|----|----|
| 全预算 PR 扫描 2.5–30% | LLM−阈值logit 峰值 +6pp；月块 CI 全含零 | `budget_sweep.csv` |
| 校准 | Brier LLM 0.221 vs 基线 0.205（≈基准率）；reliability 0.020 / resolution 0.004；ECE 0.113 | `calibration_llm.csv` |
| 平局 | 28 条 top-2 平局；全部改判 reduce → recall 0.206→0.230（CI 内） | 同上 ties 段 |
| Sector 分解 | fire rate Infra 65% vs Industrial 4%；配置 +1.8pp、组内选择 −1.5pp；80% 组内随机重排 ≥ 实际 | `sector_decomposition.csv` |
| 择时 | corr(月 fire rate, 月真实 reduce 率) = −0.31（n=23，描述性） | `timing_agreement.csv` |
| Agreement | 与确定性分歧 ρ=−0.71（诚实摘要）；与正确性 r=0.03（弃权无益）；uniform 兜底 18/575 | 同上 |
| 版本审计 | 2,875 次调用全部由 `deepseek-v4-flash` 单一版本服务 | `model_version_backfill.csv` |

### Hygiene（不改任何已报告数字）
- predictions.csv 回填真实标签（575 全，混淆矩阵与已报告逐字节一致）；postprocess 去重；概率 sum=1 校验器（3,450 向量预检 0 拒绝）
- thinking 路径加失效保护（证据：全部日志 enable_thinking=False）
- 成本重算：575 终跑 $3.97（V3 标价口径；V4-flash 牌价本地不可得，已注明）
- 测试 40 → **61 全绿**（fundamentals agent、prompt-lock SHA、edgar client、8-K recency、Item 1A 提取器）

## 10. 经济价值 / 判别力 / 校准评测轮（2026-07-08，analysis/econ_value_eval.py，seed 42，N_BOOT=10,000）

数据源（全部只读）：audit_log/predictions.csv × outputs/tables/quant_only_test_predictions.csv（575↔575 完美合并，标签一致率 100%）× data/processed/backtest_ready_panel.csv（monthly_rf）。输出：outputs/fundamentals_robustness/{portfolio_value,auc_metrics,calibration_curve}.csv。

### 10.1 组合经济价值（23 个测试月，等权月度再平衡，"规避"设计）
| 组合 | 年化超额 | 年化波动 | Sharpe | ΔSharpe 95% CI vs EW | MaxDD | 单边换手 | 净 Sharpe(25bps) |
|---|---|---|---|---|---|---|---|
| EW 全持有（基准） | 3.16% | 13.62% | 0.232 | — | −9.16% | 0.000 | 0.232 |
| LLM 规避 | 2.97% | 13.15% | 0.226 | [−0.209, +0.161] | −9.76% | 0.173 | 0.147 |
| Logistic 规避（同预算） | 3.83% | 14.61% | 0.262 | [−0.133, +0.160] | −9.99% | 0.202 | 0.179 |
| Oracle 规避（上界） | 31.83% | 7.21% | 4.416 | [+3.06, +6.41] | −1.52% | 0.448 | 4.043 |
| 随机规避（MC 均值，2,000 次） | 3.16% | — | 0.231 | MC 2.5/97.5 分位 [0.052, 0.409] | — | — | — |

- Δ年化均值 (LLM−EW) 95% CI [−2.78%, +2.03%]；ΔCER(γ=3)：LLM ≈ −0.00%，logistic +0.25%。
- Sharpe SE（Lo 2002，iid，23 月）≈ 0.21 —— 样本极短，区间宽是特征不是缺陷。
- 口径：超额 = future_ret_1m − monthly_rf；成本 = 25bps 单边 × 换手 × 12；oracle=剔除全部真实 reduce 名字（构造性上界）。

### 10.2 判别力（阈值无关；月块 bootstrap 10,000 次）
| 模型 | ROC-AUC [95% CI] | PR-AUC [95% CI] |
|---|---|---|
| LLM prob_reduce | 0.504 [0.448, 0.568] | 0.302 [0.208, 0.421] |
| Logistic | 0.508 [0.418, 0.593] | 0.286 [0.183, 0.421] |
| 差 (LLM−logit) | −0.004 [−0.117, +0.116] | +0.016 |
| 无信息基准 | 0.500 | 0.287（基率） |

### 10.3 校准（10 等宽箱）
- ECE：LLM **0.113**，logistic **0.021**。与 §9 Brier（0.221 vs 0.205）互证：logistic 是基率的复述（校准好、无分辨力），LLM 过度自信（可靠性罚 0.020）。
- 曲线数据在 calibration_curve.csv（论文校准图数据源）。

### 10.4 DM–HLN 检验(2026-07-08 追加;analysis/econ_value_eval.py::dm_hln,输出 dm_test.csv)
月度 Brier 损失差(T=23,h=1 HLN 小样本修正,双侧 t(22)):
- LLM vs 常数基率:均差 +0.0165,t=+2.52,**p=0.020** —— LLM 概率显著更差(过度自信而非信息)
- LLM vs logistic:均差 +0.0165,t=+2.34,**p=0.029**
- logistic vs 基率:均差 +0.0001,t=+0.06,p=0.950 —— logistic 即基率复述
论文挂接:§5.2 Brier 句后一句;引用 Diebold–Mariano 1995 + Harvey–Leybourne–Newbold 1997。

### 10.5 对抗性体检修正(2026-07-08):§7 表格映射更新
- 合并轮之后:**Table 7 = 组合经济价值**(portfolio_value.csv / auc_metrics.csv,§10),
  **Table 8 = 合并的理由审计**(Panel A 接地/事实性 + Panel B judge + Panel C 人机 κ);
  Table A1 已随附录移入 THESIS_SUPPLEMENT.md。§7 早期行内的旧映射(Table 7=接地、
  Table 8=质量)、"Figure 5.1"、"§6.6/§7.1"、"Appendix B.5" 等指向均为历史记录,以本节为准:
  架构图=Figure 1(TikZ 原生),市场调整标签=§5.4,聚合稳健性=§4.3 + Supplement B.5。
- 体检更正:actionability 人机分歧"从不超过 1 分"有误——45 例中 1 例相差 2 分
  (spotcheck 原始数据核实);正文与表注已改为 "one of 45"。池化 κ=0.47 为二次加权
  (非未加权);Cohen 1960 引用已移除,文献总数 86→85。

## 11. 第二评分者校准(2026-07-10;analysis/second_rater_agreement.py,输出 second_rater_agreement.csv)
- Rater 2:Kaixiang Zhang,独立盲评(双语书面指南,120 分钟,独立性声明已签);原始提交与
  已填工作表存档于 audit_log/rater2/(rater2_submission_raw_kzhang_2026-07-09.txt、
  rationale_spotcheck_worksheet_rater2_FILLED_kzhang.md);条目顺序 45/45 核对一致。
- Entailment(核心维度):R1–judge κ=0.760(不变),R2–judge κ=0.809,R1–R2 κ=0.554
  (exact 88.9%,within-1 100%);三评分者 Krippendorff α=0.709;均值 1.18/1.11/1.16。
- Relevance:κ 0.23/0.18/0.37,α=0.228(饱和边际);均值 1.78/1.58/1.93。
- Actionability:κ 0.06/0.20/−0.07,α=0.005 —— 人类之间不校准,如实披露;R2 均值(1.58)
  较 R1(1.91)更接近 judge(1.44)。
- 405 个两两分数对比中恰有 1 对相差 >1 分(R1–judge actionability)。
- 论文更新:摘要(two raters, κ 0.76/0.81)、§6.2 段落、Table 8 Panel C 重排为
  κ 矩阵 + α 列、§7 撤销 single-rater 限制。

## 12. v3 修复输入重跑(2026-07-10;llm/run_v3_corrected_feed.py,隔离于 audit_log/v3_corrected_feed/)
- 设计:锁定测试窗全量 575 决策;prompts/SHA 与 v2 逐字节相同;唯一变化 = 输入管道
  (Item 1A 提取器修复 + 8-K 保留最新五份);宏观/价格/基本面三 agent 全部缓存命中,
  仅 Disclosure+Aggregator 新调 API。v2 仍为主报告运行。
- 结果:**v3 recall 0.242**,precision 0.325,触发率 21.4%(123 flags);随机地板=0.214;
  差 +0.029,**月块 95% CI [−0.023, +0.090],REIT 块 [−0.041, +0.092]——均跨零,null 成立**。
- 决策层面:v2↔v3 一致率 74.1%(33 撤旗 / 39 新旗 / 84 共同)——输入实质变化,总体技能不变。
- 接地率(同一近似口径,去重后):v2 109/117=93.2% → v3 116/123=94.3%,不降反微升。
  (论文中 v2 的 88.0% 来自官方审计脚本口径,勿与本近似口径混用。)
- 论文挂接:§3.3 caveat 收尾改指 §5.3;§5.3 新增 corrected-feed 段;§7 撤销
  corrected-feed future-work 从句。四条外审批评至此全部闭合。
