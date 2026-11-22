# DraftSlice 生产架构（实验收敛稿）

Status: 2026-11-22. 本文是**唯一的生产架构基线**，替换此前的 v1 评审稿。
实验驱动、原始 JSON 和临时实现计划放在 `runs/`（gitignored），不进入正式
文档树。下一任务的实现计划：`runs/PRODUCTION_IMPLEMENTATION_PLAN_20261122.md`。

结论先行：

> **Built-in 是默认。DraftSlice 是一个必须按
> `(model, engine, workload, batch bucket, k)` 逐格证明收益的 profile。**
> “structured 一律切片”在 Qwen3.5-4B/B8 是实测回归（混合负载 0.767×，p99
> +36%），不能作为生产默认。

---

## 1. 失败模式留档

后续查这些结论即可，不要重复踩。原始产物在 `runs/`。

| 模式 | 实测 | 决定 | 产物 |
|---|---|---|---|
| structured → DraftSlice 默认（4B/B8 混合负载） | 0.767× 吞吐，p99 1008 ms vs 741 ms | **拒绝**；默认改 built-in | `runs/routing/results/baseline_0.5-0.3-0.2.json` |
| strict slice per-case（4B/B8，k=4/8） | chat 0.84–0.91×，math 0.84–0.91×，tool 0.85–0.92×，JSON 0.66–0.77×，regex 0.82–0.87× | **拒绝 B=8** | `runs/prod_routing_4b_k{4,8}_full_20261122.json` |
| `MAX_ROWS=512` per-row 大小门控 | JSON 仍 0.63–0.70×，regex 0.79–0.85×，router 中位 0.951× | **无效**；问题不是 valid-set 大小 | `runs/prod_routing_4b_k8_gate512_20261122.json` |
| 2-pool gateway + DraftSlice k=8 结构化池（4B） | 0.907×，p99 1103 ms | **拒绝**；池化本身没问题，profile 负收益 | `runs/routing/results/gateway_0.5-0.3-0.2.json` |
| per-workload k 路由（free k=8 / structured k=4，4B） | 均衡 mix +3.7% 但 p99 +10.9%；tool-heavy 0.957×，p99 +35.6% | **拒绝**；固定池在偏斜 mix 下输 | `runs/routing/results/gateway_mtp_k4_*.json` |
| `\|V\|=1` deterministic injection | 308 次 draft call 中 0 次命中 | **不发布**（研究对照保留） | closeout 记录 |
| static free-form screener 16k | 0.49×，tau 2.60→1.34 | **拒绝**；calibrated 只用于内存/覆盖，不作为 latency 路径 | closeout 记录 |
| DFlash all-position “1.28×” | 同轮 DFast vs stock DFlash 1.24×；但 vs built-in MTP：B1 +10.5%、B8 −1.6%；vs DraftSlice MTP B1 −10.7% | **4B tool 不启用 DFlash**；MTP slice k=8 更好 | `runs/prod_dflash_ds_20261122.json`、`runs/prod_dflash_mtp_20261122.json` |
| 4B B=1 tool，k=4 strict slice | 均值 1.033×、中位 0.998×；k=8 是 1.286×/1.242× | **k=4 不启用**；用 k=8 | `runs/prod_routing_4b_b1_k4_20261122.json` |
| 35B-A3B TP=2 gateway（free MTP + tool DraftSlice） | DraftSlice 池 4 个结构化请求后挂住；`No available shared memory broadcast block` 60s；baseline built-in 199.6/191.5 tok/s/GPU 正常 | **35B-A3B 保持 built-in 单 profile；TP>1 MoE 的 DraftSlice 是 blocker** | `runs/routing/results_35b/GATEWAY_TP2_HANG.md` |

### 1.1 已确认的正收益 cell

| model | workload | batch | action | vs best built-in | 状态 |
|---|---|---|---:|---:|---|
| Qwen3.5-4B | tool_call | 1 | `draftslice_mtp` k=8 | **1.24×**（均值 1.286×） | **启用** |
| Qwen3.5-4B | free | 1 | built-in MTP | 1.00 | built-in |
| Qwen3.5-4B | json/regex | 1 | built-in MTP | 1.00 | built-in |
| Qwen3.5-4B | all | 8 | built-in MTP | 1.00 | built-in |
| Qwen3.5-35B-A3B | all | 8/32 | built-in MTP TP=2 | 1.00 | built-in；等 TP hang 修复 |
| Qwen3.5-4B + DFlash | tool | 1/8 | — | B1 +10.5%、B8 −1.6% vs MTP | 不启用 |

---

## 2. 实验证据

环境：Qwen3.5-4B, vLLM 0.34, eager, TP=1, greedy, A100-80GB, SM clock
1410 MHz。同轮 per-case；混合负载 24 并发、40 s、2 GPU 同预算。

### 2.1 Per-case（B=8，k=8）

| domain | mtp | slice | router | slice/mtp | router/mtp |
|---|---:|---:|---:|---:|---:|
| chat | 963 | 812 | 904 | 0.843 | 0.938 |
| math | 700 | 587 | 658 | 0.839 | 0.940 |
| tool_call | 545 | 500 | 491 | 0.918 | 0.901 |
| json_schema | 789 | 523 | 508 | 0.663 | 0.644 |
| regex | 270 | 221 | 221 | 0.817 | 0.817 |

k=4：chat 0.911×、math 0.913×、tool 0.852×、JSON 0.765×、regex 0.869×
（均 slice/mtp）。`MAX_ROWS=512` 未改变结论。

### 2.2 混合负载拓扑（同 2 GPU）

| 方案 | tok/s/GPU | p99 | 相对最佳单配置 |
|---|---:|---:|---:|
| 2× built-in MTP k=8，round-robin | 1030.6 | 741 ms | 1.000 |
| 2× built-in MTP k=8，按 class 分池 | **1053.5** | **709 ms** | **1.022** |
| 免费池 MTP + 结构化池 DraftSlice k=8 | 934.4 | 1103 ms | 0.907 |
| 2× DraftSlice k=8 + router，round-robin | 790.7 | 1008 ms | **0.767** |

同臂第二次 `baseline_mtp` 为 986.1 tok/s/GPU（p99 781 ms），host drift 1–5%。
跨轮只做方向性判断，决定只用同轮 paired ratio。

### 2.3 Per-workload k 路由（4B，B=8）

| mix（free/tool/json） | gateway k 路由 | 最佳单配置 | 吞吐比 | p99 比 |
|---|---:|---:|---:|---:|
| 0.5/0.3/0.2 | 1022.9 | 986.1 | 1.037 | 1.109 |
| 0.2/0.7/0.1 | 885.1 | 924.7 | 0.957 | 1.356 |

结论：4B 不建 gateway；固定池在偏斜 mix 下会输。

### 2.4 解释

B=8 时 4B 的 draft-head share 很小，full-head GEMM 已经高效；slice 的
grammar/shadow、row gather/scatter、Python 决策带来固定开销，行数节省抵不过。
混合 batch 下，free 行仍要付 DraftSlice bookkeeping，structured 行又更慢，
所以 `router` 从 per-case 的 0.90 掉到 0.767。池化本身（同 built-in profile）
是中性偏正，说明 gateway 拓扑不是问题，**profile 选择才是**。

### 2.5 4B B=1 与 DFlash（同轮）

| 场景 | 最优动作 | vs best built-in | 备注 |
|---|---|---:|---|
| 4B B=1 tool | `draftslice_mtp` k=8 | **1.24×**（均值 1.286×） | 111.6 vs 90.2 tok/s；tau 7.67 vs 2.67–3.6 |
| 4B B=1 free | built-in MTP | 1.00 | slice 0.906×；router 0.992× |
| 4B B=1 json/regex | built-in MTP | 1.00 | slice 0.92–0.95× |
| 4B B=1 tool，k=4 | 不启用 | 1.03× 均值 / 0.998× 中位 | k=8 更好 |
| 4B B=1/8 tool，DFast | 不启用 | B1 +10.5%、B8 −1.6% vs MTP；B1 比 MTP slice 低 10.7% | 只赢 stock DFlash（1.24×） |

4B 的 profile 表当前只需一条：`(4B, vllm, tool_call, batch=1, k=8,
strict_slice)`。其余全部 built-in。35B-A3B 暂不启用任何 DraftSlice cell。

---

## 3. 架构决策

### D1 — Built-in 是默认

每个部署默认使用引擎最佳 built-in speculative method + full head。
DraftSlice 只有在匹配且 `enabled=true` 的 `ProfitCell` 上才执行。

### D2 — Profitability table 是生产契约

生产开关不是全局 `DRAFTSLICE_VLLM_SLICE=1`，而是内容哈希的 profile table：

```python
@dataclass(frozen=True, slots=True)
class ProfitCell:
    model: str
    engine: str                 # vllm | sglang
    method: str                 # mtp | eagle3 | dflash | draftslice_mtp | draftslice_dflash
    workload: str               # free | tool_call | json_schema | regex | ...
    batch_bucket: str           # 1 | 2-8 | 9-32 | 33-64 | >64
    k: int
    tp: int
    quant: str                  # bf16 | fp8 | int8
    graph: str                  # eager | full_cuda_graph
    valid_bucket: str           # 0-64 | 65-512 | 513-4096 | >4096
    action: str                 # builtin | strict_slice | mask_allpos | mask
    expected_gain: float        # vs best built-in in the same cell
    measured_p99_ratio: float
    manifest_id: str
    enabled: bool
```

- miss / disabled / engine mismatch → built-in full head；没有“最近 cell”兜底。
- `profiles/1` artifact 带 hash；runtime 校验，不匹配即硬错误。
- `DRAFTSLICE_PROFILE_MODE=off|observe|enforce`；生产先 `observe`。
- 引擎、模型 revision、tokenizer、量化、TP 变化后必须重跑 gate。

### D3 — 先单 profile；gateway 只在“k/method 冲突且收益扛得住池化”时上

单 profile 部署：一个模型一个引擎，adapter 按 batch/row 查表。若所有 enabled
cell 共享一个 `(method, k)`，不需要 gateway。

需要 gateway 的条件：enabled cells 在 **k** 或 **method family** 上冲突，且
同轮实验证明收益 > 池化损失。4B 已证伪；35B-A3B 是待验证候选（E1）。

### D4 — 完整接入 plan/cost/scheduler/union，但默认 full

- `PlanCompiler` 只为 enabled cell 生成 slice plan；其他行是 full node。
- `PlanCache`/`CandidateSetCache` 接引擎热路径，compile time/hit rate 是一等指标。
- `Scheduler` 用 profile table + 校准后的 cost model 选 full/slice；模型未校准时
  默认 full。
- `ProductionExecutor` 支持一个 batch 内混合 full/slice 节点；strict coverage
  在执行前检查。
- Union/tree 只用于 enabled tree cell；DFlash block 仍按位置串行。

### D5 — SGLang EAGLE/DFlash 挂同一张表

SGLang MTP step-0 已验证；EAGLE/DFlash hook 下一步补，但默认 built-in，只有
profile cell 命中才启用。

### D6 — 门禁只对比同 cell 最佳 built-in

禁止用“over stock DFlash”替代“over best built-in MTP”。cell 必须：
validity 1.0、greedy 输出 equal、吞吐 ≥1.05×、p99 ITL ≤1.05×、fallback ≤5%、
strict coverage 1.0；任一失败即 `enabled=false`。

### D7 — 过时文档像代码一样替换

过时的直接改，不新增平行文档；失败模式进第 1 节表格。当前修正范围：
`architecture.md` §0/§18.8、`competitive-plan.md` §3/§9、`benchmark.md` B4
段。`runs/RESEARCH_CLOSEOUT_20261122.md` 是历史记录，不作为当前状态引用。

---

## 4. 目标架构

```text
request / batch
      |
      v
┌──────────────── profile resolver ────────────────┐
│ profiles/1 (hashed)                              │
│ exact cell match; miss/disabled -> built-in full │
└──────────────────────┬───────────────────────────┘
                       v
┌──────────────── engine adapter ──────────────────┐
│ vLLM / SGLang grammar bridge + worker shadow     │
│ per-batch / per-row decision; fail-closed gates  │
└──────────────────────┬───────────────────────────┘
                       v
┌──────────────── plan / cost / scheduler ─────────┐
│ full nodes + enabled slice nodes                 │
│ plan/candidate cache; default full               │
└──────────────────────┬───────────────────────────┘
                       v
┌──────────────── executor / kernels ──────────────┐
│ full | strict slice | mask | mask_allpos | union │
└──────────────────────┬───────────────────────────┘
                       v
┌──────────────── contract / stats / gates ────────┐
│ DraftContract + StepTrace; JSONL / Prometheus    │
│ per-cell gate; failure disables the cell         │
└──────────────────────────────────────────────────┘
```

决策顺序（必须写成代码）：

1. 无 grammar / free-form → built-in（全 free batch 走 stock）。
2. profile miss / disabled / engine mismatch → built-in full。
3. 能力门禁失败（TP shard 不可证明、quant layout 不支持、FULL graph step≥1、
   非 xgrammar、temperature>0、local-argmax 不支持）→ full + reason。
4. grammar state 缺失/空/超阈值 → full + reason。
5. cell enabled 且 batch/valid bucket 命中 → slice/mask。
6. 任何异常 → full + reason；永不 fail open。

---

## 5. 生产默认

```toml
[server]
method = "builtin"                 # 不是 draftslice
profile_mode = "observe"           # off | observe | enforce
profile_table = "profiles/4b.json"

[adapter.vllm]
slice = false                      # 只由 enabled cell 打开
router = true                      # 全 free batch 走 stock
max_rows = 32768
full_mask = false
screener = ""

[safety]
tp = "auto"
quant = "auto"
full_graph = "step0"
fail_closed = true

[observability]
trace_sample = 0.01
jsonl = "/var/log/draftslice/contracts.jsonl"
prometheus = "engine"
alarm_fallback_rate = 0.05
```

4B 当前 profile table 不启用任何 B=8 cell；先 `observe`。B=1 tool 是否启用，
等 E2。

---

## 6. 拓扑

| 拓扑 | 使用条件 | 当前状态 |
|---|---|---|
| T1 单 profile | enabled cells 共享 `(method,k)`；默认 | 4B 最佳：2× built-in MTP |
| T2 profile gateway | k/method 冲突且收益扛得住池化 | 4B 已证伪；35B-A3B 待 E1 |
| T3 多模型 gateway | 多模型/多 profile 统一入口 | 可选；必须有逐 profile 收益 |

Gateway 若建：每池固定 `(engine,method,k,tp,quant,graph)`；精确 cell 路由；
按 outstanding/queue 做容量感知，但不做请求中途迁移、不在首 token 后重试；
overflow 只去 capability 兼容的 profile，并记录 degraded action；必须比同 GPU
预算的最佳单部署 ≥1.05× 吞吐或 ≥20% p99 改善。

---

## 7. 可观测性

`DraftContract` 保留；`ServingPoint.head_share` 必须拆成：

- `row_share = candidates / (rows * vocab)`（当前值）；
- `measured_head_share` — B4 实测 draft-head time / step time；
- `profile_mode`、`cell_id`、`cell_hit`、`action`、`fallback_reason`。

`StepTrace` 记录：identity（engine/model/tp/quant/graph/workload/config hash）、
decision（mode/source/k/cell/reason/candidates/rows/row_share/head_share）、
timing（compile/kernel/step）、cache（plan/candidate hit）、outcome（accepted/
tau/validity/fallback）。Sinks：进程内 registry、JSONL 采样、Prometheus
（引擎原生优先）。

---

## 8. 实验状态（本轮已完成）

| ID | 实验 | 结果 | 决定 |
|---|---|---|---|
| E1 | 35B-A3B TP=2：best single vs free-MTP + tool-DraftSlice gateway | baseline built-in 199.6/191.5 tok/s/GPU；DraftSlice TP=2 池 4 请求后 hang | **35B-A3B 保持 built-in 单 profile**；gateway 阻塞 |
| E2 | 4B B=1 全矩阵（k=4/8） | k=8 tool slice 1.24×（均值 1.286×）；k=4 tool 1.03×/中位 0.998×；free/json/regex <1 | **启用 4B B=1 tool k=8 cell**；其余 built-in |
| E3 | DFlash all-position vs built-in MTP，B=1/8 tool | DFast vs DFlash 1.24×；vs MTP B1 +10.5%、B8 −1.6%；vs MTP slice B1 −10.7% | **4B tool 不启用 DFlash** |

下一任务优先实验：

1. **TP>1 MoE hang 复现与定位**：单 `draftslice_mtp` TP=2 server + 单条 JSON 请求；
   `py-spy dump` engine/workers；再用 `DRAFTSLICE_VLLM_SLICE=0` pass-through 对照。
   重点检查 rank-local `full_logits` 是否在 TP all-gather 中造成死锁。
2. **SGLang EAGLE/DFlash**：补 hook + B4 runner；默认 built-in。
3. **Cost model MAPE**：B2 held-out；未校准前 scheduler 默认 full。
4. **B=1 tool cell 的 p99/尾延迟**：当前只测吞吐；生产启用前补 p99 与多 seed。

历史跨轮线索（仅方向性，不作为启用依据）：35B-A3B B=8 tool slice k=8 311 vs
MTP k=4 240（1.29×）；B=32 tool slice k=1 730 vs MTP k=4 585（1.25×）。

---

## 9. 风险与不做

- 不把“structured 一律 slice”带进生产；4B B=8 是 0.767×。
- 不用“over stock DFlash”替代“over best built-in”的对照。
- 不在固定池/偏斜 mix 下假设 gateway 会更好；4B k 路由 tool-heavy 是
  0.957×、p99 +36%。
- **TP>1 MoE 不能宣称可用**：35B-A3B `draftslice_mtp` TP=2 在结构化请求上
  hang；built-in MTP TP=2 正常。修复前 35B-A3B 保持 built-in。
- 4B B=1 tool cell 只按 `(workload=tool, B=1, k=8)` 启用；不要把它推广到 B=8
  或 JSON/regex。启用前补 p99 和多 seed。
- 不把 `row_share` 写成 `head_share`；B=8 head share 很小。
- 不做 learned per-cell controller；1–5 reps 会过拟合。
- 引擎升级后 profile table 失效，必须重跑 gate。

---

## 10. 文档地图

| 文档 | 角色 |
|---|---|
| `docs/production-architecture.md` | 本文；生产唯一基线 |
| `docs/architecture.md` | 组件/里程碑参考；§0/§18.8 已按当前实现修正 |
| `docs/benchmark.md` | B0–B5 指标与协议；B4 对照 best built-in |
| `docs/competitive-plan.md` | 竞争/差距；生产状态以本文为准 |
| `docs/design.md` | 公开问题与 scope |
| `docs/environment.md` | 三套环境与锁 |
| `runs/` | 实验驱动、原始 JSON、临时实现计划；不进入正式文档树 |
