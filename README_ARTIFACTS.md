# 数据与结果产物说明(Data & Results Artifacts)

本目录树由导师要求整理:代码 + 全部数据 + 全部结果,便于结合代码分析 resolve=0 的原因。

## 目录结构

| 路径 | 内容 | 大小 |
| --- | --- | --- |
| `results/official/` | 4 次官方 SWE-bench harness 评测结果 JSON(submitted/completed/resolved) | 约 110K |
| `results/predictions/` | 各模型生成的预测补丁(输入 harness 的原始文件) | 约 5.4M |
| `results/logs/` | 训练与评测完整日志(base_sft/nemotron_sft/patch_sft/harness/pipeline) | 约 1.3M |
| `results/gpt_verdicts_case.json` | GPT 轨迹判例标注样例 | 3.6K |
| `dataset/swe_verify/` | 自建 500 实例评测集 full/test/train/easy/medium/hard/rl + deepseek_traces + dpo_pairs | 约 49M |
| `dataset/nemotron/` | 轨迹训练数据(best_sft / deepseek_sft / deepseek_sft_chunks / nemotron_labeled / deepseek_labels) | gzip 压缩 |
| `dataset/patch_sft/` | 补丁桥接训练数据(250 条 issue→gold patch) | 832K |
| `dataset/swe_all_train.jsonl.gz` | 合并后的训练语料 | gzip 压缩 |

## 未上传的大文件(服务器本地路径)

- **模型 checkpoint(共 127G)**:`/home/wcx/swe/checkpoints/{coder-3b-base-ds, coder-3b-base-patch, coder-3b-base-tree, coder-3b-instruct-patch, coder-3b-nemotron}`
- **基座模型(5.8G)**:`/home/wcx/swe/models/Qwen2.5-Coder-3B`
- **官方 SWE-bench 原始数据集副本**:`/home/wcx/swe/data/swe-bench_train.jsonl`(126M)、`swe-bench_test.jsonl`(276M)——公开数据,可按需从官方仓库重新下载

## 关键结论速览(详见 README.md)

- 全部 4 轮官方评测 resolved = 0 / 250
- 轨迹 SFT 在 3B 基座上未生效(loss 稀释,模型未学会 `<tool_call>` 格式)
- 补丁桥接 SFT 学会了格式(instruct 38/250 个补丁可 apply)但未学会修复内容
- 结论:3B 容量天花板,建议换 7B / 注入上下文 / 扩数据

## dataset/swe_verify_v2 (全量标注轮, 2026-09-23)

导师要求的全量标注产物 (Qwen3.5-4B 在 500 实例上的 rollout):

| 路径 | 内容 |
| --- | --- |
| `labels_full/` | 1143 份 rollout 标注 JSON (instance_id/sample/outcome/verdict_raw) |
| `sft_clean_rendered.jsonl` | 标注后构建的 1107 条干净训练轨迹 (masked SFT 训练输入) |
| `rollouts_full.tar.gz` | 1143 条原始 rollout 打包 (标注输入, 可复现) |
| `split_train.jsonl` / `split_test.jsonl` | 400/100 按 repo+difficulty 分层划分 (seed 42) |
| `label_rollouts_v2.py` | 标注脚本 (判据 a-e + GOLD-MATCH/REVERSE/NEUTRAL) |
| `build_sft_from_labels.py` | 干净轨迹构建脚本 |
| `rejudge_outcomes.py` | 测试结果补判脚本 |

## v2 评测结果 (2026-09-24)

- test 100 实例 (Qwen3.5-4B + vl20-full LoRA): **14 PASS / 36 FAIL / 50 unknown** -> resolve 14%
- 历史对比: 3B 基座全方案 250 实例 0-1 个 (0.4%)
- unknown 为镜像测试名漂移 (F2P 方法名不匹配), 与模型无关
- 模型权重通过 Git LFS 存储 (adapter_model.safetensors)

## v2 transition-schema pipeline (2026-09-26)

导师要求的新数据格式: SFT 与 RL 共用 transition schema
(state/action/next_state/trajectory_success/keep/reason), 并修复 3 个硬问题:

1. 严格只保留成功轨迹 (outcome == pass, 174 条 / 59 实例)
2. 无伪边: 不再删除消息后拼接; 改用容器重放验证
3. keep 标签 = "删除该步后仍能 PASS" (贪心重放修剪)

| 文件 | 内容 |
| --- | --- |
| `data/build_transitions.py` | transition schema 构建 (pass-only) |
| `data/trajectory_prune.py` | 重放验证修剪 (keep/reason/diff) |
| `dataset/swe_verify_v2/transitions_v1_final.jsonl` | 2780 条 transition 含 keep 标签 (RL 用) |
| `dataset/swe_verify_v2/transitions_v1_kept.jsonl` | 839 条 keep=1 (30.2%, 压缩 SFT 用) |
| `dataset/swe_verify_v2/prune_v1.jsonl` | 修剪原始结果 (per-step keep/reason/diff) |
