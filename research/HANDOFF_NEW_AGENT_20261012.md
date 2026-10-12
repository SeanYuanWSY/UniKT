# 交接提示词：UniKT LLM-KT 支线自动研究循环（2026-10-12 停机归档版）

> 用户 2026-10-12 17:0x 明示停止："先不整了……总结……归档"。本文取代 HANDOFF_NEW_AGENT_20261011.md 成为最新交接入口；恢复循环时从本文出发。规则体系（红线/自审清单第九节/外盘协议）不变，见 20261011 版。

## 一、停机时点状态（全部实测）

- **循环机制已停**：Mac 心跳 LaunchAgent（com.unikt.loop.heartbeat.plist）已 bootout+删除；wsl_watch.sh 已杀。WSL 侧 daemon 仍在跑但队列已空（最后任务 fusion71v3 已 END 16:59:53），不会再产生新任务——无人投喂即静默。
- **git**：feat/llm-kt-plugin 全部推送至 fork（本文件随停机 commit 一并 push）。
- **外盘**：MANIFEST.sha256 在 b69/70 后 4147 条；停机增量（b71 产物+本文件）见本次备份。
- **无未收割产物**：b71 v3 已裁决完毕（下节），队列/在跑/未收割全部清零。

## 二、2026-10-12 会话成果（b60-b71，12 批）

| 批 | 内容 | 终局 |
|---|---|---|
| b60-b66 | 身份格多种子收割裁决、b61-b66 各批（详见 batch_reports/20261012_batch6*.md） | 已裁决归档 |
| b67 | 深锚审计#1：源 checkpoint 重放头条格（A17 迁移 AKT delta +0.152） | 参照链健康；G1 门规格错（混合 vs same-support 口径未声明）入档 |
| 唤醒机制 | launchd 心跳 v4 + 唤醒 agent 三态实测；一次"停滞"误判后**反转**（实际正常完成） | SELF_SUPERVISION.md（gitignore，仅外盘） |
| b68 | 13 个关闭 idea 复活条件系统扫描 | 4 复活候选（融合/M1/文本难度/embed fallback），9 永久关闭 |
| b69 | M1 转移表复活首测（同题段版） | 关闭维持；实现错位曝光（建表只含同题段=artifact 位置）；零命中归因撤回（erratum） |
| b70 | M1 跨题修正版+命中率字段 | 三域 CI 全跨零（hit 46-92%）→ M1 双实现终审关闭；措辞按 advisory 降格"未检出≠排除" |
| b71 | 融合复测（复活#1）公平口径四架构 | **域分层混合带**：A17 3/12 CI 正（+0.023~+0.027），Junyi 4/4 CI 全负，EdNet 跨零；复活门 3<4 未达，有害门触发；idea3 关闭维持 |

**advisory（外部复盘）本轮全部核实落实**：①max 对照是另一种融合器→b71 改三配对差并报；②b70"边际足够"降格"未检出"；③维度探针按架构分键（v2 只认 DKT 键会全灭 AKT/SAKT/DKVMN）；④生死簿覆盖损伤修复（a 行+3-7 号行恢复，双 M1 合并）。

## 三、新坑档案（环境）

1. **幽灵 run**：DKT_junyi2015_20261004-014541 目录名+metadata 均正常（junyi/seed42）但 checkpoint 40 items（旧预处理）。**目录名+metadata 不能证明 checkpoint 身份**；白名单须维度探针（按架构键：DKT out_layer / AKT q_embed / SAKT exercise_emb / DKVMN k_emb_layer，[0]==n_items：a17=102/ednet=189/junyi={38,39}），跳过原因打印。
2. WSL Tailscale node key 过期会导致 SSH 永久超时（曾停摆两天）；恢复需用户在 admin 后台 reauth。
3. 唤醒 agent 从 launchd 拉起需 stdout 可见环境，8 分钟无输出判停滞的阈值要配合实际产出文件核对再杀。

## 四、恢复循环时的第一动作

1. 读本文 + AUTO_LOOP_HANDBOOK.md + IDEA_PORTFOLIO.md + PAPER_NUMBERS_V2.md。
2. 重建心跳（SELF_SUPERVISION.md v4 机制，plist 骨架在外盘 mac_research_snapshot/）。
3. b72 候选（b68 扫描余量）：复活候选#3 文本难度（低优先）/#4 embed fallback；或装配第三轮（摘要/贡献列表）；或 EdNet 功效深挖（b59/b60/b71 三批都指向 EdNet 显著性支持量 n=56 问题）。
4. 队列文件 = /root/task_queue.txt（守卫 grep 后追加）；训练 `-m` 大写模型名。

## 五、资产位置

- 批次报告：research/batch_reports/20261012_batch60..71.md（12 份）
- 结果 JSON：research/transfer/results/（b69 m1_retry_a1 / b70 m1_retry_cross / b71 fusion71_fair 等）
- 脚本：research/transfer/（m1_retry_a1.py / m1_retry_cross.py / fusion71_fair.py / audit67_headline.py）
- 外盘：/Volumes/Extreme Pro/…LLM-KT支线_20261003/（checkpoints/transfer_results/mac_research_snapshot + MANIFEST.sha256）
