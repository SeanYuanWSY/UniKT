# 已发表管线 explode 泄漏审计（agent 一手代码+数据实测，2026-10-06）

## 总判定：「已发表工作受影响」成立——可点名 DKT(Piech 2015)、AKT(KDD 2020 官方数据 100% 指纹实测)、pyKT(NeurIPS 2022 D&B) KC-level one-by-one 协议（论文自认 leaky 并量化 +8.38 AUC）。

## 关键一手证据（可复现）
- pyKT：split_datasets.py extend_multi_concepts（explode 机制）+ 论文 2206.11460 表（AS2009 one-by-one vs all-in-one：DKT 0.7419→0.8262、AKT 0.7650→0.8493）+ EdNet 预处理确实展开（ednet_preprocess.py:53）。
- AKT 官方 assist2009 数据实测：32,608 相邻同题对，100% 同标签、99.96% 异 concept——explode 指纹精确匹配（16% 行占比与 pyKT 论文 1.197 KC/题交叉吻合）。
- DKT 原始 Lua：n_input = n_skills*2，(skill, correct) 输入——第一天就是 teacher forcing。

## 我们的增量（区分度）
1. pyKT 只在评测侧区分协议、训练仍 explode，且泄漏列只覆盖 AS2009（+0.08）——**EdNet 量化（0.854→0.460，−0.39）是新的且幅度远大**。
2. A1-only 三层口径体系 + 免疫定理 + 残余泄漏上界（0.45-0.51）+ KC 纯度拆分（71/29）——协议工具箱完整度超 pyKT 的二分法。
3. 迁移反预测（0.247）与"干净口径下查表 vs 部分本域深度显著胜"是 pyKT 没有的部署发现。

## 边界（防反噬，写 limitation）
assist2017/2015、SAINT（question-level）、pyKT 主表（question-level/all-in-one）不受影响；"EdNet 已发表 AUC 被膨胀"只能证机制成立（pyKT 展开 EdNet），无官方对照数字——如需补齐可跑 pyKT EdNet 双协议复现。

## 论文引用策略
- 稻草人质疑的直接回应 = pyKT 论文自己的泄漏量化表。
- AKT 官方数据指纹检测可做成论文附录的可复现脚本（一次性统计，我们已有类似工具）。
