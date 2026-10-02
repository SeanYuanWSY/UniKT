"""Prompts for the diagnosis task (v2: bindable citations, JSON contract)."""

SYSTEM_PROMPT = """你是一名教育数据分析师。你面对一个已训练并冻结的知识追踪（KT）深度模型对一个学生练习记录的内部信号统计。你的任务：
1. 基于且仅基于给定的证据，输出该学生的中文诊断报告；
2. 预测该生在"待预测知识点"上的后续正确率。

规则（必须严格遵守）：
- 只能引用证据中存在的数字，且必须精确（如证据写 0.45 就写 0.45，禁止 45%、0.4 等改写）；
- 每个 claim 必须给出引用（citations），指明 KC 编号和字段名（正确率/模型预测均值/模型就绪度/最近5次正确率/尝试次数）；
- 禁止使用任何外部知识或常识推断（如"研究表明"），证据不足时明确说"证据不足以判断"；
- 只能讨论 E1 表中列出的 KC，不得编造未出现的知识点；
- "模型就绪度"= 冻结模型对该生此刻若遇到该 KC 题目的预测正确概率，它反映模型内部状态，优先以它为掌握度依据；
- 输出必须是单个 JSON 对象，不输出其他文字。

JSON 格式：
{
  "summary": "一段话总体诊断",
  "claims": [
    {"type": "掌握|薄弱|预测|趋势|建议", "statement": "...", "citations": [{"kc": 5, "field": "正确率", "value": 0.45}]}
  ],
  "holdout_predictions": [{"kc": 5, "pred_acc": 0.4}],
  "weakest_kcs": [5, 12, 7]
}
其中 holdout_predictions 必须覆盖"待预测知识点"列表中的每一个 KC；weakest_kcs 是最薄弱的至多 3 个 KC（按薄弱到强排序）。"""

FEWSHOT_USER = """学生 ID: U9（冻结模型: DKT）

E1 已学知识点统计（evidence 段，行 = KC ID）：

| KC | 名称 | 尝试次数 | 正确率 | 模型预测均值 | 模型就绪度 | 最近5次正确率 |
|----|------|---------|--------|-------------|-----------|--------------|
| 3 | 分数加法 | 12 | 0.42 | 0.44 | 0.41 | 0.20 |
| 8 | 分数乘法 | 5 | 0.80 | 0.77 | 0.74 | 0.80 |

E2 模型意外事件（|模型预测 − 实际| 最大的步骤）：
- 步骤11: KC 3(分数加法) 模型预测 0.71，实际错

E3 全局：共 17 步，总体正确率 0.53，模型平均预测 0.55，模型对该生的 AUC 0.62

待预测知识点: 3, 8"""

FEWSHOT_ASSISTANT = """{
  "summary": "该生分数乘法较稳，分数加法薄弱：尝试多但正确率低，最近5次正确率下滑到0.20，且模型在第11步以0.71预测其做对却做错，说明模型此前高估了该生的分数加法状态。",
  "claims": [
    {"type": "薄弱", "statement": "分数加法薄弱，正确率仅0.42且最近5次正确率0.20呈下滑趋势", "citations": [{"kc": 3, "field": "正确率", "value": 0.42}, {"kc": 3, "field": "最近5次正确率", "value": 0.2}]},
    {"type": "预测", "statement": "模型曾高估该生的分数加法：第11步预测0.71但实际做错", "citations": [{"kc": 3, "field": "模型预测均值", "value": 0.44}]},
    {"type": "掌握", "statement": "分数乘法相对掌握，正确率0.80，模型就绪度0.74", "citations": [{"kc": 8, "field": "正确率", "value": 0.8}, {"kc": 8, "field": "模型就绪度", "value": 0.74}]},
    {"type": "建议", "statement": "针对分数加法安排复习（其证据显示正确率低且在下滑）", "citations": [{"kc": 3, "field": "尝试次数", "value": 12}]}
  ],
  "holdout_predictions": [{"kc": 3, "pred_acc": 0.35}, {"kc": 8, "pred_acc": 0.72}],
  "weakest_kcs": [3]
}"""


def build_messages(pack_markdown: str) -> list[dict]:
    """Assemble the chat messages for one student's evidence pack."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
        {"role": "user", "content": pack_markdown},
    ]
