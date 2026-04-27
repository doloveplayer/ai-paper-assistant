"""LLM-as-a-Judge 生成质量评估。

三个专项 Judge：
1. Faithfulness — 回答是否忠实于检索到的上下文（防幻觉）
2. Answer Relevance — 回答是否直接回应了用户的问题
3. Factual Correctness — 回答中的事实是否与 ground truth 一致

复用已有的 vLLM 文本服务 (:8000)，使用 temperature=0.0 确保确定性。
"""

from __future__ import annotations

import json
import re
import requests
from dataclasses import dataclass, field

from eval.eval_config import config

# ============================================================
# 数据结构
# ============================================================


@dataclass
class JudgeResult:
    """Judge 评估结果。"""

    metric_name: str
    score: float  # 1-5
    reasoning: str
    details: dict = field(default_factory=dict)
    raw_response: str = ""

    @property
    def passed(self) -> bool:
        """根据配置阈值判断是否通过。"""
        thresholds = {
            "faithfulness": config.thresholds.faithfulness,
            "answer_relevance": config.thresholds.answer_relevance,
            "factual_correctness": config.thresholds.factual_correctness,
        }
        threshold = thresholds.get(self.metric_name, 3.0)
        return self.score >= threshold

# ============================================================
# LLM 调用基类
# ============================================================


def _call_judge_llm(prompt: str) -> dict:
    """调用本地 vLLM 进行 Judge 推理，返回解析后的 JSON dict。"""
    try:
        resp = requests.post(
            f"{config.judge_api_base}/chat/completions",
            json={
                "model": config.judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config.judge_max_tokens,
                "temperature": config.judge_temperature,
            },
            timeout=config.judge_timeout,
        )
        if resp.status_code != 200:
            return {"error": f"vLLM returned {resp.status_code}", "raw": resp.text[:200]}

        content = resp.json()["choices"][0]["message"]["content"]

        # 尝试从回复中提取 JSON 块
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json.loads(json_match.group())
        return {"error": "No JSON found in response", "raw": content[:500]}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse failed: {e}", "raw": content[:500] if 'content' in dir() else ""}
    except Exception as e:
        return {"error": str(e), "raw": ""}

# ============================================================
# 1. Faithfulness Judge（忠实度）
# ============================================================


FAITHFULNESS_PROMPT = """你是一个严格的学术评估专家。你的任务是判断一个 AI 助手的回答是否**忠实于**给定的检索上下文。

## 评估标准（1-5 分）
- **1 分**：回答中几乎所有事实都无法在上下文中找到依据（严重幻觉）
- **2 分**：回答中有多个重要事实在上下文中找不到依据
- **3 分**：回答中大部分事实有依据，但有 1-2 处轻微偏差或无法验证
- **4 分**：回答中几乎所有事实都有依据，仅有极微小的无关细节偏差
- **5 分**：回答中的每一个事实性陈述都能在上下文中找到明确依据

## 用户问题
{query}

## 检索到的上下文（论文页面文本）
{context}

## AI 助手的回答
{answer}

## 输出格式
请严格输出 JSON（不要有其他内容）：
{{"score": <1-5 整数>, "reasoning": "<用中文写出评分理由，包括具体哪些声明有/无依据>", "claims_verified": [{{"claim": "<回答中的事实声明>", "supported": true/false, "evidence": "<上下文中的依据或说明无依据>"}}]}}
"""


class FaithfulnessJudge:
    """忠实度评估（防幻觉检测）。"""

    def __init__(self):
        self.metric_name = "faithfulness"

    def evaluate(self, query: str, answer: str, context: str) -> JudgeResult:
        prompt = FAITHFULNESS_PROMPT.format(query=query, context=context[:5000], answer=answer[:3000])
        raw = _call_judge_llm(prompt)

        if "error" in raw:
            return JudgeResult(
                metric_name=self.metric_name,
                score=0.0,
                reasoning=f"Judge 调用失败: {raw.get('error')}",
                raw_response=raw.get("raw", ""),
            )

        score = float(raw.get("score", 0))
        score = max(1.0, min(5.0, score))

        return JudgeResult(
            metric_name=self.metric_name,
            score=score,
            reasoning=raw.get("reasoning", ""),
            details={"claims_verified": raw.get("claims_verified", [])},
            raw_response=json.dumps(raw, ensure_ascii=False),
        )

# ============================================================
# 2. Answer Relevance Judge（回答相关性）
# ============================================================


ANSWER_RELEVANCE_PROMPT = """你是一个严格的学术评估专家。你的任务是判断 AI 助手的回答是否**直接、完整地回应**了用户的问题。

## 评估标准（1-5 分）
- **1 分**：回答完全偏离问题，或回答的是另一个问题
- **2 分**：回答涉及了问题相关的领域，但未直接回答问题核心
- **3 分**：回答了问题的部分内容，但遗漏了重要方面或包含大量无关内容
- **4 分**：回答了问题的主要内容，仅有少量遗漏或轻微偏题
- **5 分**：精准、完整地回应了问题的所有核心要点，无任何无关内容

## 用户问题
{query}

## AI 助手的回答
{answer}

## 输出格式
请严格输出 JSON（不要有其他内容）：
{{"score": <1-5 整数>, "reasoning": "<用中文写出评分理由>", "missing_aspects": ["<问题中被遗漏的要点>"], "irrelevant_parts": ["<回答中与问题无关的部分>"]}}
"""


class AnswerRelevanceJudge:
    """回答相关性评估。"""

    def __init__(self):
        self.metric_name = "answer_relevance"

    def evaluate(self, query: str, answer: str) -> JudgeResult:
        prompt = ANSWER_RELEVANCE_PROMPT.format(query=query, answer=answer[:3000])
        raw = _call_judge_llm(prompt)

        if "error" in raw:
            return JudgeResult(
                metric_name=self.metric_name,
                score=0.0,
                reasoning=f"Judge 调用失败: {raw.get('error')}",
                raw_response=raw.get("raw", ""),
            )

        score = float(raw.get("score", 0))
        score = max(1.0, min(5.0, score))

        return JudgeResult(
            metric_name=self.metric_name,
            score=score,
            reasoning=raw.get("reasoning", ""),
            details={
                "missing_aspects": raw.get("missing_aspects", []),
                "irrelevant_parts": raw.get("irrelevant_parts", []),
            },
            raw_response=json.dumps(raw, ensure_ascii=False),
        )

# ============================================================
# 3. Factual Correctness Judge（事实正确性）
# ============================================================


FACTUAL_CORRECTNESS_PROMPT = """你是一个严格的学术评估专家。你的任务是判断 AI 助手的回答中的关键事实是否与**标准答案事实**一致。

## 评估标准（1-5 分）
- **1 分**：回答中的事实与标准答案严重矛盾，或几乎全部错误
- **2 分**：回答中有多个关键事实错误或与标准答案矛盾
- **3 分**：回答了部分正确事实，但遗漏了多个关键事实或有部分偏差
- **4 分**：回答覆盖了大部分关键事实，仅有少量遗漏或轻微不精确
- **5 分**：回答覆盖了所有关键事实，且表述精确，与标准答案完全一致

## 标准答案事实（Ground Truth）
{gt_facts}

## AI 助手的回答
{answer}

## 输出格式
请严格输出 JSON（不要有其他内容）：
{{"score": <1-5 整数>, "reasoning": "<用中文写出评分理由>", "matched_facts": ["<回答中与标准答案匹配的事实>"], "missed_facts": ["<标准答案中被遗漏的事实>"], "contradicted_facts": ["<回答中与标准答案矛盾的事实>"]}}
"""


class FactualCorrectnessJudge:
    """事实正确性评估（需要 ground truth 事实列表）。"""

    def __init__(self):
        self.metric_name = "factual_correctness"

    def evaluate(self, answer: str, gt_facts: list[str]) -> JudgeResult:
        gt_text = "\n".join(f"- {f}" for f in gt_facts)
        prompt = FACTUAL_CORRECTNESS_PROMPT.format(gt_facts=gt_text, answer=answer[:3000])
        raw = _call_judge_llm(prompt)

        if "error" in raw:
            return JudgeResult(
                metric_name=self.metric_name,
                score=0.0,
                reasoning=f"Judge 调用失败: {raw.get('error')}",
                raw_response=raw.get("raw", ""),
            )

        score = float(raw.get("score", 0))
        score = max(1.0, min(5.0, score))

        return JudgeResult(
            metric_name=self.metric_name,
            score=score,
            reasoning=raw.get("reasoning", ""),
            details={
                "matched_facts": raw.get("matched_facts", []),
                "missed_facts": raw.get("missed_facts", []),
                "contradicted_facts": raw.get("contradicted_facts", []),
            },
            raw_response=json.dumps(raw, ensure_ascii=False),
        )
