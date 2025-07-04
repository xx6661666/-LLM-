# prompt_templates.py
def build_sentiment_prompt(text: str) -> str:
    """
    让模型仅返回“积极”“消极”或“中性”三选一，不要思考过程
    """
    return (
        "请判断下面这条新闻的情感倾向，只能回答：积极、消极或中性。\n"
        "不要输出任何解释、推理或其他文字，只回答这三个词之一：\n"
        f"{text}"
    )

def build_truth_prompt(text: str) -> str:
    return (
        "请判断下面新闻的真实性，只回答：真实 或 虚假"
        "不要输出任何解释、推理或其他文字，只回答这两个词之一：\n"
        f"{text}"
    )

def build_combined_prompt(text, sentiment):
    return (
        f"以下是一条新闻内容，以及其表达的情感信息。\n"
        f"新闻内容：{text}\n"
        f"情感倾向：{sentiment}\n\n"
        f"请你判断这条新闻是否真实，仅根据其语义与表达方式来推测，不要因为情绪强烈就直接认为其为假新闻。\n"
        f"你的回答应该是：真实 或 虚假。"
        "不要输出任何解释、推理或其他文字，只回答这两个词之一"

    )