# main_utils.py
from typing import Optional

MODEL_SENTIMENT_WORDS = ["积极", "消极", "中性"]

def strip_cot(resp: str) -> str:
    if "Thinking..." in resp and "...done thinking." in resp:
        before = resp.split("Thinking...", 1)[0]
        after  = resp.split("...done thinking.")[-1]
        resp = before + after
    return resp.strip()

def extract_label(resp: str, true_word="真实", false_word="虚假") -> int:
    if true_word in resp:  return 1
    if false_word in resp: return 0
    return -1

def extract_sentiment(resp: str) -> Optional[str]:
    for w in MODEL_SENTIMENT_WORDS:
        if w in resp:
            return w
    return None