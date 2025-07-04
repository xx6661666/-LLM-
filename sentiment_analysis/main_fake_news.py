# main_fake_news.py  （调试版，可整体覆盖原文件）
from data_loader import load_news_from_tsv
from prompt_templates import build_truth_prompt, build_sentiment_prompt, build_combined_prompt
from ollama_utils import query_ollama
from evaluation_metrics import compute_accuracy
from tqdm import tqdm
from typing import Optional

MODEL_SENTIMENT_WORDS = ["积极", "消极", "中性"]
DEBUG = True          # True ➜ 打印 prompt / 回复；False ➜ 不打印
DEBUG_SHOW_N = 3      # 最多打印前 N 条，防止刷屏


def strip_cot(resp: str) -> str:
    if "Thinking..." in resp and "...done thinking." in resp:
        before = resp.split("Thinking...", 1)[0]
        after = resp.split("...done thinking.")[-1]
        resp = before + after
    return resp.strip()


def extract_label(resp: str, true_word="真实", false_word="虚假") -> int:
    if true_word in resp:
        return 1
    if false_word in resp:
        return 0
    return -1


def extract_sentiment(resp: str) -> Optional[str]:
    for w in MODEL_SENTIMENT_WORDS:
        if w in resp:
            return w
    return None


if __name__ == "__main__":
    tsv_path = (
        "/Users/alan/科研/Rumor_Detection_Analysis/twitter_dataset/"
        "testset/posts_groundtruth.txt"
    )
    data = load_news_from_tsv(tsv_path)

    # ——— 任务 1 ———
    print("任务(1) 正在判断真假新闻...")
    preds1, labels1 = [], []
    for text, lab in tqdm(data[:10], desc="判断新闻真假", ncols=80):
        resp = strip_cot(query_ollama(build_truth_prompt(text)))
        pred = extract_label(resp)
        if pred != -1:
            preds1.append(pred)
            labels1.append(lab)
    acc1, acc1_fake, acc1_true = compute_accuracy(preds1, labels1)
    print(f"\n任务(1)：Accuracy={acc1:.2%} | fake={acc1_fake:.2%} | true={acc1_true:.2%}")

    # ——— 任务 2 ———（示例）
    print("\n任务(2) 情感示例：")
    for text, _ in data[:5]:
        s_resp = strip_cot(query_ollama(build_sentiment_prompt(text)))
        print(f"· {text[:40]}... => {s_resp}")

    # ——— 任务 3 ———
    print("\n任务(3) 情感辅助判断真假新闻...")
    preds3, labels3 = [], []
    for idx, (text, lab) in enumerate(tqdm(data[:10], desc="情感+真假判断", ncols=80), start=1):
        # step 1：情感
        sent_prompt = build_sentiment_prompt(text)
        sent_resp = strip_cot(query_ollama(sent_prompt))
        sentiment = extract_sentiment(sent_resp)
        if sentiment is None:
            continue

        # step 2：combined
        comb_prompt = build_combined_prompt(text, sentiment)
        resp = strip_cot(query_ollama(comb_prompt))
        pred = extract_label(resp)
        if pred != -1:
            preds3.append(pred)
            labels3.append(lab)

        # ——— DEBUG 打印 ———
        if DEBUG and idx <= DEBUG_SHOW_N:
            print("\n===== Sample", idx, "=====")
            print("[Sentiment prompt]\n", sent_prompt)
            print("[Sentiment resp ]\n", sent_resp)
            print("[Combined prompt]\n", comb_prompt)
            print("[Combined resp  ]\n", resp)
            print("-> sentiment =", sentiment, " pred =", pred, " label =", lab)

    acc3, acc3_fake, acc3_true = compute_accuracy(preds3, labels3)
    print(f"\n任务(3)：Accuracy={acc3:.2%} | fake={acc3_fake:.2%} | true={acc3_true:.2%}")

    # ——— 任务 4 ———
    delta = (acc3 - acc1) * 100
    trend = "提升" if delta > 0 else "下降" if delta < 0 else "持平"
    print(f"\n=== 对比 ===  情感辅助后 {trend} {abs(delta):.2f}%  "
          f"( {acc1:.2%}  →  {acc3:.2%} )")