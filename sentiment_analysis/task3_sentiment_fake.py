# task3_sentiment_fake.py  —— 所有异常一律记 0
from pathlib import Path
import csv
from tqdm import tqdm

from data_loader import load_news_from_tsv
from prompt_templates import build_combined_prompt
from ollama_utils import query_ollama
from evaluation_metrics import compute_accuracy
from main_utils import strip_cot, extract_label

DATA_PATH  = "/Users/alan/科研/Rumor_Detection_Analysis/twitter_dataset/testset/posts_groundtruth.txt"
SENT_FILE  = Path("results/task2_sentiment.tsv")
TASK1_FILE = Path("results/task1_preds.tsv")
OUT_FILE   = Path("results/task3_preds.tsv")

# —— 读取 Task-2 情感 ——
sent_map = {r["post_id"]: r["sentiment"]
            for r in csv.DictReader(SENT_FILE.open(), delimiter="\t")}

data = load_news_from_tsv(DATA_PATH)

preds3, truth_map, rows = [], {}, []
for idx, (text, lab) in enumerate(tqdm(data[:250], desc="Task-3")):   # 改成 data 跑全量
    post_id   = f"post_{idx:04d}"
    sentiment = sent_map.get(post_id, "中性")     # 若缺失给默认“中性”

    # 若 sentiment == "未知" 仍然使用 "中性" 构造 prompt
    if sentiment == "未知":
        sentiment = "中性"

    resp = strip_cot(query_ollama(build_combined_prompt(text, sentiment)))
    pred = extract_label(resp)
    if pred not in (0, 1):                       # 任意异常一律当 0
        pred = 0

    preds3.append(pred)
    truth_map[post_id] = lab
    rows.append({"post_id": post_id, "pred": pred})

# —— 保存结果 ——
with OUT_FILE.open("w", encoding="utf-8", newline="") as f:
    csv.DictWriter(f, fieldnames=["post_id", "pred"], delimiter="\t")\
       .writeheader(); f.writelines(f"{r['post_id']}\t{r['pred']}\n" for r in rows)

# —— 计算 Task-3 准确率 ——
labels3 = [truth_map[r["post_id"]] for r in rows]
acc3, acc3_f, acc3_t = compute_accuracy(preds3, labels3)
print(f"Task-3 Accuracy={acc3:.2%}  fake={acc3_f:.2%}  true={acc3_t:.2%}")

# —— 读取 Task-1 结果 ——
task1_preds = {r["post_id"]: int(r["pred"])
               for r in csv.DictReader(TASK1_FILE.open(), delimiter="\t")}

# —— 对比 ——
orig = [task1_preds[r["post_id"]] for r in rows]   # Task-1 预测
new  = [r["pred"] for r in rows]                   # Task-3 预测
truth= labels3

acc_orig, *_ = compute_accuracy(orig, truth)
delta = acc3 - acc_orig
trend = "↑提升" if delta > 0 else "↓下降" if delta < 0 else "→持平"
print(f"⊕ 情感辅助 {trend} {abs(delta)*100:.2f}%  "
      f"(Task-1 {acc_orig:.2%} → Task-3 {acc3:.2%})")