# task1_detect_fake.py  —— 未知直接当 0 处理
from pathlib import Path
from csv import DictWriter
from tqdm import tqdm

from data_loader import load_news_from_tsv
from prompt_templates import build_truth_prompt
from ollama_utils import query_ollama
from evaluation_metrics import compute_accuracy
from main_utils import strip_cot, extract_label

DATA_PATH = "/Users/alan/科研/Rumor_Detection_Analysis/twitter_dataset/testset/posts_groundtruth.txt"
OUT_DIR   = Path("results")
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE  = OUT_DIR / "task1_preds.tsv"

data = load_news_from_tsv(DATA_PATH)

preds, labels, rows = [], [], []
for idx, (text, lab) in enumerate(tqdm(data[:250], desc="Task-1")):  # 改成 data 则跑全量
    post_id = f"post_{idx:04d}"

    resp = strip_cot(query_ollama(build_truth_prompt(text)))
    pred = extract_label(resp)
    if pred == -1:
        pred = 0

    preds.append(pred)
    labels.append(lab)
    rows.append({"post_id": post_id, "pred": pred})

# —— 写出 TSV ——
with OUT_FILE.open("w", encoding="utf-8", newline="") as f:
    writer = DictWriter(f, fieldnames=["post_id", "pred"], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

# —— 计算准确率（全部样本） ——
acc, acc_fake, acc_true = compute_accuracy(preds, labels)
print(f"Task-1 Accuracy={acc:.2%} fake={acc_fake:.2%} true={acc_true:.2%}")