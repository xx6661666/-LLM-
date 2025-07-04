# task2_sentiment.py  —— 未知时也写入
from pathlib import Path
from csv import DictWriter
from tqdm import tqdm

from data_loader import load_news_from_tsv
from prompt_templates import build_sentiment_prompt
from ollama_utils import query_ollama
from main_utils import strip_cot, extract_sentiment

# DATA_PATH = "/Users/alan/科研/Rumor_Detection_Analysis/twitter_dataset/testset/posts_groundtruth.txt"
DATA_PATH = "/Users/alan/科研/Rumor_Detection_Analysis/twitter_dataset/devset/posts.txt"
# OUT_DIR = Path("results")
OUT_DIR = Path("/Users/alan/科研/Rumor_Detection_Analysis/multimodal_fusion/results")
OUT_DIR.mkdir(exist_ok=True)
# OUT_FILE  = OUT_DIR / "task2_sentiment.tsv"
OUT_FILE  = OUT_DIR / "devset_sentiment.tsv"

data = load_news_from_tsv(DATA_PATH)

rows = []
for idx, (text, _) in enumerate(tqdm(data[6750:6950], desc="Task-2")):   # 改成 data  跑全量
    post_id = f"post_{idx:04d}"
    resp    = strip_cot(query_ollama(build_sentiment_prompt(text)))
    senti   = extract_sentiment(resp)
    if senti is None:
        senti = "未知"
    rows.append({"post_id": post_id, "sentiment": senti})

with OUT_FILE.open("w", encoding="utf-8", newline="") as f:
    writer = DictWriter(f, fieldnames=["post_id", "sentiment"], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print(f"Task-2 完成，共写入 {len(rows)} 条情感结果 → {OUT_FILE}")