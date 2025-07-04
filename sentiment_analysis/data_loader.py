# data_loader.py
import csv

def load_news_from_tsv(file_path):
    """
    读取 posts.txt 或 posts_groundtruth.txt
    返回 [(text, label)]，label: 0=fake 1=real
    """
    samples = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["post_text"].strip()
            label_word = row["label"].strip().lower()
            if label_word == "fake":
                label = 0
            elif label_word == "real":
                label = 1
            else:
                continue        # 跳过未知标签
            samples.append((text, label))
    return samples