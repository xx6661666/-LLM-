import json
import csv
import time
import requests
from tqdm import tqdm

# 路径配置
POST_FILE = "/Users/alan/科研/Rumor_Detection_Analysis/multimodal_fusion/devset_expanded_posts.json"
SENTI_FILE = "/Users/alan/科研/Rumor_Detection_Analysis/multimodal_fusion/results/devset_sentiment.tsv"
OUTPUT_FILE = "/Users/alan/科研/Rumor_Detection_Analysis/multimodal_fusion/devset_post_topic_assignment.tsv"

# 模型配置
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"

# 主题文本（7个主题版本）
TOPIC_BLOCK = """
你是一个社交媒体研究员，请根据下面的推文内容判断它最接近哪个主题，仅回答 1 到 7 中的一个数字（不要输出其他任何内容）：

主题 1：关于网络谣言与虚假信息的讨论，分析谣言真相及其传播现象。
主题 2：实相关讨论，公众对真实性内容的质疑与分歧。
主题 3：April相关舆论或争议性讨论，含不实言论与误解。
主题 4：关于水下地点的谣言或调查质疑（如fake underwater）。
主题 5：与幽默或反转相关的推文事件，涉及April与fake标签。
主题 6：与FBI和Boston真实调查有关的事件讨论。
主题 7：关于 Boston 大火事件的公众反应与嫌疑调查。

推文如下：
"""

def query_llm(post_text):
    prompt = TOPIC_BLOCK + post_text + "\n请问这个推文最符合哪个主题？（只回答数字1~7）："

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        result = response.json().get("response", "").strip()
        for c in result:
            if c in "1234567":
                return f"topic_{int(c)-1}"
        return "未知"
    except Exception as e:
        print(f"请求失败: {e}")
        return "未知"

def load_sentiments(path):
    sentiments = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sentiments[row["post_id"]] = row["sentiment"]
    return sentiments

def main():
    with open(POST_FILE, 'r', encoding='utf-8') as f:
        posts = json.load(f)

    sentiments = load_sentiments(SENTI_FILE)

    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerow(["post_id", "sentiment", "most_related_topic"])

        for idx, post in enumerate(tqdm(posts, desc="分配主题")):
            post_id = f"post_{idx:04d}"
            sentiment = sentiments.get(post_id, "未知")
            topic = query_llm(post['expanded'])
            writer.writerow([post_id, sentiment, topic])
            time.sleep(0.5)  # 防止请求过快造成模型阻塞

    print(f"\n已完成所有主题分配，输出文件保存至：{OUTPUT_FILE}")

if __name__ == "__main__":
    main()