# topic_modeling/topic_expansion.py

import os
import re
import json
import requests
from tqdm import tqdm

POST_PATH = '/Users/alan/科研/Rumor_Detection_Analysis/twitter_dataset/devset/posts.txt'
OUTPUT_JSON = '/Users/alan/科研/Rumor_Detection_Analysis/multimodal_fusion/devset_expanded_posts.json'

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"

def clean_text(text):
    """正则清洗文本：去除非字母字符，统一小写"""
    text = re.sub(r'[^a-zA-Z ]+', '', text)
    return text.lower().strip()

def call_deepseek_model(prompt):
    """调用本地 DeepSeek 模型，生成上下文扩展描述"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""

def expand_posts(post_list):
    """对所有推文文本进行上下文扩展"""
    extended_data = []
    for idx, post in enumerate(tqdm(post_list, desc="Expanding posts")):
        clean_post = clean_text(post)
        prompt = f"Based on the short tweet below, generate one sentence to describe its possible background or context.\nTweet: \"{clean_post}\"\nContext:"
        context = call_deepseek_model(prompt)
        combined = clean_post + " " + context
        extended_data.append({
            "id": idx,
            "original": post.strip(),
            "cleaned": clean_post,
            "context": context,
            "expanded": combined
        })
    return extended_data

def main():
    with open(POST_PATH, 'r', encoding='utf-8') as f:
        raw_posts = f.readlines()[6750:6950]  # 处理200条推文 有real 有fake

    expanded_posts = expand_posts(raw_posts)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(expanded_posts, f, indent=2, ensure_ascii=False)

    print(f"上下文扩展完成，已保存至 {OUTPUT_JSON}")

if __name__ == "__main__":
    main()