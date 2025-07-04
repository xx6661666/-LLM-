import os
import json
import requests
from tqdm import tqdm


# 原始路径
TOPIC_PATH = '/Users/alan/科研/Rumor_Detection_Analysis/multimodal_fusion/devset_lda_results/topics_keywords.json'
OUTPUT_TXT = '/Users/alan/科研/Rumor_Detection_Analysis/multimodal_fusion/devset_lda_results/analyzed_topics.txt'

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"

def call_deepseek_model(prompt):
    """调用本地 DeepSeek 模型，返回主题分析结果"""
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
        print(f"调用模型出错: {e}")
        return ""

def analyze_topics(topic_dict):
    """对每个主题关键词调用大模型进行解释"""
    analyzed = []
    for topic_id, keywords in tqdm(topic_dict.items(), desc="分析每个主题"):
        keyword_str = ', '.join(keywords)
        prompt = f"""你是一个研究社交媒体谣言的研究员。我们使用LDA模型从推文中提取了如下关键词集合（代表某个主题）：

关键词：{keyword_str}

请你完成以下任务：
(1) 为该主题起一个简洁的中文名称；
(2) 用一句话总结该主题内容；
(3) 简要解释这个主题可能代表的网络事件背景或讨论语境。

请用如下格式回复：

主题编号：{topic_id}
主题名称：XXX
主题总结：XXX
主题解释：XXX
"""
        result = call_deepseek_model(prompt)
        analyzed.append(result)
        analyzed.append("-" * 60)
    return analyzed

def main():
    with open(TOPIC_PATH, 'r', encoding='utf-8') as f:
        topic_dict = json.load(f)

    results = analyze_topics(topic_dict)

    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))

    print(f"\n分析结果已保存至 {OUTPUT_TXT}")

if __name__ == "__main__":
    main()