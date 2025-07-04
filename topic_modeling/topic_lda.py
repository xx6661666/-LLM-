# topic_modeling/topic_lda.py

import os
import json
import re
import nltk
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from wordcloud import WordCloud
from gensim import corpora, models
from pprint import pprint

# 文件路径设置
INPUT_PATH = '/Users/alan/科研/Rumor_Detection_Analysis/topic_modeling/expanded_posts.json'
OUTPUT_DIR = '/Users/alan/科研/Rumor_Detection_Analysis/topic_modeling/lda_results'
NUM_TOPICS = 3

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def preprocess(text):
    """基础预处理：小写、去特殊符号、去停用词"""
    text = re.sub(r'[^a-zA-Z ]+', '', text.lower())
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return tokens

def load_data():
    """加载扩展后的推文"""
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['expanded'] for item in data]

def save_wordclouds(lda_model, dictionary):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i in range(NUM_TOPICS):
        topic_words = dict(lda_model.show_topic(i, topn=30))
        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
        wc.to_file(os.path.join(OUTPUT_DIR, f'topic_{i}_wordcloud.png'))

def save_topic_keywords(lda_model):
    topics = {}
    for i in range(NUM_TOPICS):
        topics[f"topic_{i}"] = [word for word, _ in lda_model.show_topic(i, topn=15)]
    with open(os.path.join(OUTPUT_DIR, 'topics_keywords.json'), 'w', encoding='utf-8') as f:
        json.dump(topics, f, indent=2)
    pprint(topics)

def visualize_lda(lda_model, corpus, dictionary):
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, os.path.join(OUTPUT_DIR, 'lda_visualization.html'))

def main():
    print("加载数据并预处理...")
    texts = load_data()
    processed_texts = [preprocess(text) for text in texts]

    print("构建词典与语料库...")
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    print("训练LDA模型...")
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        random_state=42,
        passes=15,
        alpha='auto'
    )

    print("保存可视化与关键词...")
    save_wordclouds(lda_model, dictionary)
    save_topic_keywords(lda_model)
    visualize_lda(lda_model, corpus, dictionary)

    print(f"所有结果保存至 {OUTPUT_DIR}")

if __name__ == '__main__':
    main()