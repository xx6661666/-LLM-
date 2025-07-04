import pandas as pd

# 路径配置
POSTS_PATH = "/Users/alan/科研/Rumor_Detection_Analysis/twitter_dataset/devset/posts.txt"
TOPIC_SENTI_PATH = "/Users/alan/科研/Rumor_Detection_Analysis/multimodal_fusion/devset_post_topic_assignment.tsv"
OUTPUT_TSV = "/Users/alan/科研/Rumor_Detection_Analysis/multimodal_fusion/train_fused.tsv"

# 七个主题映射：编号 -> 中文描述
topic_map = {
    "topic_0": "关于网络谣言与虚假信息的讨论",
    "topic_1": "实相关讨论",
    "topic_2": "April讨论",
    "topic_3": "something fake underwater",
    "topic_4": "april something funny or humorous",
    "topic_5": "real_fbi_boston调查",
    "topic_6": "关于 Boston 大火事件的讨论"
}

# 读取原始推文内容与标签
posts_df = pd.read_csv(POSTS_PATH, sep='\t')

# 读取每条推文的情感和主题分配结果
topic_senti_df = pd.read_csv(TOPIC_SENTI_PATH, sep='\t')

# 构造融合文本
fused_texts = []
for _, row in topic_senti_df.iterrows():
    post_id = row['post_id']
    post_index = int(post_id.split('_')[1])  # 如 post_0035 -> 35
    sentiment = row['sentiment']
    topic_code = row['most_related_topic']
    topic_description = topic_map.get(topic_code, "未知主题")

    # 获取原始推文内容和标签
    post_text = posts_df.iloc[post_index]['post_text']
    label = posts_df.iloc[post_index]['label']

    # 构造融合文本（包含情感+主题+原文）
    fused = f"情感为{sentiment}，主题是{topic_description}。这条推文内容是：{post_text}"
    fused_texts.append((fused, label))

# 保存融合后的数据
fused_df = pd.DataFrame(fused_texts, columns=["text", "label"])
fused_df.to_csv(OUTPUT_TSV, sep='\t', index=False)

print(f"✅ 融合文本已保存至：{OUTPUT_TSV}")