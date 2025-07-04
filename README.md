# 🔍 LLM-BERT-SocialRumorDetect 🚀
> **融合“大语言模型 (LLM) + BERT” 的社交媒体谣言检测实验框架**  

<div align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue?logo=python">
  <img src="https://img.shields.io/badge/pytorch-2.0-lightgrey?logo=pytorch">
  <img src="https://img.shields.io/badge/transformers-4.41-orange?logo=huggingface">
  <img src="https://img.shields.io/badge/license-MIT-green">
</div>

---

## 🗺️ 项目亮点
| 功能 | 描述 |
|---|---|
| 🎯 **多源语义融合** | 结合 LLM 生成的 **主题** + **情感** 标签，拼接成自然语言输入 🔗 |
| 🤖 **本地 LLM 部署** | 完整脚本一键运行 `DeepSeek-r1` 模型，自动抽主题 & 判情感 |
| 🐍 **简洁训练管线** | 基于 Hugging Face `Trainer` 封装：微调、评估、日志 ✅ |
| 📊 **可视化工具**  | `matplotlib` 曲线、`pyLDAvis` 主题分布、WordCloud 词云 🌥️ |
| 🔌 **可扩展接口**  | 预留图像、用户元信息等多模态入口，支持二次开发 🔄 |

---

## ⚡️ 快速上手

```bash
# 1️⃣ 克隆仓库
git clone https://github.com/xx6661666/LLM-BERT-SocialRumorDetect.git
cd LLM-BERT-SocialRumorDetect

# 2️⃣ 创建环境并安装依赖
conda env create -f environment.yml
conda activate rumor

# 3️⃣ 下载 / 准备数据集（示例 Twitter）
bash scripts/download_data.sh         # 或自行放置到 ./data

# 4️⃣ 本地部署 DeepSeek（或自行替换为其他 LLM）
bash scripts/launch_deepseek.sh

# 5️⃣ 生成主题 + 情感标签
python src/llm_generate_tags.py  --input ./data/posts.txt  --output ./outputs/tags.tsv

# 6️⃣ 构造融合文本并微调 BERT
python src/build_fused_corpus.py
python src/train_bert.py --config configs/bert_base.yaml
``` 
📦 执行完毕后，结果（准确率、F1、Loss 曲线）将自动保存在 runs/ 目录。
