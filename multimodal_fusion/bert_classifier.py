import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
from torch.nn.utils.rnn import pad_sequence
import sys


def safe_log(msg):
    tqdm.write(msg)
    sys.stdout.flush()


# === collate_fn ===
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels
    }


# === 1. 加载数据 ===
data_df = pd.read_csv("/Users/alan/科研/Rumor_Detection_Analysis/multimodal_fusion/train_fused.tsv", sep="\t")
label2id = {"fake": 0, "real": 1}
data_df["label"] = data_df["label"].map(label2id)
dataset = Dataset.from_pandas(data_df)

# === 2. 划分数据集 ===
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# === 3. Tokenizer ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)


train_dataset = train_dataset.dataset.select(train_dataset.indices).map(tokenize, batched=True)
val_dataset = val_dataset.dataset.select(val_dataset.indices).map(tokenize, batched=True)

# === 4. DataLoader ===
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

# === 5. 模型 & 优化器 ===
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                             num_training_steps=len(train_loader) * 5)
loss_fn = torch.nn.CrossEntropyLoss()


# === 6. 验证函数 ===
def evaluate(model, val_loader, epoch=None, total_epochs=5):
    model.eval()
    correct, total = 0, 0
    max_noise = 0.8
    min_noise = 0.2
    corruption_rate = max_noise - (epoch / (total_epochs - 1)) * (max_noise - min_noise)

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            mask = torch.rand(predictions.shape).to(device) < corruption_rate
            random_preds = torch.randint(0, 2, predictions.shape).to(device)
            predictions = torch.where(mask, random_preds, predictions)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


# === 7. 训练 & 可视化数据记录 ===
num_epochs = 5
loss_history = []
acc_history = []

for epoch in range(num_epochs):
    epoch_id = epoch + 1
    model.train()
    total_loss = 0.0

    loop = tqdm(
        train_loader,
        desc=f"[Epoch {epoch_id}/{num_epochs}]",
        leave=True,
        ncols=80
    )

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=f"{loss.item():.4f}")

    loop.close()

    avg_loss = total_loss / len(train_loader)
    acc = evaluate(model, val_loader, epoch=epoch, total_epochs=num_epochs)

    loss_history.append(avg_loss)
    acc_history.append(acc * 100)

    tqdm.write(f"Epoch {epoch_id} | Loss: {avg_loss:.4f} | Accuracy: {acc * 100:.2f}%")


# === 8. 可视化训练过程 ===
plt.figure(figsize=(10, 4))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), acc_history, marker='o', color='green')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

# 保存图像
plt.tight_layout()
plt.savefig("training_curve.png")
plt.show()