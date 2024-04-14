import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.fine_train import TextClassifier

# 数据准备

# 文本数据
texts = [
    "I want to transfer $100 to Anirudh.",
    "I want to transfer money.",
    "I want to do wire transfer.",
    "I want to approve my wires.",
    "I want to approve my transactions.",
    "I want to see today's transaction activity.",
    "Can I get a copy of Loan information?",
    "I want to make a loan repricing request.",
    "how do I apply for a loan?",
    "I would like to apply for a loan."
]
# 对应的标签：从0开始，数值不能超过种类数
labels = [1, 2, 2, 3, 3, 4, 5, 6, 0, 0]
num_labels = len(set(labels))


# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', num_labels=num_labels)
model = BertModel.from_pretrained('bert-base-uncased')

# 对文本数据进行分词和编码
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# 将文本编码转换为张量
train_inputs = torch.tensor(train_encodings['input_ids'])
val_inputs = torch.tensor(val_encodings['input_ids'])
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
train_masks = torch.tensor(train_encodings['attention_mask'])
val_masks = torch.tensor(val_encodings['attention_mask'])

# 定义数据加载器
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 定义文本分类模型


# 创建模型实例

model = TextClassifier(model, num_labels)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 10)

# 训练模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # 在验证集上评估模型
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            val_correct += torch.sum(preds == labels).item()
    val_loss /= len(val_loader.dataset)
    val_accuracy = val_correct / len(val_loader.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

# 保存模型到文件
torch.save(model.state_dict(), 'fine_tuned_model.pth')


# # 加载模型
# pretrained_model = BertModel.from_pretrained('bert-base-uncased')
#
# model = TextClassifier(pretrained_model, num_classes)  # 在此处需要重新创建模型实例，保证模型结构一致
# model.load_state_dict(torch.load('fine_tuned_model.pth'))
# model.eval()  # 设置模型为评估模式

