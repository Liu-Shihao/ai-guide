import torch
from transformers import BertTokenizer, BertModel
import numpy as np

from src.fine_train import TextClassifier

# 加载微调后的模型和分词器
model_path = './fine_tuned_model.pth'

# model = BertModel.from_pretrained(model_path)
# model = torch.load(model_path)
pretrained_model = BertModel.from_pretrained('bert-base-uncased')

model = TextClassifier(pretrained_model, 7)  # 在此处需要重新创建模型实例，保证模型结构一致
model.load_state_dict(torch.load('fine_tuned_model.pth'))
model.eval()  # 设置模型为评估模式
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 准备文本数据
text = "I want to transfer $100 to Anirudh."

# 分词和编码
tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

# 使用模型进行推断
with torch.no_grad():
    # outputs = model(input_ids, attention_mask)
    # hidden_states = outputs[0]  # 获取模型输出的第一个元素，即最后一层的隐藏状态
    # logits = hidden_states[:, 0, :]  # 取第一个token的隐藏状态作为logits
    # probabilities = torch.softmax(logits, dim=1)
    # predictions = torch.argmax(probabilities, dim=1)
    outputs = model(input_ids, attention_mask)
    hidden_states = outputs[0]  # 获取模型输出的第一个元素，即最后一层的隐藏状态
    print(hidden_states.shape)  # 打印隐藏状态的形状
# 输出分类结果
# print("Predicted label:", predictions.item())
