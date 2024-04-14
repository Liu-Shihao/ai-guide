from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
import evaluate


# 您的训练和测试数据集创建一个DataLoader类，以便可以迭代处理数据批次
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

# 加载您的模型，并指定期望的标签数量：
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

# 创建一个optimizer和learning rate scheduler以进行模型微调。让我们使用 PyTorch 中的 AdamW 优化器：
optimizer = AdamW(model.parameters(), lr=5e-5)

# 创建来自 Trainer 的默认learning rate scheduler：
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# 指定 device 以使用 GPU（如果有的话）。否则，使用 CPU 进行训练可能需要几个小时，而不是几分钟
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 训练循环
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# 评估
metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
