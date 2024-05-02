import spacy
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 加载 SpaCy 模型
nlp = spacy.load("your_model_name")

# 获取测试数据和真实标签
X_test = [...]  # 你的测试数据
y_test = [...]  # 真实标签，每个样本只有一个标签，表示类别

# 预测概率
y_prob = []
for text in X_test:
    doc = nlp(text)
    # 假设你的 textcat 模型名称为 "textcat"
    probs = doc.cats["textcat"]
    y_prob.append(probs)

y_prob = np.array(y_prob)

# 计算每个类别的 AUC-ROC
auc_scores = []
plt.figure(figsize=(10, 8))
for i in range(y_prob.shape[1]):
    # 取出当前类别的真实标签和预测概率
    y_true_binary = np.array([1 if label == i else 0 for label in y_test])
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
    auc = roc_auc_score(y_true_binary, y_prob[:, i])
    auc_scores.append(auc)
    # 绘制当前类别的 ROC 曲线
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-class Text Classification')
plt.legend(loc='lower right')
plt.show()
