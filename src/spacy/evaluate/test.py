import numpy as np
import spacy
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 加载 SpaCy 模型
nlp = spacy.load("en_core_web_sm")

# 获取测试数据
X_test = ["I want transfer $2 to joey on 2024-05-02"]  # 你的测试数据
y_test = ["TRANSFER"]  # 真实标签

# 预测概率
y_prob = []
for text in X_test:
    doc = nlp(text)
    # 你的 textcat 模型名称取决于你在训练过程中使用的名称
    # 这里假设你的 textcat 名称为 "textcat"
    print(doc.cats)
    probs = doc.cats
    y_prob.append(probs)

y_prob = np.array(y_prob)

# 计算 AUC-ROC
auc_scores = []
for i in range(y_prob.shape[1]):
    auc = roc_auc_score(y_test[:, i], y_prob[:, i])
    auc_scores.append(auc)

# 绘制 ROC 曲线
plt.figure(figsize=(10, 8))
for i in range(y_prob.shape[1]):
    fpr, tpr, _ = roc_curve(y_test[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_scores[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Text Classification')
plt.legend(loc='lower right')
plt.show()
