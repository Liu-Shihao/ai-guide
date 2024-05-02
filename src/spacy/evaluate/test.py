import spacy
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
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
print(y_prob)

# 计算每个类别的 AUC-ROC
auc_scores = []
plt.figure(figsize=(10, 8))
for label in y_prob[0].keys():
    y_true_binary = np.array([1 if sample_label == label else 0 for sample_label in y_true])
    y_scores = np.array([sample_probs[label] for sample_probs in y_prob])
    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-label Classification')
plt.legend()
plt.grid()
plt.show()
