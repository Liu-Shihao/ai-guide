import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
"""
pip install matplotlib
"""
# 假设你已经得到了评估数据的真实标签和模型预测的概率
# true_labels是真实标签，predictions是模型预测的概率
true_labels = np.array([1, 0, 1, 0])  # 样例真实标签
predictions = np.array([0.9, 0.3, 0.8, 0.1])  # 样例模型预测的概率

# 计算ROC曲线上的点
fpr, tpr, _ = roc_curve(true_labels, predictions)

# 计算AOC
aoc = auc(fpr, tpr)
print("AOC:", aoc)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % aoc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

