
"""
在spaCy中，`ner.evaluate()`函数用于评估命名实体识别（NER）模型的性能。它返回一个包含评估指标的字典，这些指标反映了模型在给定数据上的性能。

下面是`ner.evaluate()`函数返回的字段的解释：

1. **'ents_p'**：实体级别的准确率（Precision）。即模型正确预测的实体数与模型预测的所有实体数的比率。

2. **'ents_r'**：实体级别的召回率（Recall）。即模型正确预测的实体数与测试数据中的所有实体数的比率。

3. **'ents_f'**：实体级别的F1分数。F1分数是准确率和召回率的调和平均值，表示模型在实体识别任务中的综合性能。

4. **'ents_per_type'**：包含每种实体类型的准确率、召回率和F1分数的字典。每个键都是实体类型，对应的值是一个包含该实体类型的准确率、召回率和F1分数的元组。

5. **'tags_acc'**：标签级别的准确率。即模型正确预测的标签数与测试数据中的所有标签数的比率。这个指标对应于NER中的标签级别的精确度。

6. **'token_acc'**：令牌级别的准确率。即模型在预测每个令牌（单词）上的标签时的准确率。

这些字段提供了对模型在NER任务上表现的详细评估，你可以根据自己的需求来选择需要关注的指标。

"""
def find_substring_position(main_string, substring, label):
    start = main_string.find(substring)
    if start == -1:
        return -1, -1  # 如果子字符串不在主字符串中，则返回(-1, -1)
    end = start + len(substring)
    return (start,end,label)

# 文本分类（textcat）任务：
# 对于文本分类任务，test_data 是一个包含文本及其对应标签的列表。每个样本是一个元组，第一个元素是文本，第二个元素是标签。
test_data = [
    ("This is a positive review.", {"cats": {"POSITIVE": 1, "NEGATIVE": 0}}),
    ("This is a negative review.", {"cats": {"POSITIVE": 0, "NEGATIVE": 1}}),
    # Add more samples here
]

# 命名实体识别（NER）任务：
# 在这个例子中，每个样本都有一个文本和一个字典，字典中的键是 "entities"，对应的值是一个列表，其中每个元素是一个三元组，表示一个命名实体的起始位置（包括）、结束位置（不包括）和标签。
test_data = [
    ("Apple is looking at buying U.K. startup for $1 billion", {"entities": [(0, 5, "ORG"), (27, 30, "GPE"), (44, 52, "MONEY")]}),
    ("Elon Musk is the CEO of SpaceX and Tesla.", {"entities": [(0, 9, "PERSON"), (23, 28, "ORG"), (33, 38, "ORG")]}),
    # Add more samples here
]

