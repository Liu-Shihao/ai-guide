<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
GLUE（General Language Understanding Evaluation）任务是一个用于评估自然语言理解（NLU）模型性能的基准测试套件。它由一系列针对自然语言处理任务的数据集组成，这些任务涵盖了文本分类、句子匹配、自然语言推理等多个领域。GLUE任务的目标是提供一个标准化的评估平台，用于比较不同模型在各种NLU任务上的性能。

GLUE任务包含了以下九个子任务：

1. **CoLA (Corpus of Linguistic Acceptability)**：句子接受性判断，判断句子是否为语法正确、语义连贯的句子。

2. **SST-2 (The Stanford Sentiment Treebank)**：句子情感分类，将句子划分为积极或消极情感。

3. **MRPC (Microsoft Research Paraphrase Corpus)**：句子匹配，判断两个句子是否是语义上的等价或者是一对句子。

4. **QQP (Quora Question Pairs)**：句子相似度，判断两个问题是否语义上等价。

5. **STS-B (Semantic Textual Similarity Benchmark)**：句子相似度，给定两个句子，评估它们之间的语义相似性得分。

6. **MNLI (Multi-Genre Natural Language Inference)**：自然语言推理，给定一个前提和一个假设，判断前提是否能推出假设。

7. **QNLI (Question Natural Language Inference)**：自然语言推理，给定一个问题和一个句子，判断句子是否包含问题的答案。

8. **RTE (Recognizing Textual Entailment)**：自然语言推理，判断一个句子是否可以从另一个句子推出。

9. **WNLI (Winograd Schema Challenge)**：自然语言推理，给定一组句子，判断它们之间的关系。

通过这些任务，GLUE旨在提供一个统一的基准，帮助研究人员评估不同NLU模型的性能，并促进NLU领域的进步。
# Text classification examples
```shell
pip install -r requirements.txt
```
## GLUE tasks

Based on the script [`run_glue.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py).

Fine-tuning the library models for sequence classification on the GLUE benchmark: [General Language Understanding
Evaluation](https://gluebenchmark.com/). This script can fine-tune any of the models on the [hub](https://huggingface.co/models)
and can also be used for a dataset hosted on our [hub](https://huggingface.co/datasets) or your own data in a csv or a JSON file
(the script might need some tweaks in that case, refer to the comments inside for help).

GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:

```bash
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```

where task name can be one of cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli.

We get the following results on the dev set of the benchmark with the previous commands (with an exception for MRPC and
WNLI which are tiny and where we used 5 epochs instead of 3). Trainings are seeded so you should obtain the same
results with PyTorch 1.6.0 (and close results with different versions), training times are given for information (a
single Titan RTX was used):

| Task  | Metric                       | Result      | Training time |
|-------|------------------------------|-------------|---------------|
| CoLA  | Matthews corr                | 56.53       | 3:17          |
| SST-2 | Accuracy                     | 92.32       | 26:06         |
| MRPC  | F1/Accuracy                  | 88.85/84.07 | 2:21          |
| STS-B | Pearson/Spearman corr.       | 88.64/88.48 | 2:13          |
| QQP   | Accuracy/F1                  | 90.71/87.49 | 2:22:26       |
| MNLI  | Matched acc./Mismatched acc. | 83.91/84.10 | 2:35:23       |
| QNLI  | Accuracy                     | 90.66       | 40:57         |
| RTE   | Accuracy                     | 65.70       | 57            |
| WNLI  | Accuracy                     | 56.34       | 24            |

Some of these results are significantly different from the ones reported on the test set of GLUE benchmark on the
website. For QQP and WNLI, please refer to [FAQ #12](https://gluebenchmark.com/faq) on the website.

The following example fine-tunes BERT on the `imdb` dataset hosted on our [hub](https://huggingface.co/datasets):

```bash
python run_glue.py \
  --model_name_or_path google-bert/bert-base-cased \
  --dataset_name imdb  \
  --do_train \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/imdb/
```

> If your model classification head dimensions do not fit the number of labels in the dataset, you can specify `--ignore_mismatched_sizes` to adapt it.

## Text classification
As an alternative, we can use the script [`run_classification.py`](./run_classification.py) to fine-tune models on a single/multi-label classification task. 

The following example fine-tunes BERT on the `en` subset of  [`amazon_reviews_multi`](https://huggingface.co/datasets/amazon_reviews_multi) dataset.
We can specify the metric, the label column and aso choose which text columns to use jointly for classification. 
```bash
dataset="amazon_reviews_multi"
subset="en"
python run_classification.py \
    --model_name_or_path  google-bert/bert-base-uncased \
    --dataset_name ${dataset} \
    --dataset_config_name ${subset} \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name "review_title,review_body,product_category" \
    --text_column_delimiter "\n" \
    --label_column_name stars \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir /tmp/${dataset}_${subset}/
```
Training for 1 epoch results in acc of around 0.5958 for review_body only and 0.659 for title+body+category.

The following is a multi-label classification example. It fine-tunes BERT on the `reuters21578` dataset hosted on our [hub](https://huggingface.co/datasets/reuters21578):
```bash
dataset="reuters21578"
subset="ModApte"
python run_classification.py \
    --model_name_or_path google-bert/bert-base-uncased \
    --dataset_name ${dataset} \
    --dataset_config_name ${subset} \
    --shuffle_train_dataset \
    --remove_splits "unused" \
    --metric_name f1 \
    --text_column_name text \
    --label_column_name topics \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 15 \
    --output_dir /tmp/${dataset}_${subset}/ 
```
 It results in a Micro F1 score of around 0.82 without any text and label filtering. Note that you have to explicitly remove the "unused" split from the dataset, since it is not used for classification.

### Mixed precision training

If you have a GPU with mixed precision capabilities (architecture Pascal or more recent), you can use mixed precision
training with PyTorch 1.6.0 or latest, or by installing the [Apex](https://github.com/NVIDIA/apex) library for previous
versions. Just add the flag `--fp16` to your command launching one of the scripts mentioned above!

Using mixed precision training usually results in 2x-speedup for training with the same final results:

| Task  | Metric                       | Result      | Training time | Result (FP16) | Training time (FP16) |
|-------|------------------------------|-------------|---------------|---------------|----------------------|
| CoLA  | Matthews corr                | 56.53       | 3:17          | 56.78         | 1:41                 |
| SST-2 | Accuracy                     | 92.32       | 26:06         | 91.74         | 13:11                |
| MRPC  | F1/Accuracy                  | 88.85/84.07 | 2:21          | 88.12/83.58   | 1:10                 |
| STS-B | Pearson/Spearman corr.       | 88.64/88.48 | 2:13          | 88.71/88.55   | 1:08                 |
| QQP   | Accuracy/F1                  | 90.71/87.49 | 2:22:26       | 90.67/87.43   | 1:11:54              |
| MNLI  | Matched acc./Mismatched acc. | 83.91/84.10 | 2:35:23       | 84.04/84.06   | 1:17:06              |
| QNLI  | Accuracy                     | 90.66       | 40:57         | 90.96         | 20:16                |
| RTE   | Accuracy                     | 65.70       | 57            | 65.34         | 29                   |
| WNLI  | Accuracy                     | 56.34       | 24            | 56.34         | 12                   |


## PyTorch version, no Trainer

Based on the script [`run_glue_no_trainer.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py).

Like `run_glue.py`, this script allows you to fine-tune any of the models on the [hub](https://huggingface.co/models) on a
text classification task, either a GLUE task or your own data in a csv or a JSON file. The main difference is that this
script exposes the bare training loop, to allow you to quickly experiment and add any customization you would like.

It offers less options than the script with `Trainer` (for instance you can easily change the options for the optimizer
or the dataloaders directly in the script) but still run in a distributed setup, on TPU and supports mixed precision by
the mean of the [🤗 `Accelerate`](https://github.com/huggingface/accelerate) library. You can use the script normally
after installing it:

```bash
pip install git+https://github.com/huggingface/accelerate
```

then

```bash
export TASK_NAME=mrpc

python run_glue_no_trainer.py \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```

You can then use your usual launchers to run in it in a distributed environment, but the easiest way is to run

```bash
accelerate config
```

and reply to the questions asked. Then

```bash
accelerate test
```

that will check everything is ready for training. Finally, you can launch training with

```bash
export TASK_NAME=mrpc

accelerate launch run_glue_no_trainer.py \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```

This command is the same and will work for:

- a CPU-only setup
- a setup with one GPU
- a distributed training with several GPUs (single or multi node)
- a training on TPUs

Note that this library is in alpha release so your feedback is more than welcome if you encounter any problem using it.

## XNLI

Based on the script [`run_xnli.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_xnli.py).

[XNLI](https://cims.nyu.edu/~sbowman/xnli/) is a crowd-sourced dataset based on [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/). It is an evaluation benchmark for cross-lingual text representations. Pairs of text are labeled with textual entailment annotations for 15 different languages (including both high-resource language such as English and low-resource languages such as Swahili).

#### Fine-tuning on XNLI

This example code fine-tunes mBERT (multi-lingual BERT) on the XNLI dataset. It runs in 106 mins on a single tesla V100 16GB.

```bash
python run_xnli.py \
  --model_name_or_path google-bert/bert-base-multilingual-cased \
  --language de \
  --train_language en \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir /tmp/debug_xnli/ \
  --save_steps -1
```

Training with the previously defined hyper-parameters yields the following results on the **test** set:

```bash
acc = 0.7093812375249501
```

# Reference
- [https://huggingface.co/docs/transformers/v4.39.3/zh/training](https://huggingface.co/docs/transformers/v4.39.3/zh/training)
- [https://huggingface.co/docs/transformers/v4.39.3/zh/run_scripts](https://huggingface.co/docs/transformers/v4.39.3/zh/run_scripts)