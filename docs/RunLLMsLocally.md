# Run LLMs Locally

在本地运行大语言模型最大的优势就是可以增强隐私和安全性。

已经出现了一些框架来支持开源 LLM 在各种设备上的推理：

• llama.cpp：带有权重优化/量化的 llama 推理代码的 C++ 实现

• gpt4all：优化 C 后端推理

• Ollama：将模型权重和环境捆绑到在设备上运行并为 LLM 提供服务的应用程序中

• llamafile：将模型权重和运行模型所需的所有内容捆绑在一个文件中，允许您从此文件在本地运行 LLM，无需任何额外的安装步骤

一般来说，这些框架会做一些事情：

1. 减少原始模型权重的内存占用

2. 支持消费类硬件（例如CPU或笔记本电脑GPU）上的推理

因此,Ollama等框架通过模型打包、依赖管理和GPU加速等方式,大大简化了LLM在本地部署的难度,使得普通用户也能够在自己的电脑上运行和探索这些强大的AI模型。

# Ollama

首先需要下载： [Ollama](https://ollama.com/download)

只需要运行 `ollama run llama2` 命令

#### API

```
curl http://localhost:11434/api/generate -d '{
 "model": "llama2",
 "prompt":"Why is the sky blue?"
}'
```

#### LangChain

```
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

llm.invoke("Tell me a joke")
```

[Langchain&Ollama](https://python.langchain.com/docs/integrations/llms/ollama)

# GPT4All

下载[GPT4ALL](https://gpt4all.io/index.html)客户端.

免费使用、本地运行 、具有隐私意识的聊天机器人、无需GPU资源、无需网络。

GPT4All 是一个生态系统，用于训练和部署在消费级 CPU 上本地运行的强大且定制的大型语言模型。

#### LangChain

```
%pip install gpt4all
```

```
from langchain_community.llms import GPT4All

llm = GPT4All(
    model="/Users/rlm/Desktop/Code/gpt4all/models/nous-hermes-13b.ggmlv3.q4_0.bin"
)

llm.invoke("The first man on the moon was ... Let's think step by step")
```

# Llamafile

本地运行 LLM 最简单的方法之一是使用 [llamafile](https://github.com/Mozilla-Ocho/llamafile)

下载您想要使用的型号的 llamafile。您可以在 [HuggingFace](https://huggingface.co/models?other=llamafile)上找到许多 llamafile 格式的模型。

下载[llava-v1.5-7b-q4.llamafile](https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-q4.llamafile?download=true)(3.97 GB)模型

```shell
# 授予可执行权限
chmod +x llava-v1.5-7b-q4.llamafile
# 启动
./llava-v1.5-7b-q4.llamafile -ngl 9999
```

#### LangChain

```
from langchain_community.llms.llamafile import Llamafile

llm = Llamafile()

llm.invoke("The first man on the moon was ... Let's think step by step.")
```