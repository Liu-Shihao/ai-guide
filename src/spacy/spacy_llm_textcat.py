from spacy_llm.util import assemble

"""
ValueError: Install CUDA to load and run the LLM on the GPU, or install 'accelerate' to dynamically distribute the LLM on the CPU or even the hard disk. The latter may be slow.

这个错误表明您的系统缺少 CUDA，它是用于在 GPU 上运行深度学习模型的必要组件。要解决这个问题，您可以选择以下两种方法之一：

安装 CUDA 并在 GPU 上运行模型：如果您有 NVIDIA GPU，并且想要在 GPU 上运行模型以获得更快的速度，您需要安装 CUDA。CUDA 是 NVIDIA 提供的用于 GPU 加速计算的平台和编程模型。您可以从 NVIDIA 的官方网站上下载适用于您的系统的 CUDA 工具包，并按照安装说明进行安装。
安装 accelerate 并在 CPU 或硬盘上运行模型：如果您没有 NVIDIA GPU，或者不想安装 CUDA，您可以选择在 CPU 或硬盘上运行模型。
为了在 CPU 上进行动态分布式运行模型，您可以安装 accelerate 库。accelerate 是一个可以将深度学习模型动态分布到 CPU 或硬盘上运行的库。
您可以使用以下命令安装 accelerate：
pip install accelerate

python -m pip install huggingface_hub
huggingface-cli login
"""
nlp = assemble("config.cfg")
doc = nlp("You look gorgeous!")
print(doc.cats)
# {"COMPLIMENT": 1.0, "INSULT": 0.0}

# TypeError: BFloat16 is not supported on MPS