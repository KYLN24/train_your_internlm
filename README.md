# 从头训练一个 InternLM

> 本文档部分内容来自[InternLM](https://github.com/InternLM/InternLM)。
> 
> 本文档仅起到介绍 InternLM 训练框架的作用。完全依照本文档的步骤操作并不能让你获得一个和 [internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b) 能力相当的预训练语言模型。

本文档介绍如何使用 InternLM 官方训练代码在 [The Pile](https://arxiv.org/abs/2101.00027) 数据集上完成具有 70 亿参数的 InternLM 模型的预训练。

首先，请将本示例项目 clone 到本地。

``` sh
git clone git@github.com:KYLN24/train_your_internlm.git --recursive
```

上述命令会将本项目和 [InternLM](https://github.com/InternLM/InternLM) 及其依赖项全部 clone 至当前目录下的 `train_your_internlm` 文件夹中。

## 🖥️ 配置训练环境

参考 [InternLM/doc/install.md](InternLM/doc/install.md#环境准备) 配置训练环境。

你需要一台运行 Linux 系统的计算机。请自行安装 [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-11-7-1-download-archive) 和 GCC 10.2。

### 【可选】安装 Conda

``` sh
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && sh ./Miniconda3-latest-Linux-x86_64.sh
```

### 【可选】创建虚拟环境

``` sh
conda create --name internlm-env python=3.10 -y
conda activate internlm-env
```

### 安装 InternLM 依赖

``` sh
cd InternLM/internlm
pip install -r requirements/torch.txt 
pip install -r requirements/runtime.txt 
```

### 安装 flash-attention (version v1.0.5)
```bash
cd InternLM/third_party/flash-attention
python setup.py install
cd ./csrc
cd fused_dense_lib && pip install -v .
cd ../xentropy && pip install -v .
cd ../rotary && pip install -v .
cd ../layer_norm && pip install -v .
cd ../../../../
```

### 安装 Apex (version 23.05)
```bash
cd InternLM/third_party/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../../
```

## 🗃️ 准备数据集

完整的 The Pile 数据集大小为 825GB。为便于实验，这里以 [ola13/small-the_pile](https://huggingface.co/datasets/ola13/small-the_pile) 数据集为例，介绍数据的处理。

你可以从 Hugging Face 上获取该数据集。首先安装 Hugging Face Datasets

``` sh
pip install datasets
```

然后运行 [data/get_dataset.py](data/get_dataset.py)。

``` sh
python -u data/get_dataset.py
```

接下来使用 InternLM 提供的工具 [InternLM/tools/tokenizer.py](InternLM/tools/tokenizer.py) 对数据集进行预处理 tokenization。`tokenizer.py` 支持处理 `.txt`、`.json`或`.jsonl` 格式的数据。详情请参考 [InternLM/doc/usage.md](InternLM/doc/usage.md#数据准备-微调)

`get_dataset.py` 已经将数据集以 `jsonl` 格式保存在了 `data/train` 和 `data/train` 下。我们可以直接使用 `tokenizer.py` 来处理。

```
python InternLM/tools/tokenizer.py --text_input_path=./data/train/small-the_pile.train.jsonl --bin_output_path=./data/train/en/small-the_pile.train.bin

python InternLM/tools/tokenizer.py --text_input_path=./data/val/small-the_pile.val.jsonl --bin_output_path=./data/val/en/small-the_pile.val.bin
```

请注意，生成的数据文件 (\*.bin和\*.bin.meta) 应当按照类别放置于 `cn`、`en` 或 `code` 文件夹中。

## 🛠️ 自定义训练配置与超参数

修改 [7b_pretrain_config.py](7b_pretrain_config.py)。你可以自定义训练步数、批大小、数据集路径、模型保存路径、学习率与其他优化器参数、和多卡并行配置等。更多配置项请参考 [InternLM/configs/7B_sft.py](InternLM/configs/7B_sft.py)

## 🚀 启动训练

使用 Slurm 启动训练

``` sh
srun -p llm_o -N 1 -n 8 --ntasks-per-node=8 --gpus-per-task=1 python -u ./InternLM/train.py --config=./7b_pretrain_config.py
``` 

使用 `torchrun` 启动训练

```
torchrun --nproc_per_node=8 ./InternLM/train.py --config=./7b_pretrain_config.py`
```

使用 TensorBoard 监控训练详情

```
tensorboard --logdir=logs
```

使用本文档提供的配置文件在 8x A800 80GB 上进行训练，TGS 应该达到 4103 左右，TFLOPS 达到 183 左右。训练性能请参考 [InternLM/doc/train_performance.md](InternLM/doc/train_performance.md)。

## 🤗 转换为 Hugging Face 格式

为了方便后续工作，我们可以使用 [InternLM/tools/transformers/convert2hf.py](InternLM/tools/transformers/convert2hf.py) 将保存的 checkpoint 转换为 Hugging Face 格式。详见 [InternLM/tools/transformers/README-zh-Hans.md](InternLM/tools/transformers/README-zh-Hans.md)。

```
cd InternLM
python -u tools/transformers/convert2hf.py --src_folder=../ckpts/7b_pretrain/<select one ckpt> --tgt_folder=../ckpts/7b_pretrain/hf --tokenizer tools/V7_sft.model
cd ..
```

预览一下模型的生成效果

```
python -u test.py
```
