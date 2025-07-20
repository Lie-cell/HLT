# 基于离散单元的中文语音识别系统

---

## 1. 目录结构

HLT/ 

│

├── data/               # 处理后的数据数据 

│     ├── train/

│     ├── data-handler.py

│     ├── aishell_transcript_v0.8.txt

│     └── test/

├── dataset/            # 原始数据集 

├── model/              # 训练得到的模型

├── model-pre/          # 预训练模型 

│     └── hubert-base/

├──  report/           # 实验日志 

│     └── py.py      #计算原始预测cer

├── .gitignore 

├── main_kmeans_ctc.py     # Kmeans_ctc架构主代码

├── main_vqvae_optimize.py     # VQVAE_Transformer架构优化版主代码

├── main_vqvae_origin.py     # VQVAE_Transformer架构主代码

├── README.md 

└── requirements.txt

---

## 2. 环境安装

```bash
# 1. 克隆仓库
git clone https://github.com/nankai-hlt/HLT.git
cd HLT

# 2. 创建并激活虚拟环境（推荐 Python 3.10）
conda create -n hlt2025 python=3.10 -y
conda activate hlt2025

# 3. 安装依赖
pip install -r requirements.txt 
```

## 3.预训练模型下载

实验需要腾讯开源的 **HuBERT-base** 预训练权重，可至[TencentGameMate/chinese_speech_pretrain: chinese speech pretrained models](https://github.com/TencentGameMate/chinese_speech_pretrain)下载

## 4.运行

### 4.1 VQ-VAE 模型

```bash
# 运行原始版本
python main_vqvae_origin.py

# 运行优化版本
python main_vqvae_optimize.py

```

### 4.2 Kmeans_CTC 模型

```bash
python main_kmeans_ctc.py
```

