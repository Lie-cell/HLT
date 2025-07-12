import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import HubertModel, Wav2Vec2Processor
import torch.nn.functional as F
import re
from collections import Counter
from tqdm import tqdm
import jiwer
from sklearn.cluster import KMeans
import joblib
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
# from ctcdecode import CTCBeamDecoder
# from nemo.collections.asr.modules import BeamSearchDecoderWithLM
from fast_ctc_decode import beam_search

# 配置参数
class Config:
    # 数据参数
    data_dir = "data"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    transcript_path = os.path.join(data_dir, "aishell_transcript_v0.8.txt")
    
    # 模型参数
    hubert_model_path = "model-pre/hubert-base"
    kmeans_n_clusters = 512  # K-means聚类中心数
    embedding_dim = 768
    max_audio_len = 96000  # 6s for 16kHz
    max_text_len = 50
    
    # 训练参数
    batch_size = 16
    num_workers = 8
    lr = 1e-5
    epochs = 16
    save_dir = "model"
    report_dir = "report"
    kmeans_model_path = "model/kmeans_model.pkl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 验证集划分
    val_ratio = 0.1
    
    # CTC参数
    blank_token = 0  # CTC空白符
    beam_width = 50  # Beam search宽度
    lm_path = None  # 语言模型路径（可选）

config = Config()
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.report_dir, exist_ok=True)

# 自定义字符级分词器（针对CTC优化）
class CTCTokenizer:
    def __init__(self, texts):
        # 构建字符词汇表，特别处理空白符
        all_chars = ''.join(texts)
        char_counter = Counter(all_chars)
        
        # 排序并创建词汇表
        chars = sorted(char_counter.keys())
        self.vocab = {'<blank>': 0, '<unk>': 1}  # CTC专用词汇表
        self.vocab.update({char: idx+2 for idx, char in enumerate(chars)})
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.blank_token_id = self.vocab['<blank>']
        self.unk_token_id = self.vocab['<unk>']
    
    def encode(self, text):
        """将文本转换为ID序列（不添加特殊符号）"""
        return [self.vocab.get(char, self.unk_token_id) for char in text]
    
    def decode(self, ids):
        """将ID序列转换回文本（忽略空白符）"""
        tokens = []
        for id_ in ids:
            if id_ == self.blank_token_id:
                continue
            tokens.append(self.inv_vocab.get(id_, '<unk>'))
        return ''.join(tokens)
    
    def batch_encode(self, texts):
        """批量编码文本，返回padded序列和长度"""
        encoded_texts = [self.encode(text) for text in texts]
        lengths = torch.tensor([len(seq) for seq in encoded_texts], dtype=torch.long)
        max_len = max(lengths)
        
        padded = []
        for seq in encoded_texts:
            padded.append(seq + [-1] * (max_len - len(seq)))
        
        return torch.tensor(padded), lengths

# 数据加载和处理
class AISHELLDataset(Dataset):
    def __init__(self, audio_dir, transcript_path, max_audio_len=160000):
        self.audio_dir = audio_dir
        
        # 加载转录本
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.data = []
        self.audio_files = []
        self.texts = []
        
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                audio_id, text = parts
                audio_path = os.path.join(audio_dir, f"{audio_id}.wav")
                if os.path.exists(audio_path):
                    # 清理文本：去除空格和标点
                    clean_text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
                    if clean_text:
                        self.data.append((audio_path, clean_text))
                        self.audio_files.append(audio_path)
                        self.texts.append(clean_text)
        
        print(f"Loaded {len(self.data)} samples from {audio_dir}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path, text = self.data[idx]
        
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.squeeze()
        
        # 确保采样率正确
        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # 截取或填充音频
        if len(waveform) > config.max_audio_len:
            waveform = waveform[:config.max_audio_len]
        elif len(waveform) < config.max_audio_len:
            pad_len = config.max_audio_len - len(waveform)
            waveform = F.pad(waveform, (0, pad_len))
        
        return {
            "input_values": waveform,
            "attention_mask": torch.ones_like(waveform),
            "text": text
        }

# K-means特征量化器
class KMeansQuantizer:
    def __init__(self, n_clusters=512):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.is_trained = False
    
    def train(self, features):
        """使用K-means聚类训练量化器"""
        print(f"Training K-means with {len(features)} samples...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=0, 
            n_init=10,
            verbose=1
        )
        self.kmeans.fit(features)
        self.is_trained = True
        print("K-means training completed.")
    
    def predict(self, features):
        """预测特征对应的聚类索引"""
        if not self.is_trained:
            raise RuntimeError("K-means model not trained yet")
        return self.kmeans.predict(features)
    
    def predict_centroids(self, features):
        """预测特征对应的聚类中心向量"""
        if not self.is_trained:
            raise RuntimeError("K-means model not trained yet")
        
        # 首先获取每个特征的聚类索引
        cluster_indices = self.kmeans.predict(features)
        
        # 然后根据索引获取对应的聚类中心向量
        centroids = self.kmeans.cluster_centers_[cluster_indices]
        
        return centroids
    
    def save(self, path):
        """保存K-means模型"""
        if self.kmeans is not None:
            joblib.dump(self.kmeans, path)
            print(f"K-means model saved to {path}")
    
    def load(self, path):
        """加载K-means模型"""
        self.kmeans = joblib.load(path)
        self.is_trained = True
        print(f"K-means model loaded from {path}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x
    
class FocalCTCLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, blank=0, reduction='mean'):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.blank = blank
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # 计算基础CTC损失
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # 计算Focal Loss的调制因子
        probs = torch.exp(-loss)
        focal_factor = (1 - probs) ** self.gamma
        
        # 应用Focal Loss
        focal_loss = self.alpha * focal_factor * loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
            
# 端到端模型
class KMeansCTCASR(nn.Module):
    def __init__(self, hubert_model, quantizer, vocab_size ,tokenizer):
        super().__init__()
        # HuBERT特征提取器
        self.hubert = hubert_model
        self.tokenizer = tokenizer
        for param in self.hubert.parameters():
            param.requires_grad = False
        
        # K-means量化器
        self.quantizer = quantizer
        # self.ctc_loss = FocalCTCLoss(alpha=0.25, gamma=2.0, blank=tokenizer.blank_token_id)
        # 编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=8,
                dim_feedforward=1536,  # 增加维度以处理稀疏性
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2  # 增加层数以提高性能
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, vocab_size)
        )
        # nn.init.constant_(self.output_layer.bias[tokenizer.blank_token_id], -5.0)
        # CTC损失函数
        # nn.init.xavier_uniform_(self.output_layer.weight)
        # nn.init.constant_(self.output_layer.bias[tokenizer.blank_token_id], -20.0)
        self.ctc_loss = nn.CTCLoss(blank=config.blank_token, reduction='mean')
        self.positional_encoding = PositionalEncoding(config.embedding_dim)
        self.eembedding = nn.Embedding(config.kmeans_n_clusters, config.embedding_dim)
        # CTC解码器
        # self.ctc_decoder = CTCBeamDecoder(
        #     list(self.quantizer.kmeans.labels_),
        #     model_path=config.lm_path,
        #     alpha=0.5,
        #     beta=1.5,
        #     cutoff_top_n=40,
        #     cutoff_prob=1.0,
        #     beam_width=config.beam_width,
        #     num_processes=4,
        #     blank_id=config.blank_token,
        #     log_probs_input=True
        # )
    
    def forward(self, input_values, attention_mask, labels=None, label_lengths=None):
        # 提取HuBERT特征
        with torch.no_grad():
            outputs = self.hubert(input_values, attention_mask=attention_mask)
            features = outputs.last_hidden_state
        
        # 量化
        # print("Feature std per time step:", features.std(dim=1).mean().item())
        batch_size, seq_len, feat_dim = features.shape
        features_flat = features.reshape(-1, feat_dim).cpu().numpy()
        quant_indices = self.quantizer.predict(features_flat)
        quant_indices = torch.tensor(quant_indices, dtype=torch.long, device=config.device)
        quant_indices = quant_indices.reshape(batch_size, seq_len)
        # quant_indices = quant_indices + (features - features.detach())
        # print(quant_indices)
        # values, counts = quant_indices.flatten().unique(return_counts=True)
        # topk = torch.topk(counts, k=5)
        # print("Most frequent quantized cluster IDs:", values[topk.indices], topk.values)
        # 嵌入层
        embedded = self.eembedding(quant_indices)
        # embedded = embedded + (features - features.detach())
        print(embedded)
        # 编码器处理
        encoded = self.encoder(embedded)
        # print(encoded)
        # 输出层
        logits = self.output_layer(self.positional_encoding(encoded))
        print(torch.argmax(logits, dim=-1))
        log_probs = F.log_softmax(logits, dim=-1)
        # print(log_probs)
        
        if labels is not None:
            # 计算CTC损失
            # print(labels)
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=config.device)
            valid_labels = labels[labels != -1]  # 移除填充的占位符
            loss = self.ctc_loss(
                log_probs.permute(1, 0, 2),  # (T, N, C)
                valid_labels,                # (N, S)
                input_lengths,               # (N)
                label_lengths                # (N)
            )
            aux_head = nn.Linear(config.embedding_dim, self.tokenizer.vocab_size).to(config.device)
            aux_loss = F.cross_entropy(aux_head(encoded[:,0]), labels[:,0])  # 首帧分类
            total_loss = loss + 0.3 * aux_loss  # 避免空白符主导
            return total_loss, log_probs
        else:
            return log_probs
    
    def decode(self, log_probs):
        probs = torch.exp(log_probs)
        log_probs_np = probs.detach().cpu().numpy()
        print(log_probs_np[0])
        decoded_texts = []
        vocab_list = list(self.tokenizer.vocab.keys())
        # vocab_list[0] = ' '
        for i in range(log_probs_np.shape[0]):  # 逐样本处理
            text, _ = beam_search(
                log_probs_np[i],                # 单样本概率矩阵 [T, vocab_size]
                alphabet=vocab_list,
                beam_size=config.beam_width,
                # blank_id=config.blank_token,
                # beam_cut_threshold = 0.0     
            )
            if isinstance(text, str):
                text = text.replace('<unk>', '')  # 删除 '<unk>'
            decoded_texts.append(text)
        return decoded_texts

# 训练函数
def train(model, tokenizer, train_loader, val_loader, optimizer, scheduler):
    model.train()
    best_val_loss = float('inf')
    
    # 创建报告文件
    report_path = os.path.join(config.report_dir, "kmeans_ctc_report.txt")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("K-means + CTC ASR Training Report\n")
        report_file.write(f"Start Time: {time.ctime(time.time())}\n")
        report_file.write(f"Device: {config.device}\n")
        report_file.write(f"Batch Size: {config.batch_size}\n")
        report_file.write(f"Learning Rate: {config.lr}\n")
        report_file.write(f"Epochs: {config.epochs}\n")
        report_file.write(f"Vocab Size: {tokenizer.vocab_size}\n\n")
        report_file.write("Epoch\tTrain Loss\tVal Loss\tVal CER\tTime\n")
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        total_samples = 0
        
        # 使用 tqdm 创建进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", unit="batch")
        
        for batch in train_bar:
            optimizer.zero_grad()
            
            input_values = batch["input_values"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            texts = batch["text"]
            # print(texts)
            
            # 编码标签
            labels, label_lengths = tokenizer.batch_encode(texts)
            labels = labels.to(config.device)
            label_lengths = label_lengths.to(config.device)
            
            # 前向传播
            loss, _ = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
                label_lengths=label_lengths
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            
            # 更新统计信息
            train_loss += loss.item() * input_values.size(0)
            total_samples += input_values.size(0)
            
            # 更新进度条
            train_bar.set_postfix({"Train Loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / total_samples
        
        # 验证
        val_loss, val_cer = validate(model, tokenizer, val_loader)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config.save_dir, "kmeans_ctc_asr_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_vocab': tokenizer.vocab,
                'config': config.__dict__
            }, model_path)
        
        # 记录训练信息
        epoch_time = time.time() - epoch_start
        with open(report_path, "a", encoding="utf-8") as report_file:
            report_file.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{val_loss:.4f}\t{val_cer:.4f}\t{epoch_time:.1f}s\n")
        
        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | CER: {val_cer:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    # 记录总训练时间
    total_time = time.time() - start_time
    with open(report_path, "a", encoding="utf-8") as report_file:
        report_file.write(f"\nTotal Training Time: {total_time:.1f} seconds\n")
    
    return model

# 验证函数
def validate(model, tokenizer, val_loader):
    model.eval()
    total_loss = 0
    total_cer = 0
    total_samples = 0
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Validating", unit="batch")
        for batch in val_bar:
            input_values = batch["input_values"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            texts = batch["text"]
            
            # 编码标签
            labels, label_lengths = tokenizer.batch_encode(texts)
            labels = labels.to(config.device)
            label_lengths = label_lengths.to(config.device)
            
            # 计算损失
            loss, log_probs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
                label_lengths=label_lengths
            )
            
            # 解码预测
            pred_texts = model.decode(log_probs)
            
            # 计算指标
            print(texts[0],"val0",pred_texts[0])
            for i in range(len(texts)):
                ref = texts[i]
                hyp = pred_texts[i]
                total_samples += 1
                if ref and hyp:
                    total_cer += jiwer.cer(ref, hyp)
                    
            
            # 更新统计信息
            total_loss += loss.item() * input_values.size(0)
            
            # 更新进度条
            val_bar.set_postfix({"Val Loss": f"{loss.item():.4f}", "CER": f"{total_cer/total_samples:.4f}"})
    
    avg_loss = total_loss / len(val_loader.dataset)
    avg_cer = total_cer / total_samples if total_samples else 1.0
    
    return avg_loss, avg_cer

# 测试函数
def test(model, tokenizer, test_loader):
    model.eval()
    total_cer = 0
    total_samples = 0
    results = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing", unit="batch")
        for batch in test_bar:
            input_values = batch["input_values"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            texts = batch["text"]
            
            # 前向传播
            log_probs = model(input_values, attention_mask)
            
            # 解码预测
            pred_texts = model.decode(log_probs)
            
            # 保存结果
            for i in range(len(texts)):
                ref = texts[i]
                hyp = pred_texts[i]
                
                results.append(f"Reference: {ref}\nPredicted: {hyp}\n\n")
                
                if ref and hyp:
                    total_cer += jiwer.cer(ref, hyp)
                    total_samples += 1
            
            # 更新进度条
            test_bar.set_postfix({"CER": f"{total_cer/total_samples:.4f}"})
    
    # 计算最终指标
    final_cer = total_cer / total_samples if total_samples else 1.0
    
    # 保存测试结果
    result_path = os.path.join(config.report_dir, "kmeans_ctc_test_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Final CER: {final_cer:.4f}\n")
        f.write("Detailed Results:\n\n")
        f.writelines(results)
    
    # 更新训练报告
    report_path = os.path.join(config.report_dir, "kmeans_ctc_report.txt")
    with open(report_path, "a", encoding="utf-8") as report_file:
        report_file.write("\nTest Results:\n")
        report_file.write(f"CER: {final_cer:.4f}\n")
    
    return final_cer

# 提取HuBERT特征
def extract_hubert_features(dataset, hubert_model):
    all_features = []
    
    print("Extracting HuBERT features for K-means training...")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        input_values = sample["input_values"].unsqueeze(0).to(config.device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(config.device)
        
        with torch.no_grad():
            outputs = hubert_model(input_values, attention_mask=attention_mask)
            features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        
        all_features.append(features)
    
    # 合并所有特征
    all_features = np.vstack(all_features)
    print(f"Extracted {all_features.shape[0]} features of dimension {all_features.shape[1]}")
    return all_features

def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    texts = [item["text"] for item in batch]
        
    # 填充音频
    input_values = pad_sequence(input_values, batch_first=True, padding_value=0.0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0)
        
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "text": texts
    }

# 主函数
def main():
    global start_time
    start_time = time.time()
    
    # 加载数据集
    train_dataset = AISHELLDataset(config.train_dir, config.transcript_path)
    
    # 初始化分词器
    tokenizer = CTCTokenizer(train_dataset.texts)
    print(f"Built CTC tokenizer with vocab size: {tokenizer.vocab_size}")
    
    # 加载HuBERT模型
    hubert_model = HubertModel.from_pretrained(config.hubert_model_path)
    hubert_model.to(config.device)
    hubert_model.eval()
    
    # 训练或加载K-means量化器
    quantizer = KMeansQuantizer(n_clusters=config.kmeans_n_clusters)
    val_size1 = int(len(train_dataset) * 0.75)
    train_size1 = len(train_dataset) - val_size1
    train_dataset, _ = random_split(
        train_dataset, [train_size1, val_size1]
    )
    if os.path.exists(config.kmeans_model_path):
        quantizer.load(config.kmeans_model_path)
    else:
        # 提取特征并训练K-means
        features = extract_hubert_features(train_dataset, hubert_model)
        quantizer.train(features)
        quantizer.save(config.kmeans_model_path)
    
    # 重新加载完整训练数据集
    # train_dataset = AISHELLDataset(config.train_dir, config.transcript_path)
    # 划分训练集和验证集
    val_size = int(len(train_dataset) * config.val_ratio)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 测试集
    test_dataset = AISHELLDataset(config.test_dir, config.transcript_path)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    # 创建端到端模型
    model = KMeansCTCASR(
        hubert_model=hubert_model,
        quantizer=quantizer,
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer
    )
    model.to(config.device)
    
    # 优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.lr,
        weight_decay=0.01  # 权重衰减防止过拟合
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 训练模型
    model = train(model, tokenizer, train_loader, val_loader, optimizer, scheduler)
    
    # 测试模型
    test_cer = test(model, tokenizer, test_loader)
    print(f"Test CER: {test_cer:.4f}")

if __name__ == "__main__":
    # help(beam_search)
    main()