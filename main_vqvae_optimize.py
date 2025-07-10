import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import HubertModel
import torch.nn.functional as F
import re
from collections import Counter
from tqdm import tqdm
import jiwer

# 配置参数
class Config:
    # 数据参数
    data_dir = "data"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    transcript_path = os.path.join(data_dir, "aishell_transcript_v0.8.txt")
    
    # 模型参数
    hubert_model_path = "model-pre/hubert-base"
    vq_codebook_size = 1024
    embedding_dim = 768
    max_audio_len = 96000  # 6s for 16kHz
    max_text_len = 50
    
    # 训练参数
    batch_size = 16
    num_workers = 8
    lr = 1e-4
    epochs = 16
    save_dir = "model"
    report_dir = "report"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 验证集划分
    val_ratio = 0.1

config = Config()
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.report_dir, exist_ok=True)

# 自定义字符级分词器
class CharTokenizer:
    def __init__(self, texts):
        # 构建字符词汇表
        all_chars = ''.join(texts)
        char_counter = Counter(all_chars)
        # 排序并创建词汇表
        chars = sorted(char_counter.keys())
        self.vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.vocab.update({char: idx+4 for idx, char in enumerate(chars)})
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab['<pad>']
        self.sos_token_id = self.vocab['<sos>']
        self.eos_token_id = self.vocab['<eos>']
        self.unk_token_id = self.vocab['<unk>']
    
    def encode(self, text):
        """将文本转换为ID序列"""
        ids = [self.vocab.get(char, self.unk_token_id) for char in text]
        return [self.sos_token_id] + ids + [self.eos_token_id]
    
    def decode(self, ids):
        """将ID序列转换回文本"""
        tokens = []
        for id_ in ids:
            if id_ == self.eos_token_id:
                break
            if id_ not in (self.pad_token_id, self.sos_token_id, self.eos_token_id):
                tokens.append(self.inv_vocab.get(id_, '<unk>'))
        return ''.join(tokens)
    
    def batch_encode(self, texts):
        """批量编码文本"""
        encoded_texts = [self.encode(text) for text in texts]
        max_len = max(len(seq) for seq in encoded_texts)
        
        padded = []
        masks = []
        for seq in encoded_texts:
            pad_len = max_len - len(seq)
            padded.append(seq + [self.pad_token_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
        
        return torch.tensor(padded), torch.tensor(masks)

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
        # print(self.data[0])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path, text = self.data[idx]
        
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.squeeze()
        
        # 确保采样率正确
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # 截取或填充音频
        if len(waveform) > config.max_audio_len:
            waveform = waveform[:config.max_audio_len]
        elif len(waveform) < config.max_audio_len:
            pad_len = config.max_audio_len - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        
        return {
            "input_values": waveform,
            "attention_mask": torch.ones_like(waveform),
            "text": text
        }

# 向量量化层
class VectorQuantizerEMA(nn.Module):
    def __init__(self, codebook_size, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # 初始化码本
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.codebook.weight.data.normal_()
        
        # EMA统计量
        self.register_buffer('_ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('_ema_w', self.codebook.weight.data.clone())

    def forward(self, inputs):
        # 计算输入与码本的距离
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.codebook.weight.t()))
        
        # 获取最近邻编码
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.codebook_size).float()
        
        # 量化向量
        quantized = torch.matmul(encodings, self.codebook.weight)
        
        # EMA更新码本
        if self.training:
            # 更新EMA统计量
            self._ema_cluster_size = (self.decay * self._ema_cluster_size 
                                     + (1 - self.decay) * encodings.sum(0))
            
            # 拉普拉斯平滑防止码本坍塌
            n = self._ema_cluster_size.sum()
            smoothed_size = ((self._ema_cluster_size + self.epsilon)
                           / (n + self.codebook_size * self.epsilon) * n)
            
            # 更新权重和
            dw = torch.matmul(encodings.t(), inputs)
            self._ema_w = self.decay * self._ema_w + (1 - self.decay) * dw
            
            # 应用平滑后的码本
            self.codebook.weight.data = self._ema_w / smoothed_size.unsqueeze(1)
        
        # 直通估计器
        quantized = inputs + (quantized - inputs).detach()
        
        # 计算commitment损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        return quantized, loss, encoding_indices

class LeadCharPredictor(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim=512, num_layers=2):
        super(LeadCharPredictor, self).__init__()
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size * 2)  # 预测前两个字符
        )

    def forward(self, x):
        # x: [B, T, D]
        _, h_n = self.gru(x)  # h_n: [2*num_layers, B, H]
        # 取最后一层的正向和反向输出拼接
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=-1)  # [B, 2H]
        out = self.fc(h_last)  # [B, vocab_size * 2]
        return out
        
# 端到端模型
class VQVAEASR(nn.Module):
    def __init__(self, hubert_model, vq_codebook_size, embedding_dim, vocab_size):
        super().__init__()
        # HuBERT特征提取器
        self.hubert = hubert_model
        for param in self.hubert.parameters():
            param.requires_grad = False
        
        # VQ层
        self.vq = VectorQuantizerEMA(
            codebook_size=vq_codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=0.25,
            decay=0.99
        )
        
        # Transformer编码器-解码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=1536,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=1536,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )

        self.lead_char_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, vocab_size * 2)  # 预测前两个汉字
        )


        # self.fusion_gate = nn.Sequential(
        #     nn.Linear(embedding_dim * 2, embedding_dim),
        #     nn.Sigmoid()  # 生成0-1的融合权重
        # )
        
        # 嵌入层和输出层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.vocab_size = vocab_size
    
    def forward(self, input_values, attention_mask, labels=None):
        # 提取HuBERT特征
        with torch.no_grad():
            outputs = self.hubert(input_values, attention_mask=attention_mask)
            features = outputs.last_hidden_state
        
        # VQ量化
        batch_size, seq_len, feat_dim = features.shape
        features = features.reshape(-1, feat_dim)  # 使用reshape代替view
        quantized, vq_loss, encodedd = self.vq(features)
        embedding_layer = nn.Embedding(config.vq_codebook_size, config.embedding_dim).to(config.device)
        encodeed_indices = embedding_layer(encodedd)
        encodeed_indices = encodeed_indices.reshape(batch_size, seq_len, feat_dim)
        # quantized = quantized.reshape(batch_size, seq_len, feat_dim)  # 使用reshape代替view
        lead_logits = self.lead_char_predictor(encodeed_indices.means(dim = 1)).reshape(batch_size, 2, self.vocab_size)  # [B, 2, vocab_size]
        lead_emb = self.embedding(torch.argmax(lead_logits, dim=-1))
        concatenated_features = torch.cat([lead_emb, encodeed_indices], dim=1)
        ## 压缩声学特征为上下文向量 [B, D]
        # acoustic_context = encodeed_indices.mean(dim=1)  
        ## 广播前导嵌入至声学序列长度 [B, T, D]
        # lead_emb = lead_emb.view(lead_emb.size(0), -1)  # 形状变为 (16, 2 * 768)

        # 创建目标形状的零张量
        # seq_len = encodeed_indices.size(1)  # 获取目标序列长度
        # lead_emb_expanded = torch.zeros((lead_emb.size(0), seq_len, 768), device=lead_emb.device)

        # # 填充目标张量
        # lead_emb_expanded[:, :2, :] = lead_emb.view(lead_emb.size(0), 2, 768) 
        # ## 门控融合 [B, T, D]
        # gate = self.fusion_gate(torch.cat([encodeed_indices, lead_emb_expanded], dim=-1))
        # fused_features = gate * encodeed_indices + (1 - gate) * lead_emb_expanded
        
        # 步骤4：将融合特征输入Transformer
        encoded = self.encoder(concatenated_features)  # 此处输入已包含前导信息
        
        # # 编码器处理
        # encoded = self.encoder(encodeed_indices)
        
        # 解码器处理
        if labels is not None:
            # 训练模式
            label_emb = self.embedding(labels)
            lead_loss = F.cross_entropy(
                lead_logits.view(-1, self.vocab_size), 
                labels[:, 1:3].reshape(-1)  # 取真实标签的前两字
            )
        
            # 3. 声学-文本对齐损失
            ## 确保融合特征与文本嵌入空间一致
            # align_loss = F.mse_loss(acoustic_context, lead_emb.detach().mean(dim=1))
                
            # 创建目标序列（右移）
            decoder_input = label_emb[:, :-1, :]
            target = labels[:, 1:]
            
            # 创建掩码
            tgt_mask = torch.triu(torch.ones(decoder_input.size(1), decoder_input.size(1)), 
                                 diagonal=1).bool().to(config.device)
            tgt_key_padding_mask = (target == 0)  # 假设0是pad token
            
            # 解码器前向传播
            decoder_output = self.decoder(
                tgt=decoder_input,
                memory=encoded,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            # 输出层
            output = self.fc_out(decoder_output)
            return output, vq_loss, target ,lead_loss ,lead_logits
        # , align_loss
        else:
            # 推理模式
            return encoded, vq_loss , lead_logits
    
    def generate(self, encoded, lead_logits ,max_length=100, temperature=1.0 ):
        """自回归生成文本"""
        batch_size = encoded.size(0)
        # 初始化为SOS token
        lead_ids = torch.argmax(lead_logits, dim=-1)  # [B, 2]
    
        # 初始化Decoder输入
        generated = torch.cat([
        torch.ones(batch_size, 1).to(config.device).long() * 1,
        lead_ids.long()
        ], dim=1)
        # generated = torch.ones(batch_size, 1, dtype=torch.long, device=config.device) * 1  # 假设1是SOS token
        
        for i in range(3,max_length):
            # 嵌入
            input_emb = self.embedding(generated)
            
            # 解码器前向传播
            decoder_output = self.decoder(
                tgt=input_emb,
                memory=encoded
            )
            
            # 输出层
            logits = self.fc_out(decoder_output[:, -1, :]) / temperature
            probs = F.softmax(logits, dim=-1)
            
            # 采样下一个token
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # 如果生成了EOS token，停止生成
            if (next_token == 2).all():  # 假设2是EOS token
                break
        
        return generated
    

# 训练函数
from tqdm import tqdm
def calculate_weights(e, E):
    # 计算各个阶段的 epoch 范围
    first_stage_end = E * 6 // 16
    second_stage_end = E * 10 // 16  # 6/16 + 4/16 = 10/16

    if e <= first_stage_end:
        # 第一阶段：从 0.95 线性衰减到 0.5
        vq_weight = 0.95 - 0.45 * (e - 1) / (first_stage_end - 1)
    elif e <= second_stage_end:
        # 第二阶段：保持 0.5
        vq_weight = 0.5
    else:
        # 第三阶段：从 0.5 线性衰减到 0.05
        vq_weight = 0.5 - 0.45 * (e - second_stage_end) / (E - second_stage_end)
    
    asr_weight = 1 - vq_weight
    return vq_weight, asr_weight

def train(model, tokenizer, train_loader, val_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    start_time = time.time()
    best_val_loss = float('inf')
    
    # 创建报告文件
    report_path = os.path.join(config.report_dir, "vqvae_report.txt")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("VQVAE ASR Training Report\n")
        report_file.write(f"Start Time: {time.ctime(start_time)}\n")
        report_file.write(f"Device: {config.device}\n")
        report_file.write(f"Batch Size: {config.batch_size}\n")
        report_file.write(f"Learning Rate: {config.lr}\n")
        report_file.write(f"Epochs: {config.epochs}\n")
        report_file.write(f"Vocab Size: {tokenizer.vocab_size}\n\n")
        report_file.write("Epoch\tTrain Loss\tVal Loss\tVal CER\tTime\n")
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        step_count = 0

        vq_weight, asr_weight = calculate_weights(epoch + 1, config.epochs)
        # 使用 tqdm 创建进度条
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", unit="batch")
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_values = batch["input_values"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            texts = batch["text"]
            
            # 编码标签
            labels, _ = tokenizer.batch_encode(texts)
            labels = labels.to(config.device)
            
            # 前向传播
            outputs, vq_loss, targets ,lead_loss ,_= model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # 计算ASR损失
            outputs = outputs.reshape(-1, outputs.size(-1))  # 使用reshape代替view
            targets = targets.reshape(-1)  # 使用reshape代替view
            
            # 忽略pad token的损失
            loss_mask = targets != tokenizer.pad_token_id
            outputs = outputs[loss_mask]
            targets = targets[loss_mask]
            
            asr_loss = F.cross_entropy(outputs, targets, ignore_index=tokenizer.pad_token_id)*0.75+lead_loss*0.25
            
            # 总损失
            total_loss = vq_weight * vq_loss + asr_weight * asr_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            step_count += 1
            
            # 更新进度条描述信息
            train_loader.set_postfix({"Train Loss": f"{total_loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / step_count
        
        # 验证
        val_loss, val_cer= validate(model, tokenizer, val_loader ,epoch)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config.save_dir, "vqvae_asr_model.pt")
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
def validate(model, tokenizer, val_loader, epoch):
    model.eval()
    total_loss = 0
    total_cer = 0
    count = 0
    
    with torch.no_grad():
        val_loader = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", unit="batch")
        for batch in val_loader:
            input_values = batch["input_values"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            texts = batch["text"]
            
            # 编码标签
            labels, _ = tokenizer.batch_encode(texts)
            labels = labels.to(config.device)
            
            # # 计算损失
            # outputs, vq_loss, targets , _ ,lead_logits = model(
            #     input_values=input_values,
            #     attention_mask=attention_mask,
            #     labels=labels
            # )
            
            # outputs = outputs.reshape(-1, outputs.size(-1))  # 使用reshape代替view
            # targets = targets.reshape(-1)  # 使用reshape代替view
            
            # # 忽略pad token的损失
            # loss_mask = targets != tokenizer.pad_token_id
            # outputs = outputs[loss_mask]
            # targets = targets[loss_mask]
            
            # asr_loss = F.cross_entropy(outputs, targets, ignore_index=tokenizer.pad_token_id)
            # total_loss += (0.5*asr_loss + 0.5*vq_loss).item()
            
            # 生成预测
            encoded, _ , lead_logits = model(input_values, attention_mask)
            pred_ids = model.generate(encoded, max_length=config.max_text_len,lead_logits = lead_logits)

            
            # 解码预测
            pred_texts = []
            for ids in pred_ids:
                text = tokenizer.decode(ids.tolist())
                pred_texts.append(text)
            
            # 计算指标
            # print(texts[0],"val0",pred_texts[0])
            for i in range(len(texts)):
                ref = texts[i]
                hyp = pred_texts[i]
                
                if ref and hyp:
                    # CER：字符错误率
                    #char_errors = sum(1 for a, b in zip(ref, hyp) if a != b)
                    total_cer += jiwer.cer(ref , hyp)#char_errors / max(len(ref), 1)
                    count += 1
            val_loader.set_postfix({"Val Loss": f"{total_loss:.4f}"})
    
    avg_loss = total_loss / len(val_loader)
    avg_cer = total_cer / count if count else 1.0
    
    return avg_loss, avg_cer

# 测试函数
def test(model, tokenizer, test_loader):
    model.eval()
    total_cer = 0
    count = 0
    results = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_values = batch["input_values"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            texts = batch["text"]
            
            # 生成预测
            encoded, _ ,lead_logits= model(input_values, attention_mask)
            pred_ids = model.generate(encoded, max_length=config.max_text_len,lead_logits = lead_logits)
            
            # 解码预测
            pred_texts = []
            for ids in pred_ids:
                text = tokenizer.decode(ids.tolist())
                pred_texts.append(text)
            
            # 保存结果
            # print(texts[0],"test",pred_texts[0])
            for i in range(len(texts)):
                ref = texts[i]
                hyp = pred_texts[i]
                
                results.append(f"Reference: {ref}\nPredicted: {hyp}\n\n")
                
                if ref and hyp:
                    # CER：字符错误率
                    # print(ref,"123",hyp)
                    #char_errors = sum(1 for a, b in zip(ref, hyp) if a != b)
                    total_cer += jiwer.cer(ref , hyp)#char_errors / max(len(ref), 1)
                    count += 1
    
    # 计算最终指标
    final_cer = total_cer / count if count else 1.0
    
    # 保存测试结果
    result_path = os.path.join(config.report_dir, "vqvae_test_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Final CER: {final_cer:.4f}\n")
        f.write("Detailed Results:\n\n")
        f.writelines(results)
    
    # 更新训练报告
    report_path = os.path.join(config.report_dir, "vqvae_report.txt")
    with open(report_path, "a", encoding="utf-8") as report_file:
        report_file.write("\nTest Results:\n")
        report_file.write(f"CER: {final_cer:.4f}\n")
    
    return final_cer

# 主函数
def main():
    # 加载数据集（首先加载训练数据用于构建词汇表）
    train_dataset = AISHELLDataset(config.train_dir, config.transcript_path)
    
    # 初始化分词器
    tokenizer = CharTokenizer(train_dataset.texts)
    print(f"Built tokenizer with vocab size: {tokenizer.vocab_size}")
    
    # 重新加载完整训练数据集
    train_dataset = AISHELLDataset(config.train_dir, config.transcript_path)
    # val_size1 = int(len(train_dataset) * 0.99)
    # train_size1 = len(train_dataset) - val_size1
    # train_dataset, _ = random_split(
    #     train_dataset, [train_size1, val_size1]
    # )

    # 划分训练集和验证集
    val_size = int(len(train_dataset) * config.val_ratio)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 测试集
    test_dataset = AISHELLDataset(config.test_dir, config.transcript_path)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # 加载HuBERT模型
    hubert_model = HubertModel.from_pretrained(config.hubert_model_path)
    hubert_model.to(config.device)
    hubert_model.eval()
    
    # 创建端到端模型
    model = VQVAEASR(
        hubert_model=hubert_model,
        vq_codebook_size=config.vq_codebook_size,
        embedding_dim=config.embedding_dim,
        vocab_size=tokenizer.vocab_size
    )
    model.to(config.device)
    
    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    # val_loss, val_cer = validate(model, tokenizer, val_loader ,10)
    # 训练模型
    model = train(model, tokenizer, train_loader, val_loader, optimizer, scheduler)
    # checkpoint = torch.load(os.path.join(config.save_dir, "vqvae_asr_model.pt"), map_location=config.device)
    # tokenizer.vocab = checkpoint['tokenizer_vocab']
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试模型
    test_cer = test(model, tokenizer, test_loader)
    print(f"Test CER: {test_cer:.4f}")

if __name__ == "__main__":
    # print(config.device)
    main()