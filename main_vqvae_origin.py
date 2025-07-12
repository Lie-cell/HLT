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
    vq_codebook_size = 512
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 验证集划分
    val_ratio = 0.1
    qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # API端点
    rescoring_topk = 5  # 重打分候选数量
    semantic_correction = True  # 是否启用语义纠错

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

import requests
class QwenAPI:
    def __init__(self, api_key = "sk-950071b47e48467e95f95d84383513dd", base_url = config.qwen_base_url):
        if 'DASHSCOPE_API_KEY' in os.environ:
            self.api_key = os.environ['DASHSCOPE_API_KEY']
        else:
            self.api_key = api_key
        self.base_url = base_url
        
    def semantic_correction(self, text):
        """ 语义纠错（如：'我优不知道'->'我也不知道'） """
        payload = {
            "model": "qwen-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个中文语义纠错专家，严格遵循以下规则："
                               "1. 只修改语义错误的词，保持原句结构"
                               "2. 不添加或删除句子内容"
                               "3. 输出格式：{\"corrected_text\": \"修正后文本\"}"
                },
                {
                    "role": "user",
                    "content": f"请纠正语义错误：'{text}'"
                }
            ]
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=5
            )
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"API Error: {e}")
            return text  # 失败时返回原文
    
    def rescoring(self, candidates):
        """ 重打分排序（选择最合理的候选） """
        prompt = "评估下列语音识别候选结果的合理性，返回最自然的选项（只输出序号）：\n"
        for i, cand in enumerate(candidates):
            prompt += f"{i+1}. {cand}\n"
        
        payload = {
            "model": "qwen-max",
            "messages": [
                {"role": "system", "content": "你是一个语音识别质量评估专家"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=5
            )
            choice = int(response.json()['choices'][0]['message']['content'])
            return candidates[choice-1]
        except Exception as e:
            print(f"错误: {e}")
            return candidates[0]  # 失败时返回原始最佳结果

# 端到端模型
class VQVAEASR(nn.Module):
    def __init__(self, hubert_model, vq_codebook_size, embedding_dim, vocab_size , tokenizer):
        super().__init__()
        # HuBERT特征提取器
        self.tokenizer = tokenizer
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
        
        # 嵌入层和输出层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(config.vq_codebook_size, config.embedding_dim).to(config.device)
    def forward(self, input_values, attention_mask, labels=None):
        # 提取HuBERT特征
        with torch.no_grad():
            outputs = self.hubert(input_values, attention_mask=attention_mask)
            features = outputs.last_hidden_state
        
        # VQ量化
        batch_size, seq_len, feat_dim = features.shape
        features = features.reshape(-1, feat_dim)  # 使用reshape代替view
        quantized, vq_loss, encoded = self.vq(features)
        # embedding_layer = nn.Embedding(config.vq_codebook_size, config.embedding_dim).to(config.device)
        encodeed_indices = self.embedding_layer(encoded)
        encodeed_indices = encodeed_indices.reshape(batch_size, seq_len, feat_dim)
        quantized = quantized.reshape(batch_size, seq_len, feat_dim)  # 使用reshape代替view
        
        # 编码器处理
        encoded = self.encoder(encodeed_indices)
        
        # 解码器处理
        if labels is not None:
            # 训练模式
            label_emb = self.embedding(labels)
            
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
            return output, vq_loss, target
        else:
            # 推理模式
            return encoded, vq_loss
    
    def generate(self, encoded, max_length=100, temperature=1.0):
        """自回归生成文本"""
        batch_size = encoded.size(0)
        # 初始化为SOS token
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=config.device) * 1  # 假设1是SOS token
        
        for i in range(max_length):
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
            if (next_token == 2).all():  # 2是EOS token
                break
        
        return generated
    
    def generate_topk(self, encoded, max_length=100, beam_size=config.rescoring_topk, temperature=1.0 ):
        """
        使用 Beam Search 生成 Top-K 个候选序列
        :param encoded: 编码器输出 [batch_size, seq_len, embedding_dim]
        :param max_length: 最大生成长度
        :param beam_size: 返回的候选数量
        :param temperature: 温度控制多样性
        :return: 一个 batch 中每个样本的 top-k 候选列表
        """
        self.eval()
        batch_size = encoded.size(0)
        device = encoded.device

        results = []

        for b in range(batch_size):
            memory = encoded[b:b+1]  # [1, seq_len, embed]
            seq_len = memory.size(1)

            # 初始候选：[(token_ids, log_prob)]
            candidates = [([1], 0.0)]  # 初始为 <sos>

            for step in range(max_length):
                new_candidates = []
                for tokens, score in candidates:
                    if tokens[-1] == 2:  # EOS
                        new_candidates.append((tokens, score))
                        continue

                    # 构造输入
                    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                    input_emb = self.embedding(input_ids)

                    # 解码器前向
                    output = self.decoder(tgt=input_emb, memory=memory)
                    # print(output.shape)
                    logits = self.fc_out(output[:, -1, :]) / temperature
                    log_probs = F.log_softmax(logits, dim=-1)
                    # print(log_probs.shape)

                    # 取 top beam_size 个
                    top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                    for i in range(beam_size):
                        new_tokens = tokens + [top_indices[0, i].item()]
                        new_score = score + top_log_probs[0, i].item()
                        new_candidates.append((new_tokens, new_score))

                # 保留 top beam_size 个候选
                new_candidates.sort(key=lambda x: x[1], reverse=True)
                candidates = new_candidates[:beam_size]

                # 所有候选都结束
                if all(cand[0][-1] == 2 for cand in candidates):
                    break
            # print(candidates)
            # 解码为文本
            decoded_candidates = [self.tokenizer.decode(tokens) for tokens, _ in candidates]
            results.append(decoded_candidates)

        return results

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
            outputs, vq_loss, targets = model(
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
            
            asr_loss = F.cross_entropy(outputs, targets, ignore_index=tokenizer.pad_token_id)
            
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
        val_loss, val_cer = validate(model, tokenizer, val_loader ,epoch)
        
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
            outputs, vq_loss, targets = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels
            )
            
            outputs = outputs.reshape(-1, outputs.size(-1))  # 使用reshape代替view
            targets = targets.reshape(-1)  # 使用reshape代替view
            
            # 忽略pad token的损失
            loss_mask = targets != tokenizer.pad_token_id
            outputs = outputs[loss_mask]
            targets = targets[loss_mask]
            
            asr_loss = F.cross_entropy(outputs, targets, ignore_index=tokenizer.pad_token_id)
            total_loss += (0.5*asr_loss + 0.5*vq_loss).item()
            
            # 生成预测
            encoded, _ = model(input_values, attention_mask)
            pred_ids = model.generate(encoded, max_length=config.max_text_len)
            
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
                    # char_errors = sum(1 for a, b in zip(ref, hyp) if a != b)
                    total_cer += jiwer.cer(ref , hyp)
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

    qwen = QwenAPI(base_url = config.qwen_base_url)

    with torch.no_grad():
        for batch in test_loader:
            input_values = batch["input_values"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            texts = batch["text"]
            
            # 生成预测
            encoded, _ = model(input_values, attention_mask)
            # print(encoded)
            pred_texts = model.generate_topk(encoded, max_length=config.max_text_len)
            
            # 解码预测

            final_preds = []
            for i in range(len(pred_texts)):
                candidates = pred_texts[i]
                print(candidates )
                best_candidate = qwen.rescoring(candidates)
                final_preds.append(best_candidate)
            # print(final_preds)
            if config.semantic_correction:
                corrected_texts = []
                for text in final_preds:
                    corrected = qwen.semantic_correction(text)
                    # 解析API返回的JSON格式
                    print(corrected)
                    if corrected.startswith('{'):
                        try:
                            corrected = json.loads(corrected)['corrected_text']
                        except Exception as e:
                            print(e)
                            pass
                    corrected_texts.append(corrected)
                pred_texts = corrected_texts

            # 保存结果
            print(texts[0],"test",pred_texts[0])
            for i in range(len(texts)):
                ref = texts[i]
                hyp = pred_texts[i]
                orig_pred = final_preds[i]
                
                results.append(f"Reference: {ref}\nCorrected_predicted: {hyp}\nOrigin_predicted: {orig_pred}\n\n")
                
                if ref and hyp:
                    # CER：字符错误率
                    # print(ref,"123",hyp)
                    # char_errors = sum(1 for a, b in zip(ref, hyp) if a != b)
                    total_cer += jiwer.cer(ref , hyp)
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
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer
    )
    model.to(config.device)
    
    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    # val_loss, val_cer, val_wer = validate(model, tokenizer, val_loader ,10)
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