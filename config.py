class Config:
    # 数据集配置
    data_dir = "data"
    transcript_path = "data/aishell_transcript_v0.8.txt"
    train_ratio = 0.9
    sample_rate = 16000
    
    # 模型参数
    num_embeddings = 512     # VQVAE码本大小
    embedding_dim = 128      # 离散单元维度
    hidden_dim = 256         # Transformer隐藏层维度
    nhead = 8                # Transformer头数
    num_layers = 6           # Transformer层数
    dropout = 0.1
    
    # 训练参数
    batch_size = 16
    num_epochs = 20
    lr = 1e-4
    weight_decay = 1e-5
    clip_grad = 5.0
    log_interval = 50
    
    # 路径配置
    pretrain_path = "model-pre/hubert-base"
    save_dir = "model"
    report_dir = "report"