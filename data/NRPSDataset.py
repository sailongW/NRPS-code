import os
import torch
from torch.utils.data import Dataset

os.environ['OPENBLAS_NUM_THREADS'] = '1'

class ProSeqDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_len = df["Domain"].apply(lambda x: len(x)).max()
        self.seqs = list(self.df['Domain'])
        self.inputs = self.tokenizer(list(self.df['Domain']), return_tensors="pt", 
                                    padding="max_length", truncation=True, max_length=self.max_len)
        self.mapped_label = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mapped_label is None:
            label = 0
        else:
            label = self.mapped_label[idx]
        seqs = self.seqs[idx]
        input_ids = self.inputs["input_ids"][idx].squeeze()
        attention_mask = self.inputs["attention_mask"][idx].squeeze()
        return seqs, input_ids, attention_mask, label

class ProSeqGraphDataset(Dataset):
    def __init__(self, df, tokenizer, substrate_embeddings=None):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_len = df["Domain"].apply(lambda x: len(x)).max()
        self.seqs = list(self.df['Domain'])
        self.inputs = self.tokenizer(list(self.df['Domain']), return_tensors="pt", 
                                    padding="max_length", truncation=True, max_length=self.max_len)
        self.substrate_embeddings = substrate_embeddings
        if 'mapped_label' in self.df.columns:
            self.mapped_label = self.df['mapped_label'].values.tolist()
        else:
            self.mapped_label = None
        # 保存原始标签
        if 'label' in self.df.columns:
            self.labels = self.df['label'].values.tolist()
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mapped_label is None:
            label = 0
        else:
            label = self.mapped_label[idx]
        seqs = self.seqs[idx]
        input_ids = self.inputs["input_ids"][idx].squeeze()
        attention_mask = self.inputs["attention_mask"][idx].squeeze()
        
        # 创建邻接矩阵
        seq_length = len(seqs)
        adj_matrix = torch.zeros(self.max_len, self.max_len)
        # 填充对角线和相邻位置
        for i in range(seq_length):
            adj_matrix[i, i] = 1.0  # 对角线
            if i > 0:
                adj_matrix[i, i-1] = 1.0  # 左边
            if i < seq_length - 1:
                adj_matrix[i, i+1] = 1.0  # 右边
        
        # 获取底物嵌入（如果有）
        if self.substrate_embeddings is not None:
            # 使用原始标签获取底物嵌入
            if self.labels is not None:
                original_label = self.labels[idx]
                # 确保original_label是字符串类型
                if isinstance(original_label, str):
                    if original_label in self.substrate_embeddings:
                        substrate_emb = self.substrate_embeddings[original_label]
                    else:
                        # 如果没有对应的底物嵌入，使用零向量
                        substrate_emb = torch.zeros(256)  # 嵌入维度为256
                else:
                    # 如果original_label不是字符串，使用零向量
                    substrate_emb = torch.zeros(256)
            else:
                # 如果没有原始标签，使用零向量
                substrate_emb = torch.zeros(256)
        else:
            # 如果没有底物嵌入，使用零向量
            substrate_emb = torch.zeros(256)
        
        return seqs, input_ids, attention_mask, adj_matrix, substrate_emb, label
