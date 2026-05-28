import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import EsmTokenizer


def clean_data(df):
    """
    清洗数据，处理缺失值和重复数据
    """
    # 移除重复行
    df = df.drop_duplicates()
    
    # 处理缺失值
    df = df.dropna(subset=['A domain Sequence full length', 'Label'])
    
    # 标准化标签格式（转为小写）
    df['Label'] = df['Label'].str.lower()
    
    # 移除标签中的特殊字符和空格
    df['Label'] = df['Label'].apply(lambda x: re.sub(r'[^a-zA-Z0-9-/_]', '', x))
    
    return df


def filter_sequences(df, max_length=500):
    """
    过滤过长的序列
    """
    # 计算序列长度
    df['seq_length'] = df['A domain Sequence full length'].apply(len)
    
    # 过滤长度在合理范围内的序列
    df = df[(df['seq_length'] > 50) & (df['seq_length'] <= max_length)]
    
    return df


def encode_labels(df):
    """
    编码标签
    """
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['Label'])
    
    return df, label_encoder


def split_data(df, test_size=0.2, val_size=0.125, random_state=42):
    """
    分割数据为训练集、验证集和测试集
    """
    # 统计每个类别的样本数量
    label_counts = df['encoded_label'].value_counts()
    
    # 过滤掉样本数量少于2的类别
    valid_labels = label_counts[label_counts >= 2].index
    df_filtered = df[df['encoded_label'].isin(valid_labels)]
    
    print(f"Filtered out {len(df) - len(df_filtered)} samples from classes with fewer than 2 members")
    print(f"Remaining samples: {len(df_filtered)}")
    
    # 首先分割为训练集和测试集
    train_val_df, test_df = train_test_split(
        df_filtered, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df_filtered['encoded_label']
    )
    
    # 然后从训练集中分割出验证集
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size, 
        random_state=random_state,
        stratify=train_val_df['encoded_label']
    )
    
    return train_df, val_df, test_df


def preprocess_sequence(sequence, tokenizer, max_length=500):
    """
    预处理序列，使用ESM tokenizer进行编码
    """
    # 确保序列长度不超过max_length
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    
    # 使用tokenizer编码
    encoding = tokenizer(
        sequence, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )
    
    return encoding


def create_substrate_embeddings(labels, embedding_dim=256):
    """
    创建底物嵌入
    """
    unique_labels = sorted(list(set(labels)))
    label_to_embedding = {}
    
    for label in unique_labels:
        # 为每个标签创建一个随机嵌入
        # 在实际应用中，可以使用更复杂的嵌入方法
        embedding = torch.randn(embedding_dim)
        label_to_embedding[label] = embedding
    
    return label_to_embedding


def main(data_path, max_length=500, test_size=0.2, val_size=0.125, random_state=42):
    """
    主预处理函数
    """
    # 读取数据
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Original data shape: {df.shape}")
    
    # 清洗数据
    print("Cleaning data...")
    df = clean_data(df)
    print(f"After cleaning: {df.shape}")
    
    # 过滤序列
    print("Filtering sequences...")
    df = filter_sequences(df, max_length=max_length)
    print(f"After filtering: {df.shape}")
    
    # 编码标签
    print("Encoding labels...")
    df, label_encoder = encode_labels(df)
    print(f"Number of unique labels: {len(label_encoder.classes_)}")
    print(f"Labels: {sorted(label_encoder.classes_)}")
    
    # 分割数据
    print("Splitting data...")
    train_df, val_df, test_df = split_data(df, test_size=test_size, val_size=val_size, random_state=random_state)
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # 创建底物嵌入
    print("Creating substrate embeddings...")
    substrate_embeddings = create_substrate_embeddings(df['Label'])
    print(f"Number of substrate embeddings: {len(substrate_embeddings)}")
    
    # 保存预处理后的数据
    print("Saving preprocessed data...")
    train_df.to_csv('data/train_data.csv', index=False)
    val_df.to_csv('data/val_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)
    
    # 保存标签编码器
    import pickle
    with open('data/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # 保存底物嵌入
    torch.save(substrate_embeddings, 'data/substrate_embeddings.pt')
    
    print("Preprocessing completed!")
    return train_df, val_df, test_df, label_encoder, substrate_embeddings


if __name__ == "__main__":
    # 测试预处理函数
    data_path = 'data/data.csv'
    main(data_path)
