import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import EsmTokenizer, EsmForSequenceClassification

from data.NRPSDataset import ProSeqGraphDataset
from model.GATContrastiveModel import GATContrastiveModel
from data.preprocess_optimized import main as preprocess_main

def create_label_mapping(df):
    label_to_id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label

def preprocess(df, max_len=500):
    """数据预处理函数"""
    # 检查是否存在'Domain'列，如果不存在则从'A domain Sequence full length'列创建
    if 'Domain' not in df.columns:
        if 'A domain Sequence full length' in df.columns:
            df['Domain'] = df['A domain Sequence full length']
        elif 'A domain core sequence for phylogenetic analysis' in df.columns:
            df['Domain'] = df['A domain core sequence for phylogenetic analysis']
        else:
            raise ValueError("No domain sequence column found in the dataset")
    
    # 检查是否存在'label'列，如果不存在则从'Label'列创建
    if 'label' not in df.columns and 'Label' in df.columns:
        df['label'] = df['Label']
    
    # 过滤掉序列长度为0的样本
    df = df[df['Domain'].str.len() > 0]
    # 限制序列长度
    df['Domain'] = df['Domain'].apply(lambda x: x[:max_len])
    # 添加数据增强：仅对部分样本进行轻微的随机裁剪
    import random
    def random_crop(seq, min_len=200, max_len=500):
        if len(seq) <= min_len:
            return seq
        crop_len = random.randint(min_len, min(len(seq), max_len))
        start = random.randint(0, len(seq) - crop_len)
        return seq[start:start + crop_len]
    # 对30%的样本进行随机裁剪（降低数据增强强度）
    df['Domain'] = df['Domain'].apply(lambda x: random_crop(x) if random.random() > 0.7 else x)
    return df

def main(args):
    pl.seed_everything(42)
    
    # 使用combined_data训练数据
    print("Loading combined training data...")
    integrated_df = pd.read_csv(args.data_path)
    
    # 预处理数据
    integrated_df = preprocess(integrated_df)
    
    # 过滤掉Label为unknown的样本
    integrated_df = integrated_df[integrated_df['Label'] != 'unknown']
    
    print(f"Remaining samples: {len(integrated_df)}")
    print(f"Remaining labels: {len(integrated_df['Label'].unique())}")
    
    # 编码标签
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    integrated_df['encoded_label'] = label_encoder.fit_transform(integrated_df['Label'])
    
    # 分割数据
    from sklearn.model_selection import train_test_split
    
    # 检查每个类别的样本数量
    label_counts = integrated_df['Label'].value_counts()
    min_samples = label_counts.min()
    
    # 如果所有类别都有至少2个样本，使用stratify
    if min_samples >= 2:
        train_val_df, test_df = train_test_split(
            integrated_df, 
            test_size=0.2, 
            random_state=42,
            stratify=integrated_df['Label']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=0.125, 
            random_state=42,
            stratify=train_val_df['Label']
        )
    else:
        # 否则不使用stratify
        train_val_df, test_df = train_test_split(
            integrated_df, 
            test_size=0.2, 
            random_state=42
        )
        
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=0.125, 
            random_state=42
        )
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # 创建底物嵌入
    def create_substrate_embeddings(labels, embedding_dim=256):
        unique_labels = sorted(list(set(labels)))
        label_to_embedding = {}
        
        for label in unique_labels:
            embedding = torch.randn(embedding_dim)
            label_to_embedding[label] = embedding
        
        return label_to_embedding
    
    substrate_embeddings = create_substrate_embeddings(integrated_df['Label'].unique(), embedding_dim=args.substrate_embedding_dim)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # 创建标签映射
    label_to_id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    num_labels = len(label_to_id)
    print(f"Number of labels: {num_labels}")
    print(f"Labels: {sorted(label_to_id.keys())}")
    
    # 重命名列以匹配原有代码
    train_df = train_df.rename(columns={'encoded_label': 'mapped_label'})
    val_df = val_df.rename(columns={'encoded_label': 'mapped_label'})
    test_df = test_df.rename(columns={'encoded_label': 'mapped_label'})
    
    # 确保label列存在且为字符串类型
    train_df['label'] = train_df['label'].astype(str)
    val_df['label'] = val_df['label'].astype(str)
    test_df['label'] = test_df['label'].astype(str)
    
    print("Loading tokenizer and model...")
    tokenizer = EsmTokenizer.from_pretrained(args.model_path)
    esm_model = EsmForSequenceClassification.from_pretrained(
        args.model_path, 
        num_labels=num_labels, 
        output_hidden_states=True
    )
    
    print("Creating datasets...")
    train_dataset = ProSeqGraphDataset(train_df, tokenizer, substrate_embeddings=substrate_embeddings)
    val_dataset = ProSeqGraphDataset(val_df, tokenizer, substrate_embeddings=substrate_embeddings)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    print("Initializing model...")
    model = GATContrastiveModel(
        esm_model=esm_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        finetune_layer=args.finetune_layer,
        gama=args.gamma,
        result_path=args.result_path,
        gat_hidden_dim=args.gat_hidden_dim,
        gat_output_dim=args.gat_output_dim,
        contrastive_projection_dim=args.contrastive_projection_dim,
        temperature=args.temperature,
        substrate_embedding_dim=args.substrate_embedding_dim,
        contrastive_weight=args.contrastive_weight
    )
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='model-{epoch:02d}-{val_acc:.4f}-v2',
        save_top_k=3,
        monitor='val_acc',
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=args.patience,
        mode='max',
        verbose=True
    )
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        enable_checkpointing=True,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0  # 添加梯度裁剪
    )
    
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training completed. Best model saved to {checkpoint_callback.best_model_path}")
    
    torch.save(label_to_id, os.path.join(args.checkpoint_dir, 'label_to_id_v2.pt'))
    torch.save(id_to_label, os.path.join(args.checkpoint_dir, 'id_to_label_v2.pt'))
    print("Label mappings saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/NRPS/data/combined_data.csv')
    parser.add_argument('--model_path', type=str, default='/root/NRPS/model/esm2_t33_650M_UR50D')
    parser.add_argument('--checkpoint_dir', type=str, default='/root/autodl-tmp/checkpoints')
    parser.add_argument('--result_path', type=str, default='/root/autodl-tmp/result/train_result_v2.csv')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--finetune_layer', type=int, default=6)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    
    # GAT参数
    parser.add_argument('--gat_hidden_dim', type=int, default=256)
    parser.add_argument('--gat_output_dim', type=int, default=128)
    
    # 对比学习参数
    parser.add_argument('--contrastive_projection_dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--substrate_embedding_dim', type=int, default=256)
    parser.add_argument('--contrastive_weight', type=float, default=0.1)
    
    args = parser.parse_args()
    
    main(args)
