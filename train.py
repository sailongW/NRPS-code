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
from data.preprocess import preprocess, train_eval_split

def create_label_mapping(df):
    label_to_id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label

def prepare_data(data_path):
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    print(f"Original data shape: {df.shape}")
    
    if "A domain Sequence full length" in df.columns:
        df = df.rename(columns={"A domain Sequence full length": "sequence"})
    if " Name" in df.columns:
        df = df.rename(columns={" Name": "name"})
    if "Label" in df.columns:
        df = df.rename(columns={"Label": "label"})
    
    df = preprocess(df, max_len=500)
    print(f"After preprocessing: {df.shape}")
    
    if "sequence" in df.columns:
        df = df.rename(columns={"sequence": "Domain"})
    
    return df

def main(args):
    pl.seed_everything(42)
    
    df = prepare_data(args.data_path)
    
    train_df, val_df = train_eval_split(df, val_ratio=args.val_ratio, random_state=42)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    label_to_id, id_to_label = create_label_mapping(train_df)
    num_labels = len(label_to_id)
    print(f"Number of labels: {num_labels}")
    print(f"Labels: {sorted(label_to_id.keys())}")
    
    train_df['mapped_label'] = train_df['label'].map(label_to_id)
    val_df['mapped_label'] = val_df['label'].map(label_to_id)
    
    print("Loading tokenizer and model...")
    tokenizer = EsmTokenizer.from_pretrained(args.model_path)
    esm_model = EsmForSequenceClassification.from_pretrained(
        args.model_path, 
        num_labels=num_labels, 
        output_hidden_states=True
    )
    
    print("Creating datasets...")
    train_dataset = ProSeqGraphDataset(train_df, tokenizer)
    val_dataset = ProSeqGraphDataset(val_df, tokenizer)
    
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
        result_path=args.result_path
    )
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='model-{epoch:02d}-{val_acc:.4f}',
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
        precision="16-mixed"  # 使用混合精度训练，提升训练速度
    )
    
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training completed. Best model saved to {checkpoint_callback.best_model_path}")
    
    torch.save(label_to_id, os.path.join(args.checkpoint_dir, 'label_to_id.pt'))
    torch.save(id_to_label, os.path.join(args.checkpoint_dir, 'id_to_label.pt'))
    print("Label mappings saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/NRPS/data/combined_data.csv')
    parser.add_argument('--model_path', type=str, default='/root/NRPS/model/esm2_t33_650M_UR50D')
    parser.add_argument('--checkpoint_dir', type=str, default='/root/autodl-tmp/checkpoints')
    parser.add_argument('--result_path', type=str, default='/root/autodl-tmp/result/train_result.csv')
    parser.add_argument('--batch_size', type=int, default=16)  # 增加批量大小，充分利用32GB GPU内存
    parser.add_argument('--max_epochs', type=int, default=100)  # 增加训练轮数
    parser.add_argument('--lr', type=float, default=1e-5)  # 适当增加学习率，加速收敛
    parser.add_argument('--weight_decay', type=float, default=0.001)  # 增加权重衰减，减少过拟合
    parser.add_argument('--finetune_layer', type=int, default=5)  # 增加微调层数，更好利用预训练模型
    parser.add_argument('--gamma', type=float, default=0.9)  # 调整学习率衰减率
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=15)  # 增加早停耐心值
    parser.add_argument('--num_workers', type=int, default=8)  # 增加工作线程数，利用12 vCPU
    args = parser.parse_args()
    
    main(args)