import os
import warnings
import logging

# Suppress warnings as early as possible (before importing libraries that emit warnings)
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
from transformers import EsmTokenizer, EsmForSequenceClassification
from transformers.utils import logging as hf_logging

# Reduce third-party library logging verbosity
hf_logging.set_verbosity_error()
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

import pytorch_lightning as pl

from data.NRPSDataset import ProSeqGraphDataset
from model.GATContrastiveModel import GATContrastiveModel

MODEL_PATH = "/root/NRPS/model/esm2_t33_650M_UR50D"
DEFAULT_CHECKPOINT = "/root/autodl-tmp/checkpoints/model-epoch=00-val_acc=0.4215.ckpt"
BATCH_SIZE = 16  # 减小批量大小，避免内存不足

def main(args):
    # 读取结构域数据
    val_df = pd.read_csv(args.inference_dataset)
    
    # 检查数据是否为空
    if val_df.empty:
        raise Exception("No domains found. Please check your input sequence.")

    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    
    # 加载标签映射
    label_to_id_path = os.path.join('/root/autodl-tmp', 'checkpoints', 'label_to_id_v2.pt')
    if os.path.exists(label_to_id_path):
        label_to_id = torch.load(label_to_id_path)
        num_labels = len(label_to_id)
        print(f"Loaded {num_labels} labels from {label_to_id_path}")
    else:
        # 尝试加载旧的标签映射
        old_label_to_id_path = os.path.join('/root/autodl-tmp', 'checkpoints', 'label_to_id.pt')
        if os.path.exists(old_label_to_id_path):
            label_to_id = torch.load(old_label_to_id_path)
            num_labels = len(label_to_id)
            print(f"Loaded {num_labels} labels from old {old_label_to_id_path}")
        else:
            # 检查checkpoints目录中的所有文件，寻找可能的标签映射文件
            import glob
            checkpoints_dir = os.path.join('/root/autodl-tmp', 'checkpoints')
            if os.path.exists(checkpoints_dir):
                label_files = glob.glob(os.path.join(checkpoints_dir, '*label*.pt'))
                if label_files:
                    # 尝试加载第一个找到的标签文件
                    label_file = label_files[0]
                    try:
                        label_to_id = torch.load(label_file)
                        num_labels = len(label_to_id)
                        print(f"Loaded {num_labels} labels from {label_file}")
                    except Exception as e:
                        print(f"Error loading label file {label_file}: {e}")
                        # 如果加载失败，使用默认值
                        num_labels = 473  # 训练时使用的标签数量
                        print(f"Using default {num_labels} labels (training value)")
                else:
                    # 如果没有找到标签文件，使用训练时的标签数量
                    num_labels = 473  # 训练时使用的标签数量
                    print(f"Using default {num_labels} labels (training value)")
            else:
                # 如果checkpoints目录不存在，使用训练时的标签数量
                num_labels = 473  # 训练时使用的标签数量
                print(f"Using default {num_labels} labels (training value)")

    # 加载tokenizer
    tokenizer = EsmTokenizer.from_pretrained(MODEL_PATH)
    
    # 创建数据集
    val_dataset = ProSeqGraphDataset(val_df, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 加载ESM模型
    esm_model = EsmForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH, 
        num_labels=num_labels, 
        output_hidden_states=True
    )
    
    # 加载GAT模型，使用与训练时相同的参数
    model = GATContrastiveModel(
        esm_model=esm_model, 
        lr=1e-5, 
        weight_decay=0.01, 
        finetune_layer=6, 
        gama=0.9, 
        result_path=args.result_path,
        gat_hidden_dim=256,  # 与训练时相同
        gat_output_dim=128,   # 与训练时相同
        contrastive_projection_dim=256,  # 与训练时相同
        substrate_embedding_dim=256  # 与训练时相同
    )
    
    # 加载训练好的模型检查点
    checkpoint_path = args.checkpoint_path if hasattr(args, 'checkpoint_path') and args.checkpoint_path else DEFAULT_CHECKPOINT
    if not os.path.exists(checkpoint_path):
        # 尝试查找其他模型文件
        import glob
        checkpoints_dir = os.path.join('/root/autodl-tmp', 'checkpoints')
        model_files = glob.glob(os.path.join(checkpoints_dir, '*.ckpt'))
        if model_files:
            checkpoint_path = max(model_files, key=os.path.getctime)
            print(f"Using found model: {checkpoint_path}")
        else:
            raise Exception(f"No model checkpoint found at {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    # 加载模型时使用map_location为cpu，并设置weights_only=True以减少内存使用
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    # 加载完成后删除checkpoint以释放内存
    del checkpoint
    import gc
    gc.collect()
    print("Model loaded successfully")
    
    # 创建trainer
    trainer = pl.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False
    )
    
    # 运行预测
    print("Running prediction...")
    trainer.test(model=model, dataloaders=val_loader)
    print("Prediction completed")
    # 预测完成后释放模型内存
    del model
    del esm_model
    gc.collect()
    
    # 处理结果
    if os.path.exists(args.result_path):
        part_result = pd.read_csv(args.result_path)
        final_result = pd.DataFrame(columns=["ID", "Domain", "Top-1(score)", "Top-2(score)", "Top-3(score)"])
        
        for i in range(len(part_result)):
            domain = part_result.iloc[i]["Domain"]
            top1 = part_result.iloc[i]["Top-1(score)"]
            top2 = part_result.iloc[i]["Top-2(score)"]
            top3 = part_result.iloc[i]["Top-3(score)"]
            id = val_df.iloc[i]["ID"] if i < len(val_df) else f"domain_{i}"
            final_result.loc[len(final_result)] = [id, domain, top1, top2, top3]
        
        final_result.to_csv(args.result_path, index=False)
        print(f"Results saved to: {args.result_path}")
    else:
        raise Exception("Prediction failed: no result file generated")

if __name__ == "__main__":
    import time
    T1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_dataset', type=str, default="hmm/result/domains.csv")
    parser.add_argument('--result_path', type=str, default="/root/autodl-tmp/result/result_gat_contrastive.csv")
    parser.add_argument('--checkpoint_path', type=str, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    main(args)
    T2 = time.time()
    print(f"Time: {(T2-T1):.2f}sec")
