import pandas as pd
import random
import os

# 配置
input_files = [
    '/root/NRPS/data/combined_data.csv',
    '/root/NRPS/data/integrated_data.csv'
]
output_file = '/root/NRPS/data/test_40.fasta'

# 收集所有数据
alldata = []

for input_file in input_files:
    if os.path.exists(input_file):
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Found {len(df)} records in {input_file}")
        
        # 检查必要的列（处理空格和大小写）
        df.columns = [col.strip() for col in df.columns]
        has_name = any('name' in col.lower() for col in df.columns)
        has_sequence = any('sequence' in col.lower() and 'full' in col.lower() for col in df.columns)
        
        if has_name and has_sequence:
            # 找到实际的列名
            name_col = next(col for col in df.columns if 'name' in col.lower())
            sequence_col = next(col for col in df.columns if 'sequence' in col.lower() and 'full' in col.lower())
            
            # 过滤掉空序列
            df = df[df[sequence_col].notna()]
            df = df[df[sequence_col].str.strip() != '']
            
            # 重命名列以便后续处理
            df = df.rename(columns={name_col: 'Name', sequence_col: 'A domain Sequence full length'})
            alldata.append(df)
            print(f"Added {len(df)} valid records from {input_file}")
        else:
            print(f"Warning: Required columns not found in {input_file}")
            print(f"Available columns: {df.columns.tolist()}")
    else:
        print(f"Warning: {input_file} not found")

if not alldata:
    print("Error: No valid data found")
    exit(1)

# 合并数据
combined_df = pd.concat(alldata, ignore_index=True)
print(f"Total valid records: {len(combined_df)}")

# 确保有足够的数据
if len(combined_df) < 40:
    print(f"Warning: Only {len(combined_df)} valid records found, using all")
    selected = combined_df
else:
    # 随机选择40条
    selected = combined_df.sample(n=40, random_state=42)

# 生成FASTA文件
with open(output_file, 'w') as f:
    for _, row in selected.iterrows():
        # 使用Name作为FASTA头
        header = row['Name']
        # 使用A domain Sequence full length作为序列
        sequence = row['A domain Sequence full length']
        
        f.write(f">{header}\n")
        f.write(f"{sequence}\n")

print(f"Generated {output_file} with {len(selected)} sequences")
print(f"Sample of sequences:")
for i, (_, row) in enumerate(selected.head(5).iterrows()):
    print(f"{i+1}. {row['Name']}")
    print(f"   Sequence length: {len(row['A domain Sequence full length'])}")
    print()

# 验证生成的文件
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        content = f.read()
    print(f"\nGenerated FASTA file content (first 500 chars):")
    print(content[:500] + '...' if len(content) > 500 else content)