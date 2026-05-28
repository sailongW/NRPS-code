import pandas as pd
import os

# 读取Excel文件
df = pd.read_excel('data/data.xlsx')

# 清理列名（去除空格）
df.columns = [col.strip() for col in df.columns]

# 提取必要的列
processed_df = df[['Name', 'Label', 'A domain Sequence full length']]

# 重命名列
processed_df.columns = ['ID', 'Label', 'Domain']

# 保存为CSV文件
output_path = 'data/train_data.csv'
processed_df.to_csv(output_path, index=False)

print(f"数据处理完成！")
print(f"处理后的数据包含 {len(processed_df)} 条记录")
print(f"保存路径: {output_path}")
print("\n前5条记录:")
print(processed_df.head())
