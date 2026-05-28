#!/usr/bin/env python3
"""
从UniProt数据库搜集蓝细菌的NRPS A-domain数据并整合成数据集
"""

import requests
import pandas as pd
import time
import re
from tqdm import tqdm

class UniProtDataCollector:
    def __init__(self):
        self.base_url = "https://rest.uniprot.org/uniprotkb/search"
        self.headers = {
            "Accept": "application/json"
        }
    def search_cyanobacteria_nrps(self, query="cyanobacteria AND adenylation domain AND nonribosomal peptide synthetase", limit=1000):
        """
        搜索蓝细菌的NRPS A-domain数据
        """
        params = {
            "query": query,
            "limit": limit,
            "fields": "accession,id,protein_name,organism_name,sequence,comments,keywords,features"
        }
        
        response = requests.get(self.base_url, headers=self.headers, params=params)
        response.raise_for_status()
        
        return response.json()
    def extract_adenylation_domain(self, sequence, features):
        """
        从序列中提取A-domain
        """
        # 从features中提取A-domain位置
        a_domain_start = None
        a_domain_end = None
        
        if features:
            for feature in features:
                if feature.get('type') == 'Domain' and 'adenylation' in feature.get('description', '').lower():
                    locations = feature.get('locations', [])
                    if locations:
                        for location in locations:
                            start = location.get('start', {}).get('value')
                            end = location.get('end', {}).get('value')
                            if start and end:
                                a_domain_start = int(start)
                                a_domain_end = int(end)
                                break
                    if a_domain_start and a_domain_end:
                        break
        
        # 如果找到A-domain位置，提取序列
        if a_domain_start and a_domain_end:
            return sequence[a_domain_start-1:a_domain_end]  # Python字符串索引从0开始
        else:
            # 如果没有找到A-domain位置，返回完整序列
            return sequence
    def process_data(self, data):
        """
        处理搜索结果，提取有用信息
        """
        records = []
        
        for entry in tqdm(data.get('results', []), desc="Processing entries"):
            accession = entry.get('primaryAccession', '')
            id_ = entry.get('uniProtkbId', '')
            protein_name = entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
            organism = entry.get('organism', {}).get('scientificName', '')
            sequence = entry.get('sequence', {}).get('value', '')
            comments = entry.get('comments', [])
            keywords = entry.get('keywords', [])
            features = entry.get('features', [])
            
            # 提取A-domain序列
            a_domain_sequence = self.extract_adenylation_domain(sequence, features)
            
            # 尝试从注释中提取底物信息
            substrate = self.extract_substrate(comments, keywords)
            
            # 构建记录
            record = {
                "Accession": accession,
                "ID": id_,
                "Protein Name": protein_name,
                "Organism": organism,
                "A domain Sequence full length": a_domain_sequence,
                "Label": substrate,
                "Source": "UniProt"
            }
            
            records.append(record)
        
        return records
    def extract_substrate(self, comments, keywords):
        """
        从注释中提取底物信息
        """
        # 从comments中提取底物信息
        for comment in comments:
            if isinstance(comment, dict):
                if comment.get('type') == 'FUNCTION':
                    text = comment.get('text', [])
                    for line in text:
                        line_str = line.get('value', '')
                        # 查找底物相关信息
                        substrate_match = re.search(r'substrate[:\s]+([\w\-\/]+)', line_str, re.IGNORECASE)
                        if substrate_match:
                            return substrate_match.group(1)
                        
                        # 查找氨基酸相关信息
                        amino_acid_match = re.search(r'amino acid[:\s]+([\w\-\/]+)', line_str, re.IGNORECASE)
                        if amino_acid_match:
                            return amino_acid_match.group(1)
        
        # 从keywords中提取底物信息
        for keyword in keywords:
            if isinstance(keyword, dict):
                keyword_name = keyword.get('name', '').lower()
                if 'substrate' in keyword_name or 'amino acid' in keyword_name:
                    return keyword.get('name', '')
        
        # 如果没有找到底物信息，返回"unknown"
        return "unknown"
    def save_to_csv(self, records, output_file="data/cyanobacteria_nrps_data.csv"):
        """
        将数据保存为CSV文件
        """
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        print(f"Total records: {len(df)}")
        return df
    def run(self, output_file="data/cyanobacteria_nrps_data.csv"):
        """
        运行整个数据搜集流程
        """
        print("Searching for cyanobacteria NRPS A-domain data...")
        data = self.search_cyanobacteria_nrps()
        
        print("Processing data...")
        records = self.process_data(data)
        
        print("Saving data...")
        df = self.save_to_csv(records, output_file)
        
        return df

def integrate_with_existing_data(new_data_file="data/cyanobacteria_nrps_data.csv", existing_data_file="data/data.csv", output_file="data/integrated_data.csv"):
    """
    将新搜集的数据与现有数据整合
    """
    print("Loading existing data...")
    existing_df = pd.read_csv(existing_data_file)
    
    print("Loading new data...")
    new_df = pd.read_csv(new_data_file)
    
    # 确保列名一致
    new_df = new_df[existing_df.columns.intersection(new_df.columns)]
    
    # 添加缺失的列
    for col in existing_df.columns:
        if col not in new_df.columns:
            new_df[col] = ""
    
    # 整合数据
    integrated_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # 去重
    integrated_df = integrated_df.drop_duplicates(subset=["A domain Sequence full length"])
    
    # 保存整合后的数据
    integrated_df.to_csv(output_file, index=False)
    print(f"Integrated data saved to {output_file}")
    print(f"Original records: {len(existing_df)}")
    print(f"New records: {len(new_df)}")
    print(f"Integrated records: {len(integrated_df)}")
    print(f"Duplicates removed: {len(existing_df) + len(new_df) - len(integrated_df)}")
    
    return integrated_df

if __name__ == "__main__":
    # 初始化数据收集器
    collector = UniProtDataCollector()
    
    # 搜集蓝细菌NRPS A-domain数据
    collector.run()
    
    # 整合数据
    integrate_with_existing_data()
