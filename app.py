from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import subprocess
import sys
import pandas as pd
import json
import datetime
import shutil

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = '/root/autodl-tmp/uploads'
app.config['RESULT_FOLDER'] = '/root/autodl-tmp/result'
app.config['HISTORY_FOLDER'] = '/root/autodl-tmp/history'

# 全局变量用于跟踪预测进度
prediction_progress = {
    'progress': 0,
    'status': '准备开始预测...'
}

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['HISTORY_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    # 重置进度
    global prediction_progress
    prediction_progress = {
        'progress': 0,
        'status': '准备开始预测...'
    }
    return render_template('index.html')

@app.route('/progress')
def progress():
    global prediction_progress
    return prediction_progress

@app.route('/predict', methods=['POST'])
def predict():
    global prediction_progress
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        # 获取应用根目录的绝对路径
        app_root = os.path.dirname(os.path.abspath(__file__))
        
        # 更新进度
        prediction_progress['progress'] = 10
        prediction_progress['status'] = '保存上传文件...'
        
        # 保存上传的文件
        upload_folder = os.path.join(app_root, app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        print(f"File saved to: {file_path}")
        
        # 更新进度
        prediction_progress['progress'] = 20
        prediction_progress['status'] = '检查HMM扫描工具...'

        # 导入hmmscan_wrapper模块
        sys.path.insert(0, os.path.join(app_root, 'hmm'))
        try:
            from hmmscan_wrapper import check_hmmscan_available, check_hmm_model_available, run_hmmscan, extract_domains
            hmmscan_available = check_hmmscan_available()
            hmm_model_available = check_hmm_model_available()
        except ImportError:
            print("Warning: Could not import hmmscan_wrapper, using fallback method")
            hmmscan_available = False
            hmm_model_available = False

        # 更新进度
        prediction_progress['progress'] = 25
        prediction_progress['status'] = '准备HMM扫描...'

        # 创建结果目录
        hmm_result_dir = os.path.join(app_root, 'hmm', 'result')
        os.makedirs(hmm_result_dir, exist_ok=True)

        # 如果hmmscan和模型都可用，运行HMM扫描
        if hmmscan_available and hmm_model_available:
            print("Running HMM scan using hmmscan_wrapper...")
            domtblout_path = os.path.join(hmm_result_dir, 'domains.txt')
            csv_output_path = os.path.join(hmm_result_dir, 'domains.csv')

            try:
                # 运行hmmscan
                run_hmmscan(file_path, domtblout_path, hmm_result_dir)

                # 提取域序列
                extract_domains(file_path, domtblout_path, csv_output_path)
            except Exception as e:
                print(f"Error during HMM scan: {e}")
                # 尝试使用备选方案
                print("Falling back to alternative domain extraction method...")
                # 运行gen_domains.py作为备选方案
                gen_script = os.path.join(app_root, 'hmm', 'gen_domains.py')
                gen_result = subprocess.run(
                    [sys.executable, gen_script, '--fasta_path', file_path], 
                    check=False, 
                    capture_output=True, 
                    text=True,
                    cwd=app_root
                )
                print(f"Alternative method return code: {gen_result.returncode}")
                print(f"Alternative method stdout: {gen_result.stdout}")
                print(f"Alternative method stderr: {gen_result.stderr}")
        else:
            # 如果hmmscan不可用，使用gen_domains.py的备选方案
            print("hmmscan not available, using alternative domain extraction method...")
            # 运行gen_domains.py作为备选方案
            gen_script = os.path.join(app_root, 'hmm', 'gen_domains.py')
            gen_result = subprocess.run(
                [sys.executable, gen_script, '--fasta_path', file_path], 
                check=False, 
                capture_output=True, 
                text=True,
                cwd=app_root
            )
            print(f"Alternative method return code: {gen_result.returncode}")
            print(f"Alternative method stdout: {gen_result.stdout}")
            print(f"Alternative method stderr: {gen_result.stderr}")

        # 检查是否存在已有的HMM结果
        existing_domains = os.path.join(hmm_result_dir, 'domains.csv')
        if os.path.exists(existing_domains):
            print(f"Found domain results at: {existing_domains}")
            # 检查文件大小
            if os.path.getsize(existing_domains) > 0:
                print("Domain file is not empty, proceeding with prediction...")
            else:
                print("Domain file is empty, trying to generate domains...")
                # 再次尝试运行gen_domains.py
                gen_script = os.path.join(app_root, 'hmm', 'gen_domains.py')
                gen_result = subprocess.run(
                    [sys.executable, gen_script, '--fasta_path', file_path], 
                    check=False, 
                    capture_output=True, 
                    text=True,
                    cwd=app_root
                )
                print(f"Second attempt return code: {gen_result.returncode}")
        
        # 更新进度
        prediction_progress['progress'] = 30
        prediction_progress['status'] = '解析结构域结果...'
        
        # 解析结果
        print("Parsing domain results...")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python executable: {sys.executable}")
        
        # 运行脚本并捕获输出
        parse_script = os.path.join(app_root, 'hmm', 'parse_dbtl.py')
        print(f"Parse script path: {parse_script}")
        result = subprocess.run(
            [sys.executable, parse_script], 
            check=False, 
            capture_output=True, 
            text=True,
            cwd=app_root
        )
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        if result.returncode != 0:
            print("Using existing parsed results...")
        else:
            print("Parsing completed successfully")
        
        # 更新进度
        prediction_progress['progress'] = 40
        prediction_progress['status'] = '生成结构域序列...'
        
        # 生成结构域序列
        print("Generating domain sequences...")
        # 运行脚本并捕获输出
        gen_script = os.path.join(app_root, 'hmm', 'gen_domains.py')
        print(f"Gen domains script path: {gen_script}")
        gen_result = subprocess.run(
            [sys.executable, gen_script, '--fasta_path', file_path], 
            check=False, 
            capture_output=True, 
            text=True,
            cwd=app_root
        )
        print(f"Gen domains return code: {gen_result.returncode}")
        print(f"Gen domains stdout: {gen_result.stdout}")
        print(f"Gen domains stderr: {gen_result.stderr}")
        if gen_result.returncode != 0:
            print("Using existing domain results...")
        else:
            print("Domain generation completed successfully")
        
        # 更新进度
        prediction_progress['progress'] = 50
        prediction_progress['status'] = '运行AI预测...'
        
        # 运行预测（使用训练好的模型）
        print("Running AI prediction...")
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'result_gat_contrastive.csv')
        
        # 查找最新的模型文件
        import glob
        
        # 获取应用根目录的绝对路径
        app_root = os.path.dirname(os.path.abspath(__file__))
        print(f"App root directory: {app_root}")
        
        # 构建checkpoints目录的绝对路径
        checkpoints_dir = os.path.join('/root/autodl-tmp', 'checkpoints')
        print(f"Checkpoints directory: {checkpoints_dir}")
        
        # 检查checkpoints目录是否存在
        if not os.path.exists(checkpoints_dir):
            print("Checkpoints directory not found")
        else:
            print(f"Checkpoints directory exists, listing files:")
            for model_file in os.listdir(checkpoints_dir):
                print(f"  {model_file}")
        
        # 尝试不同的路径模式
        model_patterns = [
            os.path.join(checkpoints_dir, 'model-epoch=*.ckpt'),
            os.path.join(checkpoints_dir, '*.ckpt')
        ]
        
        model_files = []
        for pattern in model_patterns:
            found = glob.glob(pattern)
            print(f"Pattern {pattern} found: {found}")
            model_files.extend(found)
        
        # 去重
        model_files = list(set(model_files))
        
        if not model_files:
            raise Exception(f"No model checkpoint found in {checkpoints_dir}. Please train the model first.")
        
        # 使用最新的模型文件
        latest_model = max(model_files, key=os.path.getctime)
        print(f"Using model: {latest_model}")
        
        # 检查ESM模型文件
        esm_model_path = os.path.join(app_root, 'model', 'esm2_t33_650M_UR50D', 'config.json')
        print(f"ESM model path: {esm_model_path}")
        if not os.path.exists(esm_model_path):
            raise Exception(f"ESM model files not found at {esm_model_path}. Please download the model first.")
        
        # 更新进度
        prediction_progress['progress'] = 60
        prediction_progress['status'] = '查找模型文件...'
        
        # 运行预测脚本
        print("Running inference script...")
        
        # 构建推理脚本的绝对路径
        inference_script = os.path.join(app_root, 'inference_gat_contrastive.py')
        print(f"Inference script path: {inference_script}")
        
        # 构建结果路径的绝对路径
        result_path_abs = os.path.join(app_root, result_path)
        print(f"Result path: {result_path_abs}")
        
        # 更新进度
        prediction_progress['progress'] = 70
        prediction_progress['status'] = '运行模型推理...'
        
        # 构建推理数据集路径
        inference_dataset = os.path.join(hmm_result_dir, 'domains.csv')
        print(f"Inference dataset path: {inference_dataset}")
        
        # 检查domains.csv文件是否存在且不为空
        if not os.path.exists(inference_dataset):
            raise Exception(f"Domain file not found at {inference_dataset}. Please check your input sequence or HMM setup.")
        
        # 检查文件是否为空
        if os.path.getsize(inference_dataset) == 0:
            raise Exception(f"Domain file at {inference_dataset} is empty. Please check your input sequence or HMM setup.")
        
        # 运行推理脚本
        inf_result = subprocess.run(
            [sys.executable, inference_script, '--inference_dataset', inference_dataset, '--result_path', result_path_abs, '--checkpoint_path', latest_model], 
            check=False, 
            capture_output=True, 
            text=True,
            cwd=app_root  # 设置工作目录为应用根目录
        )
        
        print(f"Inference return code: {inf_result.returncode}")
        print(f"Inference stdout: {inf_result.stdout}")
        print(f"Inference stderr: {inf_result.stderr}")
        
        # 更新进度
        prediction_progress['progress'] = 90
        prediction_progress['status'] = '处理预测结果...'
        
        if inf_result.returncode != 0:
            raise Exception(f"Inference failed with return code {inf_result.returncode}. Error: {inf_result.stderr}")
        else:
            print("AI prediction completed successfully")
        
        # 读取结果
        result_df = pd.read_csv(result_path_abs)
        results = result_df.to_dict('records')
        
        # 更新进度
        prediction_progress['progress'] = 100
        prediction_progress['status'] = '预测完成！'
        
        # 保存到历史记录
        app_root = os.path.dirname(os.path.abspath(__file__))
        history_dir = os.path.join(app_root, app.config['HISTORY_FOLDER'])
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        history_subdir = os.path.join(history_dir, timestamp)
        os.makedirs(history_subdir, exist_ok=True)
        
        # 复制结果文件到历史记录目录
        history_result_path = os.path.join(history_subdir, f'result_{timestamp}.csv')
        shutil.copy2(result_path_abs, history_result_path)
        
        # 复制上传的文件到历史记录目录
        history_upload_path = os.path.join(history_subdir, file.filename)
        shutil.copy2(file_path, history_upload_path)
        
        # 创建历史记录元数据
        metadata = {
            'timestamp': timestamp,
            'filename': file.filename,
            'result_file': f'result_{timestamp}.csv',
            'upload_file': file.filename,
            'num_predictions': len(results)
        }
        
        with open(os.path.join(history_subdir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return render_template('result.html', results=results)

@app.route('/explain/<int:index>')
def explain(index):
    # 获取应用根目录的绝对路径
    app_root = os.path.dirname(os.path.abspath(__file__))
    
    # 读取结果文件
    result_path = os.path.join(app_root, app.config['RESULT_FOLDER'], 'result_gat_contrastive.csv')
    print(f"Reading result from: {result_path}")
    result_df = pd.read_csv(result_path)
    
    if index >= len(result_df):
        flash('Invalid index')
        return redirect(url_for('index'))
    
    # 获取对应的数据
    row = result_df.iloc[index]
    domain_seq = row['Domain']
    
    # 加载模型和分词器
    esm_model_path = os.path.join(app_root, 'model', 'esm2_t33_650M_UR50D')
    print(f"Loading ESM model from: {esm_model_path}")
    import torch
    from transformers import EsmTokenizer, EsmForSequenceClassification
    from model.GATContrastiveModel import GATContrastiveModel
    from model.XAI import XAI
    tokenizer = EsmTokenizer.from_pretrained(esm_model_path)
    
    # 加载标签映射，动态获取标签数量
    label_to_id_path = os.path.join('/root/autodl-tmp', 'checkpoints', 'label_to_id_v2.pt')
    if os.path.exists(label_to_id_path):
        label_to_id = torch.load(label_to_id_path)
        num_labels = len(label_to_id)
        print(f"Loaded label_to_id with {num_labels} labels")
    else:
        # 如果标签映射不存在，使用默认值
        num_labels = 43
        print("Label mapping not found, using default num_labels=43")
    
    esm_model = EsmForSequenceClassification.from_pretrained(esm_model_path, num_labels=num_labels, output_hidden_states=True)
    
    # 构建temp.csv的绝对路径
    temp_path = os.path.join(app_root, 'temp.csv')
    model = GATContrastiveModel(esm_model=esm_model, lr=1e-5, weight_decay=0.01, finetune_layer=3, gama=0.9, result_path=temp_path)
    
    # 加载预训练权重
    try:
        checkpoints_dir = os.path.join('/root/autodl-tmp', 'checkpoints')
        # 尝试查找最新的模型文件
        import glob
        model_files = glob.glob(os.path.join(checkpoints_dir, '*.ckpt'))
        
        if model_files:
            checkpoint_path = max(model_files, key=os.path.getctime)
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded checkpoint successfully!")
        else:
            print("Checkpoint not found, using random weights.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    
    # 准备输入
    inputs = tokenizer(domain_seq, return_tensors="pt", padding="max_length", truncation=True, max_length=500)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 创建邻接矩阵
    seq_length = len(domain_seq)
    adj_matrix = torch.eye(seq_length)
    for i in range(seq_length):
        if i > 0:
            adj_matrix[i, i-1] = 1.0
        if i < seq_length - 1:
            adj_matrix[i, i+1] = 1.0
    adj_matrix = adj_matrix.unsqueeze(0)
    
    # 生成解释
    xai = XAI(model)
    explanation = xai.generate_explanation(input_ids, attention_mask, adj_matrix, tokenizer=tokenizer)
    
    # 确保explanation对象的结构正确
    if 'feature_importance' not in explanation:
        explanation['feature_importance'] = {'tokens': list(domain_seq), 'importance': [0.1] * len(domain_seq)}
    elif 'tokens' not in explanation['feature_importance']:
        explanation['feature_importance']['tokens'] = list(domain_seq)
    elif 'importance' not in explanation['feature_importance']:
        explanation['feature_importance']['importance'] = [0.1] * len(domain_seq)
    
    if 'grad_cam' not in explanation:
        explanation['grad_cam'] = [0.5] * len(domain_seq)
    
    return render_template('explain.html', domain_seq=domain_seq, explanation=explanation)

@app.route('/history')
def history():
    app_root = os.path.dirname(os.path.abspath(__file__))
    history_dir = os.path.join(app_root, app.config['HISTORY_FOLDER'])
    
    # 获取所有历史记录
    history_records = []
    if os.path.exists(history_dir):
        for timestamp in sorted(os.listdir(history_dir), reverse=True):
            history_subdir = os.path.join(history_dir, timestamp)
            if os.path.isdir(history_subdir):
                metadata_file = os.path.join(history_subdir, 'metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    # 格式化时间戳
                    formatted_time = datetime.datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
                    history_records.append({
                        'timestamp': timestamp,
                        'formatted_time': formatted_time,
                        'filename': metadata.get('filename', 'Unknown'),
                        'num_predictions': metadata.get('num_predictions', 0)
                    })
    
    return render_template('history.html', history_records=history_records)

@app.route('/history/<timestamp>')
def history_detail(timestamp):
    app_root = os.path.dirname(os.path.abspath(__file__))
    history_subdir = os.path.join(app_root, app.config['HISTORY_FOLDER'], timestamp)
    
    if not os.path.exists(history_subdir):
        flash('历史记录不存在')
        return redirect(url_for('history'))
    
    # 读取元数据
    metadata_file = os.path.join(history_subdir, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # 读取结果文件
    result_file = os.path.join(history_subdir, metadata.get('result_file', ''))
    if os.path.exists(result_file):
        result_df = pd.read_csv(result_file)
        results = result_df.to_dict('records')
    else:
        results = []
    
    # 格式化时间戳
    formatted_time = datetime.datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template('history_detail.html', timestamp=timestamp, formatted_time=formatted_time, 
                         metadata=metadata, results=results)

@app.route('/history/<timestamp>/download')
def download_history(timestamp):
    app_root = os.path.dirname(os.path.abspath(__file__))
    history_subdir = os.path.join(app_root, app.config['HISTORY_FOLDER'], timestamp)
    
    if not os.path.exists(history_subdir):
        flash('历史记录不存在')
        return redirect(url_for('history'))
    
    # 读取元数据
    metadata_file = os.path.join(history_subdir, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        flash('元数据不存在')
        return redirect(url_for('history'))
    
    # 下载结果文件
    result_file = os.path.join(history_subdir, metadata.get('result_file', ''))
    if os.path.exists(result_file):
        return send_file(result_file, as_attachment=True, download_name=f'result_{timestamp}.csv')
    else:
        flash('结果文件不存在')
        return redirect(url_for('history'))

@app.route('/history/<timestamp>/delete', methods=['POST'])
def delete_history(timestamp):
    app_root = os.path.dirname(os.path.abspath(__file__))
    history_subdir = os.path.join(app_root, app.config['HISTORY_FOLDER'], timestamp)
    
    if os.path.exists(history_subdir):
        try:
            shutil.rmtree(history_subdir)
            flash('历史记录已删除')
        except Exception as e:
            flash(f'删除失败: {str(e)}')
    else:
        flash('历史记录不存在')
    
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
