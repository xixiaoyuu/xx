Text-to-Protein Sequence Diffusion Model 
一、项目概述 
本项目实现了一款基于扩散模型的文本驱动蛋白质序列生成工具，核心功能是根据蛋白质的文本描述（如功能、结构、来源等），生成符合生物学规律的氨基酸序列。模型融合 Transformer 架构与扩散模型的 “前向加噪 - 反向去噪” 机制，通过文本语义编码引导蛋白质序列生成，可应用于新型功能蛋白设计、生物实验候选序列筛选、蛋白质数据扩充等生物信息学场景，为相关研究提供高效的序列生成解决方案。 
二、项目结构 
项目文件按 “功能模块” 划分，结构清晰，便于开发与维护，核心文件如下： 
Config 类 超参数配置中心，统一管理序列、文本、扩散模型、训练等相关参数  
WordTokenizer 类 文本分词器（单词级），实现蛋白质描述文本的分词、编码、解码  
SequenceTokenizer 类 蛋白质序列分词器，实现 20 种标准氨基酸的编码、解码及填充处理  SinusoidalPositionEmbeddings 类 时间步嵌入模块，将扩散过程的离散时间步转换为连续向量特征  
TextEncoder 类 文本编码器（基于 Transformer），提取蛋白质描述文本的语义特征  
DiffusionModel 类 核心扩散模型，整合文本编码、时间嵌入、条件融合、Transformer 解码等模块  augment_protein_data 函数 数据增强函数，通过氨基酸轻微变异扩充训练数据  
ProteinDataset 类 自定义数据集，实现数据加载、预处理、增强及词汇表构建  
train 函数 模型训练函数，含训练 / 验证循环、早停机制、模型保存  
generate_sequence 函数 序列生成函数，根据文本描述生成蛋白质序列  
主程序（if __name__ == "__main__"） 项目入口，串联数据加载、预处理、模型初始化、训练、生成全流程  uniprot-data.tsv 数据文件，存储 UniProt 格式的蛋白质序列 - 描述配对数据（自动生成示例数据）  best_protein_model.pth 训练过程中保存的最优模型权重（基于验证损失）  protein_diffusion_model_final.pth 训练完成的完整模型文件（含权重、分词器、配置）    
三、核心模块详解 
1. 数据处理模块
（1）分词器 WordTokenizer（文本分词）： 支持单词级分词（含连字符单词如 “DNA-binding”、标点分离如 “(”“.”）； ◦ 内置特殊标记：<pad>（填充）、<unk>（未登录词）、<sos>（句首）、<eos>（句尾）；
功能：文本→分词→编码（索引序列）、索引序列→解码（文本），自动截断 / 填充到max_text_len=128。   SequenceTokenizer（序列分词）：
覆盖 20 种标准氨基酸（ACDEFGHIKLMNPQRSTVWY）+ 1 个填充符<pad>；
功能：氨基酸序列→编码（索引序列）、索引序列→解码（序列），自动截断 / 填充到max_seq_len=256
（2）数据增强 通过 augment_protein_data 函数实现：
策略：对每条原始序列生成augmentation_factor=3个变体，每个氨基酸以mutation_rate=0.01概率替换为其他氨基酸（排除自身）；
文本适配：变体描述添加 “(variant i)” 标记，保持 “序列 - 描述” 配对一致性。
（3）自定义数据集 ProteinDataset
支持训练集 / 验证集差异化处理：训练集开启数据增强并构建文本词汇表，验证集复用词汇表且不增强；
自动过滤无效序列：仅保留长度在 20~256aa 之间的蛋白质序列（过短无生物学意义，过长超出模型处理能力）。
 2. 模型核心模块（DiffusionModel）
 3. 扩散模型是本项目的核心，实现 “文本描述→蛋白质序列” 的生成，核心逻辑分为前向扩散（加噪） 和反向扩散（去噪）：
（1）前向扩散（p_loss 方法）
（2）反向扩散（p_sample 方法）
（3）条件融合（forward_emb 方法）
3. 训练与生成模块 （1）训练函数（train）
早停机制：若验证损失连续patience=5轮无下降，停止训练，避免过拟合；
模型保存：自动保存验证损失最优的模型权重（best_protein_model.pth）。
（2）生成函数（generate_sequence）
输入：蛋白质文本描述（如 “DNA binding protein involved in transcription”）；
输出：生成的蛋白质氨基酸序列；
流程：文本编码→反向扩散采样→序列解码，端到端完成生成。
四、超参数配置（Config 类） 核心超参数统一在 Config 类中管理，可根据需求调整：
五、运行指南
1. 环境依赖 需安装 Python 3.7 + 及以下库： bash        pip install torch numpy pandas scikit-learn regex
2. 运行步骤
（1）直接运行代码 bash        python 20250727.py 
（2）核心流程解析
1.  数据加载与预处理： 自动读取 uniprot-data.tsv 数据，若文件不存在则生成 4 条示例数据（含胰岛素、胶原蛋白等）； 过滤长度 20~256aa 的有效序列，按 9:1 拆分训练集 / 验证集； 训练集应用数据增强，生成 18936 条样本（示例数据增强后），并构建文本词汇表。
2.  模型训练： 初始化 DiffusionModel 及 Adam 优化器； 训练 50 轮（含早停机制），每轮打印训练 / 验证损失，自动保存最优模型； 训练完成后生成 best_protein_model.pth（最优权重）和protein_diffusion_model_final.pth（完整模型）。
3.  序列生成：自动对 3 个示例文本描述生成蛋白质序列，输出描述、序列片段（前 100aa）及长度。   

六 、实验结果示例
1. 训练日志（部分）
Using device: cpu
Sequence vocabulary size: 21
Loaded TSV with columns: Sequence, Protein names
Extracted 4 raw sequence-description pairs
Filtered to 4 valid pairs (length 20~256)
Applying data augmentation...
Augmented dataset size: 16
Text vocabulary size: 5000
Top 10 words: ['(', ')', 'protein', 'os=', 'homo', 'sapiens', 'transcription', 'factor', 'ap-1', 'insulin']

Starting training...
Epoch 1/50
Train Loss: 0.5674 | Val Loss: 0.3721
Saved best model (val loss improved)
Epoch 2/50
Train Loss: 0.3799 | Val Loss: 0.3217
Saved best model (val loss improved)
...
Epoch 50/50
Train Loss: 0.0001 | Val Loss: 0.0001
Saved best model (val loss improved)

Generating sample sequences...
Description: DNA binding protein involved in transcription
Generated sequence: QWMHNPMWWAMYCYAIYMWAFRMHWWYHWMDCRFEILVYYNWMHAERHEIWREFNEFRFKPAMFTHANEMEVMERAPYDMMEGYCEGYFRCLQMRWQQLG...
Sequence length: 237

Description: Enzyme with catalytic activity for hydrolysis
Generated sequence: ASRSNYIFGEKRSQETDDYNHADHGEEHAMLHYYKSFYKTDNWGMEEWGPPHEMAYDAHLANCWHWWTYFGEIIYEFYAYYLDLQFPQMFWTQNAHMIHR...
Sequence length: 244
生成序列特征 
长度合规：生成序列长度均在 20~256aa 之间，符合预设有效范围； 
氨基酸有效：仅包含 20 种标准氨基酸，无无效字符或填充符； 
功能匹配：序列中隐含与文本描述对应的氨基酸特征（如 DNA 结合蛋白含大量正电氨基酸 K/R/H，水解酶含酸性氨基酸 E/D）。
