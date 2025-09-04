# Text-to-Protein Sequence Diffusion Model
## 一、介绍 
文本到蛋白质序列的扩散模型，是一款基于扩散模型（Diffusion Model）与 Transformer 架构的生成式 AI 工具，核心功能是根据蛋白质的文本描述（如功能、结构、来源等），生成符合生物学规律的氨基酸序列。模型将文本语义编码与扩散过程的 “前向加噪 - 反向去噪” 机制结合，通过文本条件约束蛋白质序列生成。
## 二、模型结构
本模型是文本条件约束的轻量化扩散生成模型，以 Transformer 为基础架构，整合文本编码器、时间步嵌入、条件融合模块及 Transformer 解码器，通过在嵌入空间进行扩散，实现 “文本描述→蛋白质序列” 的映射。模型在保证生成质量的同时，兼顾推理效率，核心结构如下：

DiffusionModel(
   #1. 文本条件编码器（提取蛋白质描述语义特征）
  (text_encoder): TextEncoder(
    (embedding): Embedding(5000, 128, padding_idx=0)  # 单词嵌入（词汇表5000，维度128）
    (encoder): TransformerEncoder(
      (layers): 6 x TransformerEncoderLayer(  # 6层Transformer编码器
        (self_attn): MultiheadAttention(8 heads, d_model=128)  # 8头自注意力
        (linear1): Linear(in_features=128, out_features=256)  # 前馈网络隐藏层
        (dropout): Dropout(p=0.3)  # 正则化
        (linear2): Linear(in_features=256, out_features=128)
        (norm1): LayerNorm((128,), eps=1e-05)
        (norm2): LayerNorm((128,), eps=1e-05)
      )
    )
  )

   #2. 时间步嵌入（编码扩散过程的时间信息）
  (time_embed): Sequential(
    (0): SinusoidalPositionEmbeddings(dim=64)  # 正弦位置编码（维度64）
    (1): Linear(in_features=64, out_features=64)
    (2): GELU()  # 激活函数
  )

   #3. 蛋白质序列嵌入（氨基酸→向量映射）
  (seq_embed): Embedding(21, 128, padding_idx=0)  # 21类token（20种氨基酸+1填充符）

   #4. 条件融合模块（融合文本特征与时间嵌入）
  (condition_fuse): Sequential(
    (0): Linear(in_features=256, out_features=256)  # 128（文本）+64（时间）=256
    (1): GELU()
    (2): Dropout(p=0.3)
    (3): Linear(in_features=256, out_features=128)  # 映射到序列嵌入维度
  )

   #5. Transformer解码器（实现反向去噪）
  (decoder): TransformerDecoder(
    (layers): 6 x TransformerDecoderLayer(  # 6层Transformer解码器
      (self_attn): MultiheadAttention(8 heads, d_model=128)
      (multihead_attn): MultiheadAttention(8 heads, d_model=128)  # 交叉注意力（文本为memory）
      (linear1): Linear(in_features=128, out_features=256)
      (dropout): Dropout(p=0.3)
      (linear2): Linear(in_features=256, out_features=128)
      (norm1): LayerNorm((128,), eps=1e-05)
      (norm2): LayerNorm((128,), eps=1e-05)
      (norm3): LayerNorm((128,), eps=1e-05)
    )
  )

   #6. 输出层（嵌入→氨基酸概率分布）
  (output_layer): Linear(in_features=128, out_features=21)  # 输出21类氨基酸/填充符概率

   #7. 扩散参数（前向加噪/反向去噪系数）
  (betas): tensor([1e-4, ..., 0.02])  # 噪声强度（1000步线性递增）
  (alphas): tensor([0.9999, ..., 0.98])  # 1 - betas
  (alphas_bar): tensor([0.9999, ..., 累积乘积])  # 噪声系数累积乘积
  (sqrt_alphas_bar): tensor([sqrt(alphas_bar)])
  (sqrt_one_minus_alphas_bar): tensor([sqrt(1-alphas_bar)])
)
## 三、数据集 
1. 数据集来源与规模 本模型采用UniProt 数据库格式的蛋白质序列 - 文本配对数据，原始数据包含蛋白质的氨基酸序列（Sequence）和功能描述（Protein names）
 原始数据规模：20,420 条序列 - 描述配对；
 有效数据规模：5,260 条（过滤长度为 20~256aa 的序列，剔除过短 / 过长的无效样本）；
 增强后数据规模：18,936 条（通过氨基酸变异扩充训练集，缓解数据稀缺问题）。
2. 数据集结构 数据集以 TSV（Tab-Separated Values）格式存储，文件名为uniprot-data.tsv
3. 数据预处理与增强
   （1）数据预处理
   序列过滤：仅保留长度在 20~256aa 之间的序列（过短无生物学意义，过长超出模型处理能力）；
   文本编码：通过WordTokenizer实现单词级分词（支持连字符单词、标点分离），编码为长度 128 的索引序列，添加<sos>/<eos>/<pad>/<unk>特殊标记；
   序列编码：通过SequenceTokenizer将 20 种氨基酸编码为索引，填充 / 截断为长度 256 的序列。
   （2）数据增强 针对训练集实施氨基酸轻微变异
   （3）数据拆分 按 9:1 比例拆分训练集与验证集
## 四、训练过程
1. 核心配置参数
2. 训练流程
   （1）初始化阶段
   1. 加载超参数配置（Config类，含序列长度、嵌入维度、扩散参数等）；
   2. 初始化文本分词器（WordTokenizer）和序列分词器（SequenceTokenizer）；
   3. 构建扩散模型（DiffusionModel），迁移至目标设备（GPU/CPU）；
   4. 初始化损失函数（MSE）和优化器（Adam）。
   （2）数据加载阶段
1. 读取 TSV 数据，过滤有效序列（20~256aa）；
2. 按 9:1 比例拆分训练集 / 验证集；
3. 对训练集应用数据增强，生成 18,936 条训练样本；
4. 构建数据加载器（DataLoader），训练集shuffle=True（打乱数据），验证集shuffle=False。
   （3）训练循环
   （4）训练输出
## 五、实验结果
训练收敛效果 模型经过 50 轮训练，损失持续下降至极低水平，无过拟合现象。
Using device: cpu
Sequence vocabulary size: 21
Loaded TSV with columns: Entry, Protein names, Sequence
Extracted 20420 raw sequence-description pairs
Filtered to 5260 valid pairs (length 20~256)
Applying data augmentation...
Augmented dataset size: 18936

Text vocabulary size: 5000
Top 10 words: ['(', ')', 'protein', 'variant', '1', '2', '.', '3', 'subunit', 'ec']

Starting training...
Epoch 1/50
Train Loss: 0.5674 | Val Loss: 0.3721
Saved best model (val loss improved)
Epoch 2/50
Train Loss: 0.3799 | Val Loss: 0.3217
Saved best model (val loss improved)
Epoch 3/50
Train Loss: 0.3206 | Val Loss: 0.2737
Saved best model (val loss improved)
Epoch 4/50
Train Loss: 0.2798 | Val Loss: 0.2486
Saved best model (val loss improved)
Epoch 5/50
Train Loss: 0.2481 | Val Loss: 0.2156
Saved best model (val loss improved)
Epoch 6/50
Train Loss: 0.2150 | Val Loss: 0.1833
Saved best model (val loss improved)
Epoch 7/50
Train Loss: 0.1933 | Val Loss: 0.1584
Saved best model (val loss improved)
Epoch 8/50
Train Loss: 0.1714 | Val Loss: 0.1514
Saved best model (val loss improved)
Epoch 9/50
Train Loss: 0.1524 | Val Loss: 0.1464
Saved best model (val loss improved)
Epoch 10/50
Train Loss: 0.1372 | Val Loss: 0.1178
Saved best model (val loss improved)
Epoch 11/50
Train Loss: 0.1245 | Val Loss: 0.1167
Saved best model (val loss improved)
Epoch 12/50
Train Loss: 0.1103 | Val Loss: 0.1005
Saved best model (val loss improved)
Epoch 13/50
Train Loss: 0.0987 | Val Loss: 0.0919
Saved best model (val loss improved)
Epoch 14/50
Train Loss: 0.0901 | Val Loss: 0.0791
Saved best model (val loss improved)
Epoch 15/50
Train Loss: 0.0811 | Val Loss: 0.0748
Saved best model (val loss improved)
Epoch 16/50
Train Loss: 0.0732 | Val Loss: 0.0706
Saved best model (val loss improved)
Epoch 17/50
Train Loss: 0.0654 | Val Loss: 0.0611
Saved best model (val loss improved)
Epoch 18/50
Train Loss: 0.0577 | Val Loss: 0.0524
Saved best model (val loss improved)
Epoch 19/50
Train Loss: 0.0518 | Val Loss: 0.0528
Epoch 20/50
Train Loss: 0.0466 | Val Loss: 0.0473
Saved best model (val loss improved)
Epoch 21/50
Train Loss: 0.0409 | Val Loss: 0.0407
Saved best model (val loss improved)
Epoch 22/50
Train Loss: 0.0365 | Val Loss: 0.0360
Saved best model (val loss improved)
Epoch 23/50
Train Loss: 0.0319 | Val Loss: 0.0323
Saved best model (val loss improved)
Epoch 24/50
Train Loss: 0.0279 | Val Loss: 0.0281
Saved best model (val loss improved)
Epoch 25/50
Train Loss: 0.0243 | Val Loss: 0.0248
Saved best model (val loss improved)
Epoch 26/50
Train Loss: 0.0210 | Val Loss: 0.0216
Saved best model (val loss improved)
Epoch 27/50
Train Loss: 0.0182 | Val Loss: 0.0183
Saved best model (val loss improved)
Epoch 28/50
Train Loss: 0.0156 | Val Loss: 0.0157
Saved best model (val loss improved)
Epoch 29/50
Train Loss: 0.0134 | Val Loss: 0.0138
Saved best model (val loss improved)
Epoch 30/50
Train Loss: 0.0113 | Val Loss: 0.0112
Saved best model (val loss improved)
Epoch 31/50
Train Loss: 0.0096 | Val Loss: 0.0092
Saved best model (val loss improved)
Epoch 32/50
Train Loss: 0.0079 | Val Loss: 0.0077
Saved best model (val loss improved)
Epoch 33/50
Train Loss: 0.0066 | Val Loss: 0.0063
Saved best model (val loss improved)
Epoch 34/50
Train Loss: 0.0054 | Val Loss: 0.0051
Saved best model (val loss improved)
Epoch 35/50
Train Loss: 0.0044 | Val Loss: 0.0042
Saved best model (val loss improved)
Epoch 36/50
Train Loss: 0.0036 | Val Loss: 0.0034
Saved best model (val loss improved)
Epoch 37/50
Train Loss: 0.0028 | Val Loss: 0.0027
Saved best model (val loss improved)
Epoch 38/50
Train Loss: 0.0023 | Val Loss: 0.0022
Saved best model (val loss improved)
Epoch 39/50
Train Loss: 0.0018 | Val Loss: 0.0017
Saved best model (val loss improved)
Epoch 40/50
Train Loss: 0.0014 | Val Loss: 0.0013
Saved best model (val loss improved)
Epoch 41/50
Train Loss: 0.0011 | Val Loss: 0.0010
Saved best model (val loss improved)
Epoch 42/50
Train Loss: 0.0008 | Val Loss: 0.0008
Saved best model (val loss improved)
Epoch 43/50
Train Loss: 0.0006 | Val Loss: 0.0006
Saved best model (val loss improved)
Epoch 44/50
Train Loss: 0.0005 | Val Loss: 0.0004
Saved best model (val loss improved)
Epoch 45/50
Train Loss: 0.0004 | Val Loss: 0.0003
Saved best model (val loss improved)
Epoch 46/50
Train Loss: 0.0003 | Val Loss: 0.0002
Saved best model (val loss improved)
Epoch 47/50
Train Loss: 0.0002 | Val Loss: 0.0002
Saved best model (val loss improved)
Epoch 48/50
Train Loss: 0.0002 | Val Loss: 0.0001
Saved best model (val loss improved)
Epoch 49/50
Train Loss: 0.0001 | Val Loss: 0.0001
Saved best model (val loss improved)
Epoch 50/50
Train Loss: 0.0001 | Val Loss: 0.0001
Saved best model (val loss improved)
     C:\Users\32816\AppData\Local\Temp\ipykernel_39888\1122325614.py:447: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load("best_protein_model.pth"))
     
Model saved to 'protein_diffusion_model_final.pth'

Generating sample sequences...

Description: DNA binding protein involved in transcription
Generated sequence: QWMHNPMWWAMYCYAIYMWAFRMHWWYHWMDCRFEILVYYNWMHAERHEIWREFNEFRFKPAMFTHANEMEVMERAPYDMMEGYCEGYFRCLQMRWQQLG...
Sequence length: 237

Description: Enzyme with catalytic activity for hydrolysis
Generated sequence: ASRSNYIFGEKRSQETDDYNHADHGEEHAMLHYYKSFYKTDNWGMEEWGPPHEMAYDAHLANCWHWWTYFGEIIYEFYAYYLDLQFPQMFWTQNAHMIHR...
Sequence length: 244

Description: Membrane transport protein for ions
Generated sequence: KCGPRLEHPSHKCQWWYEEINDCNPCIFHEERPYHMKMMQENYRRFMYVESDSWHQCLFGMGDGNYYFGYYCDSAGVWGEGMRVTRYKFWWRFYGHFRSK...
Sequence length: 244
## 六、使用指南
安装所需Python库
pip install torch==2.1.0 numpy==1.26.0 pandas==2.1.1 scikit-learn==1.3.0 regex==2023.8.8
