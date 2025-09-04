Text-to-Protein Sequence Diffusion Model
一、介绍 
文本到蛋白质序列的扩散模型，是一款基于扩散模型（Diffusion Model）与 Transformer 架构的生成式 AI 工具，核心功能是根据蛋白质的文本描述（如功能、结构、来源等），生成符合生物学规律的氨基酸序列。模型将文本语义编码与扩散过程的 “前向加噪 - 反向去噪” 机制结合，通过文本条件约束蛋白质序列生成，可应用于新型功能蛋白设计、生物实验候选序列筛选、蛋白质数据扩充等生物信息学场景，为相关研究提供高效的序列生成解决方案。
二、模型结构
本模型是文本条件约束的轻量化扩散生成模型，以 Transformer 为基础架构，整合文本编码器、时间步嵌入、条件融合模块及 Transformer 解码器，通过在嵌入空间进行扩散，实现 “文本描述→蛋白质序列” 的精准映射。模型在保证生成质量的同时，兼顾推理效率，核心结构如下：
DiffusionModel(
  # 1. 文本条件编码器（提取蛋白质描述语义特征）
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

  # 2. 时间步嵌入（编码扩散过程的时间信息）
  (time_embed): Sequential(
    (0): SinusoidalPositionEmbeddings(dim=64)  # 正弦位置编码（维度64）
    (1): Linear(in_features=64, out_features=64)
    (2): GELU()  # 激活函数
  )

  # 3. 蛋白质序列嵌入（氨基酸→向量映射）
  (seq_embed): Embedding(21, 128, padding_idx=0)  # 21类token（20种氨基酸+1填充符）

  # 4. 条件融合模块（融合文本特征与时间嵌入）
  (condition_fuse): Sequential(
    (0): Linear(in_features=256, out_features=256)  # 128（文本）+64（时间）=256
    (1): GELU()
    (2): Dropout(p=0.3)
    (3): Linear(in_features=256, out_features=128)  # 映射到序列嵌入维度
  )

  # 5. Transformer解码器（实现反向去噪）
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

  # 6. 输出层（嵌入→氨基酸概率分布）
  (output_layer): Linear(in_features=128, out_features=21)  # 输出21类氨基酸/填充符概率

  # 7. 扩散参数（前向加噪/反向去噪系数）
  (betas): tensor([1e-4, ..., 0.02])  # 噪声强度（1000步线性递增）
  (alphas): tensor([0.9999, ..., 0.98])  # 1 - betas
  (alphas_bar): tensor([0.9999, ..., 累积乘积])  # 噪声系数累积乘积
  (sqrt_alphas_bar): tensor([sqrt(alphas_bar)])
  (sqrt_one_minus_alphas_bar): tensor([sqrt(1-alphas_bar)])
)
三、数据集 
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
四、训练过程
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
五、实验结果
训练收敛效果 模型经过 50 轮训练，损失持续下降至极低水平，无过拟合现象。
六、使用指南
# 安装所需Python库
pip install torch==2.1.0 numpy==1.26.0 pandas==2.1.1 scikit-learn==1.3.0 regex==2023.8.8
