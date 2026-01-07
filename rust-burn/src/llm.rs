use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Gelu, Linear, LinearConfig, RmsNorm, RmsNormConfig,
        RotaryEncoding, RotaryEncodingConfig,
    },
    optim::AdamWConfig,
    prelude::*,
    tensor::{Tensor, activation::softmax},
    train::ClassificationOutput,
};
use std::fs;
use tokenizers::Tokenizer;

// ============================================================================
// 数据集相关结构
// ============================================================================

/// 单个训练样本
/// 
/// # 字段说明
/// * `input_ids`: 输入token ID序列，维度: [seq_len]
/// * `target_ids`: 目标token ID序列，维度: [seq_len]
/// 
/// # 训练目标
/// 对于输入序列中的每个位置i，模型需要预测target_ids[i]（即input_ids[i+1]）
/// 例如：input_ids = [1, 2, 3, 4]，target_ids = [2, 3, 4, 5]
/// 模型学习将 [1,2,3,4] 映射到 [2,3,4,5]
#[derive(Debug, Clone)]
pub struct TokenDatasetItem {
    pub input_ids: Vec<i64>,
    pub target_ids: Vec<i64>,
}

/// Token数据集
/// 
/// 用于存储和管理训练样本，实现Dataset trait以便与DataLoader配合使用
pub struct TokenDataset {
    items: Vec<TokenDatasetItem>,  // 所有训练样本的列表
    tokenizer: Tokenizer,         // Tokenizer实例，用于文本编码
    seq_len: usize,             // 序列长度
}

impl TokenDataset {
    /// 使用tokenizer从文本文件创建数据集
    /// 
    /// # 数据处理流程
    /// 1. 读取文本文件（约10MB，几百万字符）
    /// 2. 使用tokenizer将文本编码为token ID序列（维度: [N]，N约2-3M）
    /// 3. 使用滑动窗口创建训练样本
    /// 4. 每个样本包含input和target，用于下一个token预测任务
    /// 
    /// # 滑动窗口示例
    /// 假设tokens = [1, 2, 3, 4, 5, 6, 7, 8]，seq_len=4, stride=2
    /// 样本1: input=[1,2,3,4], target=[2,3,4,5]
    /// 样本2: input=[3,4,5,6], target=[4,5,6,7]
    /// 样本3: input=[5,6,7,8], target=[6,7,8,?] (需要seq_len+1个token)
    /// 
    /// # 参数说明
    /// * `text_path`: 训练文本文件路径
    /// * `tokenizer_path`: tokenizer配置文件路径（JSON格式）
    /// * `seq_len`: 序列长度（例如256）
    /// * `stride`: 滑动窗口步长（例如256，无重叠）
    /// 
    /// # 返回
    /// 包含所有训练样本的TokenDataset
    pub fn new_with_tokenizer(
        text_path: &str,
        tokenizer_path: &str,
        seq_len: usize,
        stride: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // ============ 步骤1: 加载Tokenizer ============
        // 从JSON文件加载预训练的tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        // ============ 步骤2: 读取文本文件 ============
        // 读取整个文本文件到内存
        let text = fs::read_to_string(text_path)?;

        // ============ 步骤3: 文本编码 ============
        // 将文本转换为token ID序列
        // encoding维度: [N]，N为token总数（约2-3M）
        let encoding = tokenizer.encode(text.as_str(), false)?;
        let tokens: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        let token_len = tokens.len();
        
        // ============ 步骤4: 验证token数量 ============
        // 需要至少 seq_len + 1 个token（input: seq_len, target: seq_len，最后一个位置需要额外token）
        if token_len < seq_len + 1 {
            return Err(format!(
                "Not enough tokens: {} tokens available, need at least {} (seq_len={} + 1 target)",
                token_len,
                seq_len + 1,
                seq_len
            )
            .into());
        }

        // ============ 步骤5: 计算滑动窗口参数 ============
        // 最大的起始索引：token_len - seq_len - 1
        // 例如：token_len=1000, seq_len=256，则max_start_idx=743
        let max_start_idx = token_len - seq_len - 1;
        let effective_stride = if stride == 0 { 1 } else { stride };

        let mut items = Vec::new();

        // ============ 步骤6: 创建训练样本 ============
        // 使用滑动窗口遍历token序列，创建训练样本
        for i in (0..=max_start_idx).step_by(effective_stride) {
            // 确保有足够的token用于input和target
            if i + seq_len + 1 > token_len {
                break;
            }
            
            // 提取input序列: tokens[i..i+seq_len]
            // 维度: [seq_len]，例如 [256]
            let input_ids = tokens[i..i + seq_len].to_vec();
            
            // 提取target序列: tokens[i+1..i+seq_len+1]
            // 维度: [seq_len]，例如 [256]
            // target_ids[j] = input_ids[j+1]，预测下一个token
            let target_ids = tokens[i + 1..i + seq_len + 1].to_vec();

            items.push(TokenDatasetItem {
                input_ids,
                target_ids,
            });
        }

        // ============ 步骤7: 防御性检查 ============
        // 如果没有创建任何样本（极端情况），至少创建一个
        if items.is_empty() {
            if token_len >= seq_len + 1 {
                let input_ids = tokens[0..seq_len].to_vec();
                let target_ids = tokens[1..seq_len + 1].to_vec();
                items.push(TokenDatasetItem {
                    input_ids,
                    target_ids,
                });
            }
        }

        println!(
            "创建了 {} 个训练样本 (seq_len={}, stride={}, total_tokens={})",
            items.len(),
            seq_len,
            stride,
            token_len
        );

        Ok(TokenDataset {
            items,
            tokenizer,
            seq_len,
        })
    }

    /// 获取词汇表大小
    /// 
    /// # 返回
    /// 词汇表大小，例如151669
    /// 这个值决定了embedding层的行数和输出层的维度
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

/// 实现Dataset trait，使TokenDataset可以与DataLoader配合使用
impl Dataset<TokenDatasetItem> for TokenDataset {
    /// 获取指定索引的样本
    fn get(&self, index: usize) -> Option<TokenDatasetItem> {
        self.items.get(index).cloned()
    }

    /// 获取数据集大小
    fn len(&self) -> usize {
        self.items.len()
    }
}

// ============================================================================
// 批次处理相关结构
// ============================================================================

/// Token批次处理器
/// 
/// 负责将多个训练样本合并为一个批次，供模型批量处理
#[derive(Clone, Default)]
pub struct TokenBatcher;

/// 批次数据
/// 
/// # 字段说明
/// * `input_ids`: 输入token批次，维度: [batch_size, seq_len]
/// * `target_ids`: 目标token批次，维度: [batch_size, seq_len]
/// 
/// # 维度示例
/// 假设batch_size=4, seq_len=256:
/// - input_ids: [4, 256]，包含4个样本，每个256个token
/// - target_ids: [4, 256]，对应的下一个token预测目标
#[derive(Clone, Debug)]
pub struct TokenBatch<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,
    pub target_ids: Tensor<B, 2, Int>,
}

/// 实现Batcher trait，定义如何将多个样本批次化
impl<B: Backend> Batcher<B, TokenDatasetItem, TokenBatch<B>> for TokenBatcher {
    /// 将多个训练样本合并为一个批次
    /// 
    /// # 数据转换流程
    /// 1. 对于每个样本：
    ///    - 从Vec<i64>创建TensorData: [seq_len]
    ///    - 转换为张量: [seq_len]
    ///    - 添加批次维度: [1, seq_len]
    /// 2. 合并所有样本:
    ///    - 沿批次维度连接: [batch_size, seq_len]
    /// 
    /// # 参数说明
    /// * `items`: 多个训练样本的列表，每个样本的input_ids和target_ids维度为[seq_len]
    /// * `device`: 张量所在的设备（CPU或GPU）
    /// 
    /// # 返回
    /// 包含批次化数据的TokenBatch，维度: [batch_size, seq_len]
    fn batch(&self, items: Vec<TokenDatasetItem>, device: &B::Device) -> TokenBatch<B> {
        // ============ 步骤1: 处理input_ids ============
        // 将每个样本的input_ids转换为张量
        let input_ids = items
            .iter()
            .map(|item| {
                // 从Vec<i64>创建TensorData
                // item.input_ids维度: [seq_len]，例如 [256]
                TensorData::new(item.input_ids.clone(), [item.input_ids.len()])
                    .convert::<B::IntElem>()  // 转换为后端的整数类型
            })
            .map(|data| {
                // 创建1D张量: [seq_len]
                Tensor::<B, 1, Int>::from_data(data, device)
                    .unsqueeze_dim(0)  // 添加批次维度: [1, seq_len]
            })
            .collect();  // 收集为Vec<Tensor<B, 1, Int>>

        // ============ 步骤2: 处理target_ids ============
        // 同样处理target_ids
        let target_ids = items
            .iter()
            .map(|item| {
                // item.target_ids维度: [seq_len]，例如 [256]
                TensorData::new(item.target_ids.clone(), [item.target_ids.len()])
                    .convert::<B::IntElem>()
            })
            .map(|data| {
                // 创建1D张量并添加批次维度: [1, seq_len]
                Tensor::<B, 1, Int>::from_data(data, device).unsqueeze_dim(0)
            })
            .collect();

        // ============ 步骤3: 合并为批次 ============
        // 沿维度0（批次维度）连接所有样本
        // input_ids维度从[1, seq_len]的列表变为[batch_size, seq_len]
        let input_ids: Tensor<B, 2, Int> = Tensor::cat(input_ids, 0);
        
        // target_ids维度从[1, seq_len]的列表变为[batch_size, seq_len]
        let target_ids: Tensor<B, 2, Int> = Tensor::cat(target_ids, 0);

        TokenBatch {
            input_ids,
            target_ids,
        }
    }
}

// ============================================================================
// 模型架构相关结构
// ============================================================================

/// 前馈神经网络（SwiGLU变体）
/// 
/// 这是Transformer中使用的非线性变换层，通常在注意力机制之后
/// 
/// # 架构说明
/// - 输入维度: [batch_size, seq_len, embedding_dim]
/// - up分支: 输入 → 隐藏层（4倍embedding_dim）
/// - gate分支: 输入 → 隐藏层（4倍embedding_dim）→ GeLU激活
/// - 元素乘法: up * gate（门控机制）
/// - 下投影: [batch_size, seq_len, hidden_dim] → [batch_size, seq_len, embedding_dim]
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    up: Linear<B>,         // 上分支线性层
    gate: Linear<B>,        // 门分支线性层
    down: Linear<B>,       // 下投影线性层
    activation: Gelu,      // GeLU激活函数
}

/// 前馈网络配置
#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    in_dim: usize,         // 输入维度（embedding_dim）
    hidden_dim: usize,     // 隐藏层维度（通常4倍于in_dim）
    out_dim: usize,        // 输出维度（embedding_dim）
}

impl FeedForwardConfig {
    /// 初始化前馈网络
    /// 
    /// # 参数说明
    /// * `device`: 张量所在的设备
    /// 
    /// # 返回
    /// 初始化好的FeedForward模块
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            // up分支: [embedding_dim] → [hidden_dim]
            up: LinearConfig::new(self.in_dim, self.hidden_dim).init(device),
            // gate分支: [embedding_dim] → [hidden_dim]
            gate: LinearConfig::new(self.in_dim, self.hidden_dim).init(device),
            // 下投影: [hidden_dim] → [embedding_dim]
            down: LinearConfig::new(self.hidden_dim, self.out_dim).init(device),
            activation: Gelu::new(),
        }
    }
}

impl<B: Backend> FeedForward<B> {
    /// 前向传播
    /// 
    /// # 数据维度变换
    /// 输入 x: [batch_size, seq_len, embedding_dim]
    /// up_x: [batch_size, seq_len, hidden_dim]
    /// gate_x: [batch_size, seq_len, hidden_dim]
    /// mul_cat: [batch_size, seq_len, hidden_dim]
    /// 输出: [batch_size, seq_len, embedding_dim]
    /// 
    /// # 参数说明
    /// * `x`: 输入张量，维度: [batch_size, seq_len, embedding_dim]
    /// 
    /// # 返回
    /// 输出张量，维度: [batch_size, seq_len, embedding_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x_clone = x.clone();
        
        // 上分支投影
        // 输入: [batch_size, seq_len, embedding_dim]
        // 输出: [batch_size, seq_len, hidden_dim]
        let up_x = self.up.forward(x_clone);
        
        // 门分支投影 + GeLU激活
        // 输入: [batch_size, seq_len, embedding_dim]
        // gate输出: [batch_size, seq_len, hidden_dim]
        let gate_x = self.activation.forward(self.gate.forward(x));
        
        // 元素乘法（SwiGLU的门控机制）
        // mul_cat: [batch_size, seq_len, hidden_dim]
        let mul_cat = up_x * gate_x;
        
        // 下投影回embedding维度
        // 输入: [batch_size, seq_len, hidden_dim]
        // 输出: [batch_size, seq_len, embedding_dim]
        self.down.forward(mul_cat)
    }
}

/// 自注意力块（Transformer的核心组件）
/// 
/// 这是Transformer的基本构建块，包含：
/// 1. 预层归一化（RMSNorm）
/// 2. 多头自注意力机制
/// 3. 残差连接
/// 4. 后层归一化
/// 5. 前馈网络
/// 6. 第二个残差连接
#[derive(Module, Debug)]
pub struct AttentionBlock<B: Backend> {
    rms_norm1: RmsNorm<B>,         // 预层归一化
    q_proj: Linear<B>,              // Query投影
    k_proj: Linear<B>,              // Key投影
    v_proj: Linear<B>,              // Value投影
    o_proj: Linear<B>,              // 输出投影
    rope: RotaryEncoding<B>,         // 旋转位置编码
    n_heads: usize,                // 注意力头数
    head_dim: usize,               // 每个头的维度
    rms_norm2: RmsNorm<B>,         // 后层归一化
    feed_forward: FeedForward<B>,    // 前馈网络
}

/// 注意力块配置
#[derive(Config, Debug)]
pub struct AttentionBlockConfig {
    embedding_dim: usize,       // 嵌入维度
    n_head: usize,             // 注意力头数
    n_kv_head: usize,          // Key-Value注意力头数
    hidden_dim: usize,         // 前馈网络的隐藏层维度
    max_context: usize,        // 最大上下文长度
    #[config(default = 1e-6)]
    eps: f64,                  // RMSNorm的epsilon
}

impl AttentionBlockConfig {
    /// 初始化注意力块
    /// 
    /// # 参数说明
    /// * `device`: 张量所在的设备
    /// 
    /// # 返回
    /// 初始化好的AttentionBlock模块
    pub fn init<B: Backend>(&self, device: &B::Device) -> AttentionBlock<B> {
        // 注意：当前实现不支持Grouped Query Attention（GQA）
        // 因此n_head必须等于n_kv_head
        assert_eq!(
            self.n_head, self.n_kv_head,
            "AttentionBlock requires n_head == n_kv_head (does not support GQA)"
        );

        // 计算每个注意力头的维度
        // 例如：embedding_dim=512, n_head=8 → head_dim=64
        let head_dim = self.embedding_dim / self.n_head;

        AttentionBlock {
            // 预层归一化
            // 维度: [embedding_dim]，例如 [512]
            rms_norm1: RmsNormConfig::new(self.embedding_dim)
                .with_epsilon(self.eps)
                .init(device),
            
            // Query投影: [embedding_dim] → [embedding_dim]
            q_proj: LinearConfig::new(self.embedding_dim, self.embedding_dim).init(device),
            
            // Key投影: [embedding_dim] → [embedding_dim]
            k_proj: LinearConfig::new(self.embedding_dim, self.embedding_dim).init(device),
            
            // Value投影: [embedding_dim] → [embedding_dim]
            v_proj: LinearConfig::new(self.embedding_dim, self.embedding_dim).init(device),
            
            // 输出投影: [embedding_dim] → [embedding_dim]
            o_proj: LinearConfig::new(self.embedding_dim, self.embedding_dim).init(device),
            
            // 旋转位置编码
            // RoPE用于注入位置信息到Q和K
            rope: RotaryEncodingConfig::new(self.max_context, head_dim)
                .with_theta(10000.0)
                .init(device),
            
            n_heads: self.n_head,
            head_dim,
            
            // 后层归一化
            // 维度: [embedding_dim]
            rms_norm2: RmsNormConfig::new(self.embedding_dim)
                .with_epsilon(self.eps)
                .init(device),
            
            // 前馈网络
            // FeedForward: [embedding_dim] → [4*embedding_dim] → [embedding_dim]
            feed_forward: FeedForwardConfig::new(
                self.embedding_dim,
                self.hidden_dim,
                self.embedding_dim,
            )
            .init::<B>(device),
        }
    }
}

impl<B: Backend> AttentionBlock<B> {
    /// 注意力块的前向传播
    /// 
    /// # 数据维度变换流程
    /// 输入 x: [batch_size, seq_len, embedding_dim]
    /// 
    /// 步骤1 - 预归一化:
    /// x_norm1: [batch_size, seq_len, embedding_dim]
    /// 
    /// 步骤2 - 投影到Q, K, V:
    /// q, k, v: [batch_size, seq_len, n_heads, head_dim]
    /// 重组后: [batch_size, n_heads, seq_len, head_dim]
    /// 
    /// 步骤3 - 应用RoPE:
    /// q, k: [batch_size, n_heads, seq_len, head_dim]
    /// 
    /// 步骤4 - 计算注意力分数:
    /// attn_scores: [batch_size, n_heads, seq_len, seq_len]
    /// 
    /// 步骤5 - 应用softmax:
    /// attn_weights: [batch_size, n_heads, seq_len, seq_len]
    /// 
    /// 步骤6 - 应用注意力到值:
    /// output: [batch_size, n_heads, seq_len, head_dim]
    /// 重组后: [batch_size, seq_len, embedding_dim]
    /// 
    /// 步骤7 - 输出投影:
    /// x_atten: [batch_size, seq_len, embedding_dim]
    /// 
    /// 步骤8 - 残差连接和前馈网络:
    /// 最终输出: [batch_size, seq_len, embedding_dim]
    /// 
    /// # 参数说明
    /// * `x`: 输入张量，维度: [batch_size, seq_len, embedding_dim]
    /// * `_mask`: 注意力掩码（当前未使用）
    /// 
    /// # 返回
    /// 输出张量，维度: [batch_size, seq_len, embedding_dim]
    pub fn forward(&self, x: Tensor<B, 3>, _mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        // ============ 步骤1: 预层归一化 ============
        // x: [batch_size, seq_len, embedding_dim]
        let x_norm1 = self.rms_norm1.forward(x.clone());
        let [batch_size, seq_len, _] = x_norm1.dims();

        // ============ 步骤2: 投影到Q, K, V ============
        // Query投影
        // x_norm1: [batch_size, seq_len, embedding_dim]
        // q: [batch_size, seq_len, embedding_dim]
        // reshape后: [batch_size, seq_len, n_heads, head_dim]
        // permute后: [batch_size, n_heads, seq_len, head_dim]
        let q = self
            .q_proj
            .forward(x_norm1.clone())
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3]); // [batch, n_heads, seq_len, head_dim]

        // Key投影
        // k: [batch_size, n_heads, seq_len, head_dim]
        let k = self
            .k_proj
            .forward(x_norm1.clone())
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3]); // [batch, n_heads, seq_len, head_dim]

        // Value投影
        // v: [batch_size, n_heads, seq_len, head_dim]
        let v = self
            .v_proj
            .forward(x_norm1.clone())
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3]); // [batch, n_heads, seq_len, head_dim]

        // ============ 步骤3: 应用RoPE（旋转位置编码） ============
        // RoPE将位置信息编码到Q和K中
        // q, k: [batch_size, n_heads, seq_len, head_dim]
        let q = self.rope.forward(q);
        let k = self.rope.forward(k);

        // ============ 步骤4: 计算注意力分数 ============
        // 注意力分数 = Q @ K^T / sqrt(d_k)
        // q: [batch, n_heads, seq_len, head_dim]
        // k.transpose(): [batch, n_heads, head_dim, seq_len]
        // attn_scores: [batch, n_heads, seq_len, seq_len]
        let attn_scores = q
            .matmul(k.transpose())
            .div_scalar((self.head_dim as f32).sqrt());

        // ============ 步骤5: 应用softmax ============
        // 沿最后一个维度（seq_len）应用softmax
        // attn_weights: [batch, n_heads, seq_len, seq_len]
        let attn_weights = softmax(attn_scores, 3);

        // ============ 步骤6: 应用注意力到值 ============
        // output = attn_weights @ v
        // attn_weights: [batch, n_heads, seq_len, seq_len]
        // v: [batch, n_heads, seq_len, head_dim]
        // output: [batch, n_heads, seq_len, head_dim]
        let output = attn_weights.matmul(v);

        // ============ 步骤7: 重组并投影 ============
        // output: [batch, n_heads, seq_len, head_dim]
        // permute后: [batch, seq_len, n_heads, head_dim]
        // reshape后: [batch, seq_len, embedding_dim]
        let output = output
            .permute([0, 2, 1, 3]) // [batch, seq_len, n_heads, head_dim]
            .reshape([batch_size, seq_len, self.n_heads * self.head_dim]); // [batch, seq_len, embedding_dim]

        // 输出投影
        // x_atten: [batch, seq_len, embedding_dim]
        let x_atten = self.o_proj.forward(output);

        // ============ 步骤8: 残差连接和前馈网络 ============
        // 第一个残差连接: x + x_atten
        // shortcut: [batch, seq_len, embedding_dim]
        let shortcut = x + x_atten;

        // 后层归一化
        // x_norm2: [batch, seq_len, embedding_dim]
        let x_norm2 = self.rms_norm2.forward(shortcut.clone());

        // 前馈网络
        // x_feed: [batch, seq_len, embedding_dim]
        let x_feed = self.feed_forward.forward(x_norm2);

        // 第二个残差连接: shortcut + x_feed
        // 最终输出: [batch, seq_len, embedding_dim]
        shortcut + x_feed
    }
}

/// 大语言模型（LLM）
/// 
/// 这是完整的Transformer架构LLM，包含：
/// 1. Token嵌入层
/// 2. 多个注意力块（Transformer层）
/// 3. 最终归一化层
/// 4. 输出投影层
#[derive(Module, Debug)]
pub struct LLM<B: Backend> {
    embedding: Embedding<B>,                       // Token嵌入层
    attention_blocks: Vec<AttentionBlock<B>>,       // 注意力块堆叠
    final_rms: RmsNorm<B>,                    // 最终归一化层
    out_proj: Linear<B>,                        // 输出投影层
    max_context: usize,                          // 最大上下文长度
}

/// LLM配置
#[derive(Config, Debug)]
pub struct LLMConfig {
    pub vocab_size: usize,        // 词汇表大小
    pub embedding_dim: usize,      // 嵌入维度
    pub n_block: usize,          // 注意力块数（层数）
    pub n_head: usize,           // 注意力头数
    pub n_kv_head: usize,        // Key-Value注意力头数
    pub max_context: usize,       // 最大上下文长度
    pub hidden_dim: usize,        // 前馈网络隐藏层维度
    pub eps: f64,                // RMSNorm的epsilon
}

impl LLMConfig {
    #[allow(unused)]
    pub fn default() -> Self {
        Self::new(151669, 512, 8, 8, 4, 512, 1024, 1e-6)
    }
}

impl LLMConfig {
    /// 初始化LLM模型
    /// 
    /// # 参数说明
    /// * `device`: 张量所在的设备
    /// 
    /// # 返回
    /// 初始化好的LLM模型
    pub fn init<B: Backend>(&self, device: &B::Device) -> LLM<B> {
        // Token嵌入层
        // 维度: [vocab_size, embedding_dim]
        // 例如：[151669, 512]
        let embedding = EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device);

        // 创建多个注意力块
        let mut attention_blocks = Vec::new();
        for _ in 0..self.n_block {
            attention_blocks.push(
                AttentionBlockConfig::new(
                    self.embedding_dim,
                    self.n_head,
                    self.n_kv_head,
                    self.hidden_dim,
                    self.max_context,
                )
                .with_eps(self.eps)
                .init(device),
            );
        }

        // 最终归一化层
        // 维度: [embedding_dim]
        let final_rms = RmsNormConfig::new(self.embedding_dim)
            .with_epsilon(self.eps)
            .init(device);

        // 输出投影层
        // 维度: [embedding_dim, vocab_size]
        // 例如：[512, 151669]
        let out_proj = LinearConfig::new(self.embedding_dim, self.vocab_size).init(device);

        LLM {
            embedding,
            attention_blocks,
            final_rms,
            out_proj,
            max_context: self.max_context,
        }
    }
}

impl<B: Backend> LLM<B> {
    /// LLM的前向传播
    /// 
    /// # 数据维度变换流程
    /// 输入 x: [batch_size, seq_len]
    /// 
    /// 步骤1 - Token嵌入:
    /// x: [batch_size, seq_len, embedding_dim]
    /// 
    /// 步骤2 - 多个注意力块:
    /// 每个block将输入转换为新的表示
    /// x: [batch_size, seq_len, embedding_dim]（通过多个block后）
    /// 
    /// 步骤3 - 最终归一化:
    /// x: [batch_size, seq_len, embedding_dim]
    /// 
    /// 步骤4 - 展平:
    /// x_2d: [batch_size * seq_len, embedding_dim]
    /// 
    /// 步骤5 - 输出投影:
    /// logits: [batch_size * seq_len, vocab_size]
    /// 
    /// # 参数说明
    /// * `x`: 输入token ID张量，维度: [batch_size, seq_len]
    /// 
    /// # 返回
    /// 输出logits，维度: [batch_size * seq_len, vocab_size]
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        // ============ 步骤1: Token嵌入 ============
        // x: [batch_size, seq_len]
        // embedding输出: [batch_size, seq_len, embedding_dim]
        let mut x = self.embedding.forward(x);

        // ============ 步骤2: 多个注意力块 ============
        // 注意：当前未使用causal mask，实际应用中应该添加
        for block in &self.attention_blocks {
            // 每个block: [batch_size, seq_len, embedding_dim] → [batch_size, seq_len, embedding_dim]
            x = block.forward(x, None);
        }

        // ============ 步骤3: 最终归一化 ============
        // x: [batch_size, seq_len, embedding_dim]
        let x = self.final_rms.forward(x);

        // ============ 步骤4: 展平前两个维度 ============
        // x_2d: [batch_size * seq_len, embedding_dim]
        // 例如：[4 * 256, 512] = [1024, 512]
        let x_2d = x.flatten::<2>(0, 1);

        // ============ 步骤5: 输出投影到词汇表维度 ============
        // logits: [batch_size * seq_len, vocab_size]
        // 例如：[1024, 151669]
        self.out_proj.forward(x_2d)
    }

    /// 前向传播并计算分类损失
    /// 
    /// # 数据维度
    /// input_ids: [batch_size, seq_len]
    /// target_ids: [batch_size, seq_len]
    /// logits_flat: [batch_size * seq_len, vocab_size]
    /// targets_flat: [batch_size * seq_len]
    /// 
    /// # 损失计算
    /// 使用交叉熵损失计算预测与目标之间的差异
    /// 
    /// # 参数说明
    /// * `input_ids`: 输入token IDs
    /// * `target_ids`: 目标token IDs
    /// 
    /// # 返回
    /// 包含损失、logits和目标的ClassificationOutput
    pub fn forward_classification(
        &self,
        input_ids: Tensor<B, 2, Int>,
        target_ids: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        // 前向传播获取logits
        let logits = self.forward(input_ids);

        // ============ 展平以计算损失 ============
        // logits: [batch_size * seq_len, vocab_size]
        // logits_flat: [batch_size * seq_len, vocab_size]
        let logits_flat = logits.clone().flatten::<2>(0, 1);

        // target_ids: [batch_size, seq_len]
        // targets_2d: [batch_size * seq_len, embedding_dim]
        let targets_2d = target_ids.clone().flatten::<2>(0, 1);

        // targets_flat: [batch_size * seq_len]
        let targets_flat = targets_2d.flatten::<1>(0, 1);

        // ============ 计算交叉熵损失 ============
        // 损失衡量模型预测与真实目标的差异
        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits_flat.clone(), targets_flat.clone());

        ClassificationOutput::new(loss, logits_flat, targets_flat)
    }
}

// ============================================================================
// 训练相关实现
// ============================================================================

use burn::{
    data::dataloader::DataLoaderBuilder,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, TrainOutput, TrainStep, ValidStep, metric::LossMetric},
};

/// 实现TrainStep trait，定义训练步骤
impl<B: AutodiffBackend> TrainStep<TokenBatch<B>, ClassificationOutput<B>> for LLM<B> {
    /// 执行单个训练步骤
    /// 
    /// # 流程
    /// 1. 前向传播: 计算logits和损失
    /// 2. 反向传播: 计算梯度
    /// 3. 返回梯度供优化器更新参数
    /// 
    /// # 参数说明
    /// * `batch`: 批次数据
    ///   - input_ids: [batch_size, seq_len]
    ///   - target_ids: [batch_size, seq_len]
    /// 
    /// # 返回
    /// 包含梯度和输出的TrainOutput
    fn step(&self, batch: TokenBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        // 前向传播
        let output = self.forward_classification(batch.input_ids, batch.target_ids);

        // 反向传播计算梯度
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

/// 实现ValidStep trait，定义验证步骤
impl<B: Backend> ValidStep<TokenBatch<B>, ClassificationOutput<B>> for LLM<B> {
    /// 执行单个验证步骤
    /// 
    /// # 与训练步骤的区别
    /// - 不需要计算梯度
    /// - 只需前向传播计算损失
    /// 
    /// # 返回
    /// 包含损失的ClassificationOutput
    fn step(&self, batch: TokenBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.input_ids, batch.target_ids)
    }
}

/// 训练配置
#[derive(Config, Debug)]
pub struct LLMTrainingConfig {
    pub model: LLMConfig,           // 模型架构配置
    pub optimizer: AdamWConfig,      // 优化器配置
    #[config(default = 10)]
    pub num_epochs: usize,          // 训练轮数
    #[config(default = 8)]
    pub batch_size: usize,          // 批次大小
    #[config(default = 0)]
    pub num_workers: usize,         // 数据加载线程数
    #[config(default = 42)]
    pub seed: u64,                 // 随机种子
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,         // 学习率
}

/// 创建训练输出目录
fn create_artifact_dir(artifact_dir: &str) {
    // 删除现有目录以确保干净的开始
    std::fs::remove_dir_all(artifact_dir).ok();
    // 创建新目录
    std::fs::create_dir_all(artifact_dir).ok();
}

/// LLM训练主函数
/// 
/// # 训练流程
/// 1. 创建输出目录
/// 2. 保存配置
/// 3. 设置随机种子
/// 4. 创建数据加载器
/// 5. 构建学习器
/// 6. 执行训练
/// 
/// # 参数说明
/// * `artifact_dir`: 训练输出目录
/// * `config`: 训练配置
/// * `train_dataset`: 训练数据集
/// * `val_dataset`: 验证数据集
/// * `device`: 计算设备
pub fn train_llm<B: AutodiffBackend>(
    artifact_dir: &str,
    config: LLMTrainingConfig,
    train_dataset: TokenDataset,
    val_dataset: TokenDataset,
    device: B::Device,
) {
    // ============ 步骤1: 创建输出目录 ============
    create_artifact_dir(artifact_dir);

    // ============ 步骤2: 保存配置 ============
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    // ============ 步骤3: 设置随机种子 ============
    // 确保结果可复现
    B::seed(&device, config.seed);

    // ============ 步骤4: 创建批次处理器 ============
    let batcher = TokenBatcher;

    // ============ 步骤5: 创建训练数据加载器 ============
    // 数据维度变换：
    // - 从数据集获取样本: [seq_len], [seq_len]
    // - 批次化: [batch_size, seq_len], [batch_size, seq_len]
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)      // 批次大小，例如4
        .shuffle(config.seed)             // 打乱数据
        .num_workers(config.num_workers)   // 工作线程数，0表示不使用多线程
        .build(train_dataset);

    // ============ 步骤6: 创建验证数据加载器 ============
    let dataloader_val = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(val_dataset);

    // ============ 步骤7: 构建学习器 ============
    // Learner负责管理整个训练流程：
    // - 前向传播
    // - 反向传播
    // - 参数更新
    // - 验证
    // - 模型保存
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())        // 训练损失指标
        .metric_valid_numeric(LossMetric::new())         // 验证损失指标
        .with_file_checkpointer(CompactRecorder::new())  // 模型检查点
        .learning_strategy(burn::train::LearningStrategy::SingleDevice(device.clone()))  // 单设备策略
        .num_epochs(config.num_epochs)               // 训练轮数
        .summary()                                   // 打印摘要
        .build(
            config.model.init::<B>(&device),    // 初始化模型
            config.optimizer.init(),               // 初始化优化器
            config.learning_rate,               // 学习率
        );

    // ============ 步骤8: 执行训练 ============
    // 训练循环：
    // - 遍历每个epoch
    // - 遍历每个batch
    // - 执行前向传播
    // - 执行反向传播
    // - 更新参数
    // - 定期在验证集上评估
    let _model_trained = learner.fit(dataloader_train, dataloader_val);
}