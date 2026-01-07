# 大语言模型训练指南

本文档说明如何使用Burn框架训练大语言模型（LLM）。

## 功能特性

- ✅ 使用tokenizers进行文本分词
- ✅ 支持标准的Transformer架构
- ✅ 集成RotaryEncoding位置编码
- ✅ 命令行界面支持
- ✅ 完整的训练流程

## 模型架构

当前实现的LLM包含：

- **Embedding层**: 将token IDs转换为向量表示
- **AttentionBlock**: 自注意力机制
  - RMSNorm归一化
  - 多头注意力（支持RoPE）
  - 残差连接
  - GLU前馈网络
- **输出层**: 将隐藏状态映射到词汇表

## 快速开始

### 1. 准备数据

确保你有以下文件：
- 训练文本文件（例如：`src/assets/sub_wiki_0_99.txt`）
- Tokenizer JSON文件（例如：`src/assets/tokenizer.json`）

### 2. 训练LLM

```bash
cd rust-burn

# 训练LLM
cargo run --release -- train-llm   --text src/assets/sub_wiki_0_99.txt   --tokenizer src/assets/tokenizer.json   --artifact-dir ../tmp/llm
```

### 3. 查看训练进度

训练过程中会显示：
- 训练和验证损失
- 每个epoch的指标
- 模型检查点保存

## 命令行选项

### train-llm - 训练大语言模型

```bash
cargo run --release -- train-llm [OPTIONS]
```

**选项:**
- `--text <TEXT>` - 训练文本文件路径（必需）
- `--tokenizer <TOKENIZER>` - Tokenizer文件路径（必需）
- `--artifact-dir <DIR>` - 模型保存目录（默认：`../tmp/llm`）

### 其他命令

```bash
# 训练图像分类模型
cargo run --release -- train-image

# 图像推理
cargo run --release -- infer --image <path> --size <size>
```

## 模型配置

默认配置在`llm_train.rs`中：

```rust
LLMConfig {
    vocab_size: 从tokenizer获取,
    embedding_dim: 512,
    n_block: 6,           // Transformer层数
    n_head: 8,            // 注意力头数
    n_kv_head: 8,          // 必须等于n_head
    max_context: 512,       // 最大序列长度
    hidden_dim: 2048,      // 前馈网络隐藏层维度
    eps: 1e-6,           // RMSNorm epsilon
}
```

训练配置：

```rust
LLMTrainingConfig {
    num_epochs: 10,        // 训练轮数
    batch_size: 4,         // 批次大小
    learning_rate: 1e-4,   // 学习率
    optimizer: Adam,        // 优化器
}
```

## 自定义配置

编辑`src/llm_train.rs`中的`train_llm_main`函数来自定义配置：

```rust
// 更大的模型
let model_config = LLMConfig {
    vocab_size,
    embedding_dim: 768,      // 增加嵌入维度
    n_block: 12,              // 增加层数
    n_head: 12,               // 增加注意力头数
    n_kv_head: 12,
    max_context: 1024,        // 增加上下文长度
    hidden_dim: 3072,         // 4倍于embedding_dim
    eps: 1e-6,
};

// 更长的训练
let training_config = LLMTrainingConfig {
    num_epochs: 20,
    batch_size: 8,
    learning_rate: 5e-5,
    // ...
};
```

## 训练数据准备

### 使用现有Tokenizer

```rust
let dataset = TokenDataset::new_with_tokenizer(
    "path/to/text.txt",
    "path/to/tokenizer.json",
    seq_len,    // 序列长度（例如512）
    stride,      // 滑动窗口步长（例如256）
)?;
```

### 创建自定义Tokenizer

使用HuggingFace tokenizers库：

```bash
pip install tokenizers
python -c "from tokenizers import Tokenizer; t = Tokenizer.from_pretrained('bert-base-uncased'); t.save('tokenizer.json')"
```

## 模型架构说明

### Attention Block

每个Attention Block包含：

1. **Pre-LN RMSNorm**: 归一化输入
2. **Multi-Head Self-Attention**:
   - Q, K, V投影
   - RotaryEncoding位置编码
   - 注意力分数计算和softmax
   - 残差连接
3. **Post-LN RMSNorm**: 归一化注意力输出
4. **GLU Feed-Forward**:
   - Gated Linear Unit结构
   - 2倍于embedding_dim的隐藏层
   - 残差连接

### 注意事项

1. **不支持GQA**: 当前实现要求`n_head == n_kv_head`
2. **序列长度**: 受限于`max_context`配置
3. **内存使用**: 较大的batch_size和序列长度需要更多GPU内存

## 性能优化建议

1. **调整batch_size**: 根据GPU内存调整
   - 8GB GPU: batch_size=2-4
   - 16GB GPU: batch_size=4-8
   - 32GB GPU: batch_size=8-16

2. **序列长度**:
   - 训练：512-1024
   - 推理：可达2048+

3. **梯度累积**（未来实现）:
   - 使用小batch_size但累积梯度
   - 模拟大批次训练

4. **混合精度训练**（未来实现）:
   - 使用f16/bf16减少内存
   - 加速训练

## 故障排除

### 内存不足

```rust
// 减小batch_size
batch_size: 2,

// 或减小序列长度
seq_len: 256,
```

### 训练不稳定

```rust
// 降低学习率
learning_rate: 5e-5,

// 使用梯度裁剪（需要手动实现）
```

### 损失不下降

- 检查学习率是否过大
- 确认数据预处理正确
- 验证tokenizer和文本编码匹配

## 下一步

- [ ] 实现文本生成功能
- [ ] 添加Causal Mask支持
- [ ] 实现梯度累积
- [ ] 添加混合精度训练
- [ ] 实现模型量化
- [ ] 添加更多评估指标（Perplexity等）

## 参考资源

- Burn框架: https://burn.dev
- Transformer论文: https://arxiv.org/abs/1706.03762
- RoPE论文: https://arxiv.org/abs/2104.09864
- Tokenizers库: https://github.com/huggingface/tokenizers