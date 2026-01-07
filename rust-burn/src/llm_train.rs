use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamWConfig,
};

use crate::llm::{LLMConfig, LLMTrainingConfig, TokenDataset, train_llm};

/// LLM训练主函数 - 整个训练流程的入口
/// 
/// 训练流程概述：
/// 1. 数据准备：将文本转换为token序列，使用滑动窗口创建训练样本
/// 2. 模型初始化：配置并初始化Transformer架构的LLM
/// 3. 训练循环：使用数据加载器批量训练模型
/// 4. 验证评估：定期在验证集上评估模型性能
/// 5. 模型保存：保存训练好的模型参数
/// 
/// 数据流：
/// 文本 → Token序列 → 滑动窗口切片 → 批次处理 → 模型前向传播 → 损失计算 → 反向传播
/// 
/// # 参数说明
/// * `text_path`: 训练文本文件路径
/// * `tokenizer_path`: tokenizer配置文件路径（JSON格式）
/// * `artifact_dir`: 模型和训练日志的保存目录
pub fn train_llm_main(text_path: &str, tokenizer_path: &str, artifact_dir: &str) {
    // ============ 后端配置 ============
    // MyBackend: WGPU后端，使用f32作为精度，i32作为整数类型
    // AutodiffBackend: 自动微分后端，用于计算梯度
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    // 初始化GPU设备（使用默认GPU）
    let device = burn::backend::wgpu::WgpuDevice::default();

    println!("正在加载数据集...");

    // ============ 数据集创建 ============
    // 创建训练数据集
    // 数据维度信息：
    // - 输入文本: 字符串，长度约10MB（从文件读取）
    // - Token序列: [N]，N为token总数（约2-3M tokens）
    // - 单个样本: [seq_len] = [256]，包含256个token ID
    // - 单个样本的target: [seq_len] = [256]，每个位置预测下一个token
    // - 数据集大小: 约 (N - seq_len) / stride 个样本
    let train_dataset = TokenDataset::new_with_tokenizer(
        text_path,
        tokenizer_path,
        256, // seq_len: 序列长度，每个样本包含256个token
        256, // stride: 滑动窗口步长，每256个token创建一个样本（无重叠）
    )
    .expect("Failed to create train dataset");

    // 创建验证数据集
    // 注意：在实际应用中，应该使用与训练集不同的数据
    // 这里为了简化，使用了相同的文本，但通常应该划分独立的验证集
    let val_dataset = TokenDataset::new_with_tokenizer(
        text_path,
        tokenizer_path,
        256, // seq_len: 序列长度
        256, // stride: 滑动窗口步长
    )
    .expect("Failed to create validation dataset");

    // 获取词汇表大小
    // 词汇表大小决定了模型的输出层维度和embedding层的行数
    let vocab_size = train_dataset.vocab_size();
    println!("词汇表大小: {}", vocab_size);

    // ============ 模型配置 ============
    // LLMConfig定义了模型的架构参数
    let model_config = LLMConfig {
        vocab_size,              // 词汇表大小，例如151669
        embedding_dim: 512,      // 嵌入维度，每个token表示为512维向量
        n_block: 6,            // Transformer块的数量（层数）
        n_head: 8,             // 多头注意力的头数
        n_kv_head: 8,          // Key-Value注意力头数（当前实现必须等于n_head，不支持GQA）
        max_context: 512,       // 最大上下文长度，模型能处理的最大序列长度
        hidden_dim: 2048,       // 前馈网络的隐藏层维度（通常是embedding_dim的4倍）
        eps: 1e-6,            // RMSNorm中的epsilon，防止除零
    };

    // ============ 训练配置 ============
    // LLMTrainingConfig定义了训练超参数
    let training_config = LLMTrainingConfig {
        model: model_config,                // 模型架构配置
        optimizer: AdamWConfig::new(),     // 优化器配置（AdamW）
        num_epochs: 10,                   // 训练轮数（完整遍历数据集10次）
        batch_size: 4,                    // 批次大小，每次处理4个样本
        // 批次数据维度：[batch_size, seq_len] = [4, 256]
        // 批次target维度：[batch_size, seq_len] = [4, 256]
        num_workers: 0,                   // 数据加载的线程数（0表示不使用多线程）
        seed: 42,                        // 随机种子，确保结果可复现
        learning_rate: 1.0e-4,           // 学习率，控制参数更新步长
    };

    println!("开始训练...");
    println!("训练配置:");
    println!("  - 词汇表大小: {}", training_config.model.vocab_size);
    println!("  - 嵌入维度: {}", training_config.model.embedding_dim);
    println!("  - 注意力块数: {}", training_config.model.n_block);
    println!("  - 注意力头数: {}", training_config.model.n_head);
    println!("  - 最大上下文: {}", training_config.model.max_context);
    println!("  - 训练轮数: {}", training_config.num_epochs);
    println!("  - 批次大小: {}", training_config.batch_size);
    println!("  - 学习率: {}", training_config.learning_rate);

    // ============ 开始训练 ============
    // train_llm函数执行完整的训练流程：
    // 1. 创建数据加载器（DataLoader）
    // 2. 初始化模型和优化器
    // 3. 创建学习器（Learner）
    // 4. 执行训练循环
    train_llm::<MyAutodiffBackend>(
        artifact_dir,
        training_config,
        train_dataset,
        val_dataset,
        device,
    );

    println!("训练完成！模型保存在: {}", artifact_dir);
}
