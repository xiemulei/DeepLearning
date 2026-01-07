use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};

use crate::llm::{LLMConfig, LLMTrainingConfig, TokenDataset, train_llm};

/// 训练LLM的主函数
pub fn train_llm_main(text_path: &str, tokenizer_path: &str, artifact_dir: &str) {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    println!("正在加载数据集...");

    // 创建训练数据集
    let train_dataset = TokenDataset::new_with_tokenizer(
        text_path,
        tokenizer_path,
        512, // seq_len
        256, // stride
    )
    .expect("Failed to create train dataset");

    // 创建验证数据集（使用相同的文本，但在实际应用中应该使用不同的数据）
    let val_dataset = TokenDataset::new_with_tokenizer(
        text_path,
        tokenizer_path,
        512, // seq_len
        256, // stride
    )
    .expect("Failed to create validation dataset");

    // 获取词汇表大小
    let vocab_size = train_dataset.vocab_size();
    println!("词汇表大小: {}", vocab_size);

    // 模型配置
    let model_config = LLMConfig {
        vocab_size,
        embedding_dim: 512,
        n_block: 6,
        n_head: 8,
        n_kv_head: 8, // 必须等于n_head（不支持GQA）
        max_context: 512,
        hidden_dim: 2048, // 通常4倍于embedding_dim
        eps: 1e-6,
    };

    // 训练配置
    let training_config = LLMTrainingConfig {
        model: model_config,
        optimizer: AdamConfig::new(),
        num_epochs: 10,
        batch_size: 4,
        num_workers: 0,
        seed: 42,
        learning_rate: 1.0e-4,
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

    // 开始训练
    train_llm::<MyAutodiffBackend>(
        artifact_dir,
        training_config,
        train_dataset,
        val_dataset,
        device,
    );

    println!("训练完成！模型保存在: {}", artifact_dir);
}
