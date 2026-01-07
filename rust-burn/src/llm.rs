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

#[derive(Debug, Clone)]
pub struct TokenDatasetItem {
    pub input_ids: Vec<i64>,
    pub target_ids: Vec<i64>,
}

pub struct TokenDataset {
    items: Vec<TokenDatasetItem>,
    tokenizer: Tokenizer,
    seq_len: usize,
}

impl TokenDataset {
    /// 使用tokenizer创建数据集
    pub fn new_with_tokenizer(
        text_path: &str,
        tokenizer_path: &str,
        seq_len: usize,
        stride: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // 读取tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        // 读取文本文件
        let text = fs::read_to_string(text_path)?;

        // 使用tokenizer编码文本
        let encoding = tokenizer.encode(text.as_str(), false)?;
        let tokens: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        let token_len = tokens.len();

        // Need at least seq_len + 1 tokens (input + target)
        if token_len < seq_len + 1 {
            return Err(format!(
                "Not enough tokens: {} tokens available, need at least {} (seq_len={} + 1 target)",
                token_len,
                seq_len + 1,
                seq_len
            )
            .into());
        }

        // The last valid starting index is token_len - seq_len - 1
        let max_start_idx = token_len - seq_len - 1;
        let effective_stride = if stride == 0 { 1 } else { stride };

        let mut items = Vec::new();

        // 滑动窗口创建训练样本
        for i in (0..=max_start_idx).step_by(effective_stride) {
            // Ensure we have enough tokens for both input and target
            if i + seq_len + 1 > token_len {
                break;
            }

            let input_ids = tokens[i..i + seq_len].to_vec();
            let target_ids = tokens[i + 1..i + seq_len + 1].to_vec();

            items.push(TokenDatasetItem {
                input_ids,
                target_ids,
            });
        }

        if items.is_empty() {
            // Create at least one sample if possible
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
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

impl Dataset<TokenDatasetItem> for TokenDataset {
    fn get(&self, index: usize) -> Option<TokenDatasetItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone, Default)]
pub struct TokenBatcher;

#[derive(Clone, Debug)]
pub struct TokenBatch<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,
    pub target_ids: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, TokenDatasetItem, TokenBatch<B>> for TokenBatcher {
    fn batch(&self, items: Vec<TokenDatasetItem>, device: &B::Device) -> TokenBatch<B> {
        let input_ids = items
            .iter()
            .map(|item| {
                TensorData::new(item.input_ids.clone(), [item.input_ids.len()])
                    .convert::<B::IntElem>()
            })
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device).unsqueeze_dim(0))
            .collect();

        let target_ids = items
            .iter()
            .map(|item| {
                TensorData::new(item.target_ids.clone(), [item.target_ids.len()])
                    .convert::<B::IntElem>()
            })
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device).unsqueeze_dim(0))
            .collect();

        let input_ids: Tensor<B, 2, Int> = Tensor::cat(input_ids, 0);
        let target_ids: Tensor<B, 2, Int> = Tensor::cat(target_ids, 0);

        TokenBatch {
            input_ids,
            target_ids,
        }
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    up: Linear<B>,
    gate: Linear<B>,
    down: Linear<B>,
    activation: Gelu,
}

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            up: LinearConfig::new(self.in_dim, self.hidden_dim).init(device),
            gate: LinearConfig::new(self.in_dim, self.hidden_dim).init(device),
            down: LinearConfig::new(self.hidden_dim, self.out_dim).init(device),
            activation: Gelu::new(),
        }
    }
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x_clone = x.clone();
        let up_x = self.up.forward(x_clone);
        let gate_x = self.activation.forward(self.gate.forward(x));
        let mul_cat = up_x * gate_x;
        self.down.forward(mul_cat)
    }
}

#[derive(Module, Debug)]
pub struct AttentionBlock<B: Backend> {
    rms_norm1: RmsNorm<B>,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    rope: RotaryEncoding<B>,
    n_heads: usize,
    head_dim: usize,
    rms_norm2: RmsNorm<B>,
    feed_forward: FeedForward<B>,
}

#[derive(Config, Debug)]
pub struct AttentionBlockConfig {
    embedding_dim: usize,
    n_head: usize,
    n_kv_head: usize,
    hidden_dim: usize,
    max_context: usize,
    #[config(default = 1e-6)]
    eps: f64,
}

impl AttentionBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AttentionBlock<B> {
        // Note: This implementation does not support Grouped Query Attention (GQA),
        // so we use n_head for both queries and key-value heads
        assert_eq!(
            self.n_head, self.n_kv_head,
            "AttentionBlock requires n_head == n_kv_head (does not support GQA)"
        );

        let head_dim = self.embedding_dim / self.n_head;

        AttentionBlock {
            rms_norm1: RmsNormConfig::new(self.embedding_dim)
                .with_epsilon(self.eps)
                .init(device),
            q_proj: LinearConfig::new(self.embedding_dim, self.embedding_dim).init(device),
            k_proj: LinearConfig::new(self.embedding_dim, self.embedding_dim).init(device),
            v_proj: LinearConfig::new(self.embedding_dim, self.embedding_dim).init(device),
            o_proj: LinearConfig::new(self.embedding_dim, self.embedding_dim).init(device),
            rope: RotaryEncodingConfig::new(self.max_context, head_dim)
                .with_theta(10000.0)
                .init(device),
            n_heads: self.n_head,
            head_dim,
            rms_norm2: RmsNormConfig::new(self.embedding_dim)
                .with_epsilon(self.eps)
                .init(device),
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
    pub fn forward(&self, x: Tensor<B, 3>, _mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let x_norm1 = self.rms_norm1.forward(x.clone());
        let [batch_size, seq_len, _] = x_norm1.dims();

        // Project to Q, K, V
        let q = self
            .q_proj
            .forward(x_norm1.clone())
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3]); // [batch, n_heads, seq_len, head_dim]
        let k = self
            .k_proj
            .forward(x_norm1.clone())
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3]); // [batch, n_heads, seq_len, head_dim]
        let v = self
            .v_proj
            .forward(x_norm1.clone())
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3]); // [batch, n_heads, seq_len, head_dim]

        // Apply RoPE to Q and K
        let q = self.rope.forward(q); // [batch, n_heads, seq_len, head_dim]
        let k = self.rope.forward(k); // [batch, n_heads, seq_len, head_dim]

        // Compute attention scores
        let attn_scores = q
            .matmul(k.transpose())
            .div_scalar((self.head_dim as f32).sqrt()); // [batch, n_heads, seq_len, seq_len]

        // Apply softmax
        let attn_weights = softmax(attn_scores, 3);

        // Apply attention to values
        let output = attn_weights.matmul(v); // [batch, n_heads, seq_len, head_dim]
        let output = output
            .permute([0, 2, 1, 3]) // [batch, seq_len, n_heads, head_dim]
            .reshape([batch_size, seq_len, self.n_heads * self.head_dim]); // [batch, seq_len, embedding_dim]

        // Output projection
        let x_atten = self.o_proj.forward(output);

        let shortcut = x + x_atten;
        let x_norm2 = self.rms_norm2.forward(shortcut.clone());
        let x_feed = self.feed_forward.forward(x_norm2);
        shortcut + x_feed
    }
}

#[derive(Module, Debug)]
pub struct LLM<B: Backend> {
    embedding: Embedding<B>,
    attention_blocks: Vec<AttentionBlock<B>>,
    final_rms: RmsNorm<B>,
    out_proj: Linear<B>,
    max_context: usize,
}

#[derive(Config, Debug)]
pub struct LLMConfig {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub n_block: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub max_context: usize,
    pub hidden_dim: usize,
    pub eps: f64,
}

impl LLMConfig {
    #[allow(unused)]
    pub fn default() -> Self {
        Self::new(151669, 512, 8, 8, 4, 512, 1024, 1e-6)
    }
}

impl LLMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LLM<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device);
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
        let final_rms = RmsNormConfig::new(self.embedding_dim)
            .with_epsilon(self.eps)
            .init(device);
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
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let mut x = self.embedding.forward(x);

        // For now, we'll pass None as mask - in a real implementation you'd create a causal mask
        for block in &self.attention_blocks {
            x = block.forward(x, None);
        }

        let x = self.final_rms.forward(x);
        let x_2d = x.flatten::<2>(0, 1);
        self.out_proj.forward(x_2d)
    }

    pub fn forward_classification(
        &self,
        input_ids: Tensor<B, 2, Int>,
        target_ids: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let logits = self.forward(input_ids);
        // Flatten the logits and targets for cross-entropy loss
        let logits_flat = logits.clone().flatten::<2>(0, 1);
        let targets_2d = target_ids.clone().flatten::<2>(0, 1);
        let targets_flat = targets_2d.flatten::<1>(0, 1);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits_flat.clone(), targets_flat.clone());

        ClassificationOutput::new(loss, logits_flat, targets_flat)
    }
}

// Training implementation
use burn::{
    data::dataloader::DataLoaderBuilder,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, TrainOutput, TrainStep, ValidStep, metric::LossMetric},
};

impl<B: AutodiffBackend> TrainStep<TokenBatch<B>, ClassificationOutput<B>> for LLM<B> {
    fn step(&self, batch: TokenBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let output = self.forward_classification(batch.input_ids, batch.target_ids);
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<TokenBatch<B>, ClassificationOutput<B>> for LLM<B> {
    fn step(&self, batch: TokenBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.input_ids, batch.target_ids)
    }
}

#[derive(Config, Debug)]
pub struct LLMTrainingConfig {
    pub model: LLMConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 0)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before getting accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train_llm<B: AutodiffBackend>(
    artifact_dir: &str,
    config: LLMTrainingConfig,
    train_dataset: TokenDataset,
    val_dataset: TokenDataset,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let batcher = TokenBatcher;

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_val = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(val_dataset);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .learning_strategy(burn::train::LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let _model_trained = learner.fit(dataloader_train, dataloader_val);
}
