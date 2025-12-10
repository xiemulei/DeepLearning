use crate::{
    data::{CatDogBatch, CatDogBatcher, CatDogDataset},
    model::{Model, ModelConfig},
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    nn::loss::BinaryCrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::{
        Int, Tensor,
        backend::{AutodiffBackend, Backend},
    },
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 0)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
    #[config(default = 128)]
    pub img_size: usize,
    #[config(default = 0.8)]
    pub train_ratio: f32,
    #[config(default = 0.5)]
    pub dropout: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // 在获取准确的学习器摘要之前移除现有工件
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let data_dir = r"/Users/x/Rust/DeepLearning/PetImages";
    // 创建数据集
    println!("正在加载数据集...");
    let dataset = CatDogDataset::new(data_dir, config.img_size).expect("Failed to create dataset");

    let (train_dataset, valid_dataset) = dataset.split(config.train_ratio);
    println!(
        "训练集大小: {}, 验证集大小: {}",
        train_dataset.len(),
        valid_dataset.len()
    );

    let batcher = CatDogBatcher {
        img_size: config.img_size,
    };

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    let model_config = ModelConfig::new().with_dropout(config.dropout);
    let optimizer_config = AdamConfig::new();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .learning_strategy(burn::train::LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            model_config.init::<B>(&device),
            optimizer_config.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    println!("训练完成！模型已保存到 {}/model", artifact_dir);
}

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let targets_2d = targets.clone().reshape([output.dims()[0], 1]);
        let loss = BinaryCrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets_2d);

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<CatDogBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: CatDogBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<CatDogBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: CatDogBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
