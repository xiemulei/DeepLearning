use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};
use clap::{Parser, Subcommand};

use crate::{llm_train::train_llm_main, model::ModelConfig, training::TrainingConfig};

mod data;
mod inference;
mod llm;
mod llm_train;
mod model;
mod rope;
mod training;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 训练图像分类模型
    TrainImage,
    /// 训练大语言模型
    TrainLlm {
        /// 训练文本文件路径
        #[arg(short, long)]
        text: String,
        /// Tokenizer文件路径
        #[arg(short, long)]
        tokenizer: String,
        /// 模型保存目录
        #[arg(short, long, default_value = "../tmp/llm")]
        artifact_dir: String,
    },
    /// 图像推理
    Infer {
        /// 图像路径
        #[arg(short, long)]
        image: String,
        /// 图像大小
        #[arg(short, long, default_value = "128")]
        size: usize,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::TrainImage => {
            train_image_classification();
        }
        Commands::TrainLlm {
            text,
            tokenizer,
            artifact_dir,
        } => {
            train_llm_main(&text, &tokenizer, &artifact_dir);
        }
        Commands::Infer { image, size } => {
            run_inference(&image, size);
        }
    }
}

fn train_image_classification() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "../tmp/guide";

    println!("开始训练图像分类模型...");

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
        device.clone(),
    );

    println!("图像分类模型训练完成！");

    let pred = crate::inference::infer::<MyBackend>(
        artifact_dir,
        &device,
        r"/Users/x/Rust/DeepLearning/PetImages/Cat/1.jpg",
        128,
    )
    .unwrap();

    println!("预测结果: {pred} (0=Cat, 1=Dog)");
}

fn run_inference(image_path: &str, img_size: usize) {
    type MyBackend = Wgpu<f32, i32>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "../tmp/guide";

    println!("正在推理: {}", image_path);

    let pred =
        crate::inference::infer::<MyBackend>(artifact_dir, &device, image_path, img_size).unwrap();

    println!("预测结果: {pred} (0=Cat, 1=Dog)");
}
