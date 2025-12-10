use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};

use crate::{model::ModelConfig, training::TrainingConfig};

mod data;
mod inference;
mod model;
mod training;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "../tmp/guide";

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
        device.clone(),
    );

    let pred = crate::inference::infer::<MyBackend>(
        artifact_dir,
        &device,
        r"/Users/x/Rust/DeepLearning/PetImages/Cat/1.jpg",
        128,
    )
    .unwrap();

    println!("Predicate: {pred}");
}
