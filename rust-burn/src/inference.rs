use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
    tensor::TensorData,
};

use crate::training::TrainingConfig;

pub fn infer<B: Backend<FloatElem = f32>>(
    artifact_dir: &str,
    device: &B::Device,
    image_path: &str,
    img_size: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Train model should exist; run train first");

    let model = config.model.init::<B>(device).load_record(record);

    use image;

    // 加载和预处理图片
    let img = image::open(image_path)?.to_rgb8();
    let img = image::imageops::resize(
        &img,
        img_size as u32,
        img_size as u32,
        image::imageops::FilterType::Lanczos3,
    );

    // 转换为张量
    let mut data = Vec::with_capacity(3 * img_size * img_size);
    for pixel in img.pixels() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        data.push((r - 0.485) / 0.229);
        data.push((g - 0.456) / 0.224);
        data.push((b - 0.406) / 0.225);
    }

    let tensor_data = TensorData::new(data, [img_size, img_size, 3]).convert::<B::FloatElem>();
    let image_tensor = Tensor::<B, 3>::from_data(tensor_data, device)
        .unsqueeze_dim(0) // [1, height, width, channels]
        .swap_dims(1, 3) // [1, channels, width, height]
        .swap_dims(2, 3); // [1, channels, height, width]

    // 推理
    let output = model.forward(image_tensor);
    let probability = output.into_scalar();

    Ok(probability.into())
}
