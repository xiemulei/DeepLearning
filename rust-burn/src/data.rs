use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use std::path::PathBuf;
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct CatDogItem {
    pub image_path: PathBuf,
    pub label: i32, // 0 for cat, 1 for dog
}

pub struct CatDogDataset {
    items: Vec<CatDogItem>,
    img_size: usize,
}

impl CatDogDataset {
    pub fn new(data_dir: &str, img_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut items = Vec::new();

        // 处理 Cat 文件夹
        let cat_dir = PathBuf::from(data_dir).join("Cat");
        if cat_dir.exists() {
            for entry in WalkDir::new(&cat_dir).into_iter().flatten() {
                if let Some(extension) = entry.path().extension() {
                    if let Some(ext_str) = extension.to_str() {
                        if ["jpg", "jpeg", "png"].contains(&ext_str.to_lowercase().as_str()) {
                            // 尝试加载图片以验证它是否有效
                            if image::open(entry.path()).is_ok() {
                                items.push(CatDogItem {
                                    image_path: entry.path().to_path_buf(),
                                    label: 0, // Cat
                                });
                            }
                        }
                    }
                }
            }
        }

        // 处理 Dog 文件夹
        let dog_dir = PathBuf::from(data_dir).join("Dog");
        if dog_dir.exists() {
            for entry in WalkDir::new(&dog_dir).into_iter().flatten() {
                if let Some(extension) = entry.path().extension() {
                    if let Some(ext_str) = extension.to_str() {
                        if ["jpg", "jpeg", "png"].contains(&ext_str.to_lowercase().as_str()) {
                            // 尝试加载图片以验证它是否有效
                            if image::open(entry.path()).is_ok() {
                                items.push(CatDogItem {
                                    image_path: entry.path().to_path_buf(),
                                    label: 1, // Dog
                                });
                            }
                        }
                    }
                }
            }
        }

        println!("找到 {} 张有效图片", items.len());
        Ok(CatDogDataset { items, img_size })
    }

    pub fn split(self, train_ratio: f32) -> (Self, Self) {
        let total_len = self.items.len();
        let train_len = (total_len as f32 * train_ratio) as usize;

        let mut items = self.items;
        use rand::rng;
        use rand::seq::SliceRandom;
        items.shuffle(&mut rng());

        let (train_items, valid_items) = items.split_at(train_len);

        let train_dataset = CatDogDataset {
            items: train_items.to_vec(),
            img_size: self.img_size,
        };

        let valid_dataset = CatDogDataset {
            items: valid_items.to_vec(),
            img_size: self.img_size,
        };

        (train_dataset, valid_dataset)
    }
}

impl Dataset<CatDogItem> for CatDogDataset {
    fn get(&self, index: usize) -> Option<CatDogItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone, Default)]
pub struct CatDogBatcher {
    pub img_size: usize, // 图片缩放到同一尺寸
}

#[derive(Clone, Debug)]
pub struct CatDogBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, CatDogItem, CatDogBatch<B>> for CatDogBatcher {
    fn batch(&self, items: Vec<CatDogItem>, device: &B::Device) -> CatDogBatch<B> {
        let images = items
            .iter()
            .map(|item| {
                // 加载图片
                let img = image::open(&item.image_path)
                    .expect("Failed to load image")
                    .to_rgb8();

                // 调整大小
                let img = image::imageops::resize(
                    &img,
                    self.img_size as u32,
                    self.img_size as u32,
                    image::imageops::FilterType::Lanczos3,
                );

                // 转换为张量数据
                let mut data = Vec::with_capacity(3 * self.img_size * self.img_size);
                for pixel in img.pixels() {
                    // 归一化到 [0, 1] 并应用 ImageNet 标准化
                    let r = pixel[0] as f32 / 255.0;
                    let g = pixel[1] as f32 / 255.0;
                    let b = pixel[2] as f32 / 255.0;

                    // ImageNet 标准化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    data.push((r - 0.485) / 0.229);
                    data.push((g - 0.456) / 0.224);
                    data.push((b - 0.406) / 0.225);
                }

                TensorData::new(data, [self.img_size, self.img_size, 3]).convert::<B::FloatElem>()
            })
            .map(|data| Tensor::<B, 3>::from_data(data, device))
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            })
            .collect();

        let images = Tensor::stack(images, 0).permute([0, 3, 1, 2]); // [batch_size, channels, height, width]
        let targets = Tensor::cat(targets, 0);

        CatDogBatch { images, targets }
    }
}
