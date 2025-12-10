use burn::{
    nn::{
        Dropout, DropoutConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    max_pool1: MaxPool2d,
    conv2: Conv2d<B>,
    max_pool2: MaxPool2d,
    conv3: Conv2d<B>,
    max_pool3: MaxPool2d,
    conv4: Conv2d<B>,
    max_pool4: MaxPool2d,
    conv5: Conv2d<B>,
    avg_pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    relu: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 0.5)]
    dropout: f64,
}

impl ModelConfig {
    /// 返回初始化的模型。
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            // 输入: [batch_size, 3, 128, 128]
            conv1: Conv2dConfig::new([3, 16], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device), // -> [batch_size, 16, 128, 128]
            max_pool1: MaxPool2dConfig::new([2, 2]).init(),
            conv2: Conv2dConfig::new([16, 32], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device), // -> [batch_size, 32, 64, 64]
            max_pool2: MaxPool2dConfig::new([2, 2]).init(),
            conv3: Conv2dConfig::new([32, 64], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device), // -> [batch_size, 64, 32, 32]
            max_pool3: MaxPool2dConfig::new([2, 2]).init(),
            conv4: Conv2dConfig::new([64, 128], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device), // -> [batch_size, 128, 16, 16]
            max_pool4: MaxPool2dConfig::new([2, 2]).init(),
            conv5: Conv2dConfig::new([128, 1], [1, 1]).init(device), // -> [batch_size, 1, 8, 8]
            avg_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),   // -> [batch_size, 1, 1, 1]
            relu: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # 形状
    ///   - 图像 [batch_size, channels, height, width]
    ///   - 输出 [batch_size, 1]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, _channels, _height, _width] = images.dims();

        // 第一层卷积块
        let x = self.conv1.forward(images); // [batch_size, 16, height, width]
        let x = self.relu.forward(x);
        let x = self.max_pool1.forward(x);

        // 第二层卷积块
        let x = self.conv2.forward(x); // [batch_size, 32, height/2, width/2]
        let x = self.relu.forward(x);
        let x = self.max_pool2.forward(x); // [batch_size, 32, height/4, width/4]

        // 第三层卷积块
        let x = self.conv3.forward(x); // [batch_size, 64, height/4, width/4]
        let x = self.relu.forward(x);
        let x = self.max_pool3.forward(x); // [batch_size, 64, height/8, width/8]

        // 第四层卷积块
        let x = self.conv4.forward(x); // [batch_size, 128, height/8, width/8]
        let x = self.relu.forward(x);
        let x = self.max_pool4.forward(x); // [batch_size, 128, height/16, width/16]

        // 1x1 卷积层
        let x = self.conv5.forward(x); // [batch_size, 1, height/16, width/16]

        // 全局平均池化
        let x = self.avg_pool.forward(x); // [batch_size, 1, 1, 1]

        // 展平并应用 dropout
        let x = x.reshape([batch_size, 1]);
        let x = self.dropout.forward(x);

        // 应用 sigmoid 激活函数进行二分类
        burn::tensor::activation::sigmoid(x)
    }
}
