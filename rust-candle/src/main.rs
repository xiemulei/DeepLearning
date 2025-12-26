use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, Optimizer, SGD, VarBuilder, VarMap, linear, loss, ops};

mod utils;
use rand::seq::SliceRandom;
use utils::net::print_varmap;

pub struct SimpleModel {
    linear1: Linear,
    linear2: Linear,
    linear3: Linear,
}

impl SimpleModel {
    pub fn new(vb: VarBuilder, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Result<Self> {
        let linear1 = linear(in_dim, hidden_dim, vb.pp("linear1"))?;
        let linear2 = linear(hidden_dim, hidden_dim, vb.pp("linear2"))?;
        let linear3 = linear(hidden_dim, out_dim, vb.pp("linear3"))?;

        Ok(Self {
            linear1,
            linear2,
            linear3,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.silu()?;
        let x = self.linear2.forward(&x)?;
        let x = x.silu()?;
        let x = self.linear3.forward(&x)?;

        Ok(x)
    }
}

pub trait Dataset {
    fn len(&self) -> Result<usize>;
    fn get_batch(&self, start: usize, end: usize) -> Result<(Tensor, Tensor)>;
    fn shuffle(&mut self) -> Result<()>;
}

pub struct DemoDataset {
    inputs: Tensor,
    targets: Tensor,
}

impl DemoDataset {
    pub fn new(x: Tensor, y: Tensor) -> Result<Self> {
        Ok(Self {
            inputs: x,
            targets: y,
        })
    }

    pub fn get_idx(&self, idx: usize) -> Result<(Tensor, Tensor)> {
        let x_idx = self.inputs.narrow(0, idx, 1)?;
        let y_idx = self.targets.narrow(0, idx, 1)?;

        Ok((x_idx, y_idx))
    }
}

impl Dataset for DemoDataset {
    fn len(&self) -> Result<usize> {
        Ok(self.inputs.dim(0)?)
    }

    fn get_batch(&self, start: usize, end: usize) -> Result<(Tensor, Tensor)> {
        let x_idx = self.inputs.narrow(0, start, end - start)?;
        let y_idx = self.targets.narrow(0, start, end - start)?;

        Ok((x_idx, y_idx))
    }

    fn shuffle(&mut self) -> Result<()> {
        let len = self.len()?;
        let mut indices: Vec<u32> = (0..len).map(|i| i as u32).collect();
        let mut rng = rand::rng();
        indices.shuffle(&mut rng);
        let indices_tensot = Tensor::from_vec(indices, (len,), self.inputs.device())?;
        self.inputs = self.inputs.index_select(&indices_tensot, 0)?;
        self.targets = self.targets.index_select(&indices_tensot, 0)?;

        Ok(())
    }
}

pub struct DataLoader<'a> {
    dataset: Box<dyn Dataset + 'a>,
    batch_size: usize,
    current_index: usize,
    shuffle: bool,
}

impl<'a> DataLoader<'a> {
    pub fn new<D: Dataset + 'a>(dataset: D, batch_size: usize, shuffle: bool) -> Result<Self> {
        Ok(Self {
            dataset: Box::new(dataset),
            batch_size,
            current_index: 0,
            shuffle,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.current_index = 0;
        if self.shuffle {
            let _ = self.dataset.shuffle()?;
        }

        Ok(())
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.current_index * self.batch_size;
        let end = std::cmp::min(start + self.batch_size, self.dataset.len().ok()?);

        if start >= end {
            return None;
        }

        let batch = self.dataset.get_batch(start, end).ok()?;
        self.current_index += 1;
        Some(Ok(batch))
    }
}

fn main() -> Result<()> {
    let device = Device::new_metal(0).expect("Metal Not Found");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = SimpleModel::new(vb, 2, 20, 2)?;
    let _ = print_varmap(&varmap);

    let x_train = Tensor::from_vec(
        vec![-1.2f32, 3.1, -0.9, 2.9, -0.5, 2.6, 2.3, -1.1, 2.7, -1.5],
        (5, 2),
        &device,
    )?;
    let y_train = Tensor::from_vec(vec![0u32, 0, 0, 1, 1], 5, &device)?;
    let x_val = Tensor::from_vec(vec![-0.8f32, 2.8, 2.6, -1.6], (2, 2), &device)?;
    let y_val = Tensor::from_vec(vec![0u32, 1], (2,), &device)?;

    let train_dataset = DemoDataset::new(x_train, y_train)?;
    let val_dataset = DemoDataset::new(x_val, y_val)?;
    let mut train_loader = DataLoader::new(train_dataset, 2, true)?;
    let mut val_loader = DataLoader::new(val_dataset, 2, false)?;
    let mut sgd = SGD::new(varmap.all_vars(), 0.01)?;
    let epochs = 3;

    for epoch in 0..epochs {
        let _ = train_loader.reset();
        let _ = val_loader.reset();

        for batch in &mut train_loader {
            let (x_, y_) = batch?;
            let predict = model.forward(&x_)?;
            let loss_ = loss::cross_entropy(&predict, &y_)?;
            sgd.backward_step(&loss_)?;
            println!("Epoch: {epoch} train loss: {loss_}");
        }

        for batch in &mut val_loader {
            let (x_, y_) = batch?;
            let predict = model.forward(&x_)?;
            let loss_ = loss::cross_entropy(&predict, &y_)?;
            println!("Epoch: {epoch} val loss: {loss_}");
        }
    }

    let _ = train_loader.reset();
    let _ = val_loader.reset();

    for batch in &mut train_loader {
        let (x_, y_) = batch?;
        let predict = model.forward(&x_)?;
        let softmax = ops::softmax(&predict, 1)?;
        let label = softmax.argmax(1)?;
        println!("train label: {}", label);
        println!("true label: {}", y_);
        println!(
            "train acc: {}",
            label
                .eq(&y_)?
                .sum(0)?
                .to_dtype(DType::F32)?
                .affine(1.0 / (x_.dim(0)? as f64), 0.0)?
        )
    }

    for batch in &mut val_loader {
        let (x_, y_) = batch?;
        let predict = model.forward(&x_)?;
        let softmax = ops::softmax(&predict, 1)?;
        let label = softmax.argmax(1)?;
        println!("val label: {}", label);
        println!("true label: {}", y_);
        println!(
            "val acc: {}",
            label
                .eq(&y_)?
                .sum(0)?
                .to_dtype(DType::F32)?
                .affine(1.0 / (x_.dim(0)? as f64), 0.0)?
        )
    }

    varmap.save("model.safetensors")?;

    Ok(())
}
