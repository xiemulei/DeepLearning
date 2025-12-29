use core::f32;

use candle_core::{D, Device, Error, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap, embedding, linear_no_bias, ops};
use rand::seq::{self, SliceRandom};
use tokenizers::Tokenizer;

use crate::utils::{DataLoader, Dataset, file::read_text};

mod chapter2;
mod chapter3;
mod utils;

pub struct TokenDataset {
    input_ids: Tensor,
    target_ids: Tensor,
}

impl TokenDataset {
    pub fn new(
        text: String,
        tokenizer: &Tokenizer,
        seq_len: usize,
        stride: usize,
        device: &Device,
    ) -> Result<Self> {
        let encode = tokenizer
            .encode(text, true)
            .map_err(|e| Error::Msg(format!("tokenizer encode error: {}", e)))?;
        let token_ids = encode.get_ids();
        let token_len = token_ids.len();
        let max_token_id = token_len - seq_len;

        let mut input_ids_vec = Vec::new();
        let mut target_ids_vec = Vec::new();
        for i in (0..max_token_id).step_by(stride) {
            input_ids_vec.extend_from_slice(&token_ids[i..i + seq_len]);
            target_ids_vec.extend_from_slice(&token_ids[i + 1..i + seq_len + 1]);
        }

        let bs = input_ids_vec.len() / seq_len;
        let input_ids = Tensor::from_vec(input_ids_vec, (bs, seq_len), device)?;
        let target_ids = Tensor::from_vec(target_ids_vec, (bs, seq_len), device)?;

        Ok(Self {
            input_ids,
            target_ids,
        })
    }
}

impl Dataset for TokenDataset {
    fn len(&self) -> Result<usize> {
        Ok(self.input_ids.dim(0)?)
    }

    fn get_batch(&self, start: usize, end: usize) -> Result<(Tensor, Tensor)> {
        let x_idx = self.input_ids.narrow(0, start, end - start)?;
        let y_idx = self.target_ids.narrow(0, start, end - start)?;

        Ok((x_idx, y_idx))
    }

    fn shuffle(&mut self) -> Result<()> {
        let len = self.len()?;
        let mut indices: Vec<u32> = (0..len).map(|i| i as u32).collect();
        let mut rng = rand::rng();
        indices.shuffle(&mut rng);
        let indices_tensor = Tensor::from_vec(indices, (len,), self.input_ids.device())?;
        self.input_ids = self.input_ids.index_select(&indices_tensor, 0)?;
        self.target_ids = self.target_ids.index_select(&indices_tensor, 0)?;

        Ok(())
    }
}

pub struct SinusoidalPositionEmbedding {
    pub pos_embedding: Tensor,
}

impl SinusoidalPositionEmbedding {
    pub fn new(seq_len: usize, hidden_dim: usize, device: &Device) -> Result<Self> {
        assert_eq!(hidden_dim % 2, 0, "hidden_dim must be even");

        let mut pos_embedding_vec = Vec::new();
        for pos in 0..seq_len {
            for i in (0..hidden_dim).step_by(2) {
                let pos_i = pos as f32 / 10000.0_f32.powf(i as f32 / hidden_dim as f32);
                let sin = pos_i.sin();
                let cos = pos_i.cos();
                pos_embedding_vec.push(sin);
                pos_embedding_vec.push(cos);
            }
        }

        let pos_embedding = Tensor::from_vec(pos_embedding_vec, (seq_len, hidden_dim), device)?;

        Ok(Self { pos_embedding })
    }
}

pub struct RoPE {
    sin: Tensor,
    cos: Tensor,
}

impl RoPE {
    pub fn new(seq_len: usize, embedding_dim: usize, device: &Device) -> Result<Self> {
        assert_eq!(embedding_dim % 2, 0, "hidden_dim must be even");

        let mut angle = Vec::new();
        for pos in 0..seq_len {
            for i in (0..embedding_dim).step_by(2) {
                let pos_i = pos as f32 / 10000.0_f32.powf(i as f32 / embedding_dim as f32);
                angle.extend_from_slice(&[pos_i, pos_i]);
            }
        }

        let angle_tensor = Tensor::from_vec(angle, (seq_len, embedding_dim), device)?;
        let cos = angle_tensor.cos()?;
        let sin = angle_tensor.sin()?;

        Ok(Self { sin, cos })
    }

    pub fn new0_half(seq_len: usize, embedding_dim: usize, device: &Device) -> Result<Self> {
        assert_eq!(embedding_dim % 2, 0, "hidden_dim must be even");

        // let mut angle = Vec::new();
        // for pos in 0..seq_len {
        //     for i in (0..embedding_dim).step_by(2) {
        //         let pos_i = pos as f32 / 10000.0_f32.powf(i as f32 / embedding_dim as f32);
        //         angle.push(pos_i);
        //     }
        // }
        // let angle_tensor = Tensor::from_vec(angle, (seq_len, embedding_dim / 2), device)?;

        let pos_vec = (0..seq_len).map(|i| i as f32).collect();
        let angle_base = (0..embedding_dim)
            .step_by(2)
            .map(|i| 1.0_f32 / 10000.0_f32.powf(i as f32 / embedding_dim as f32))
            .collect();
        let pos = Tensor::from_vec(pos_vec, (seq_len, 1), device)?;
        let angle_base = Tensor::from_vec(angle_base, (1, embedding_dim / 2), device)?;
        let angle_tensor = pos.matmul(&angle_base)?;
        let angle_tensor = Tensor::cat(&[&angle_tensor, &angle_tensor], 1)?;
        let sin = angle_tensor.sin()?;
        let cos = angle_tensor.cos()?;
        Ok(Self { sin, cos })
    }

    pub fn forward0_half(&self, x: &Tensor) -> Result<Tensor> {
        println!("x: {x}");
        let x_cos = x.broadcast_mul(&self.cos)?;
        let dims = x.dims();
        let half_dim = dims[dims.len() - 1] / 2;
        let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
        let x2 = x2.affine(-1.0, 0.0)?;
        let rotate_x = Tensor::cat(&[&x2, &x1], D::Minus1)?;
        println!("rotate_x: {rotate_x}");
        let x_sin = rotate_x.broadcast_mul(&self.sin)?;
        let rotate = x_cos.add(&x_sin)?;
        Ok(rotate)
    }

    /// RoPE (Rotary Position Embedding) 前向传播
    ///
    /// RoPE 核心公式: RoPE(x, m) = x * cos(mθ) + rotate(x) * sin(mθ)
    /// 其中 rotate(x) 是将向量 x 的两两元素交换并取负号
    ///
    /// 参数:
    ///   x: 输入张量，形状为 [batch_size, seq_len, embedding_dim]
    ///
    /// 返回:
    ///   应用 RoPE 旋转后的张量
    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        // 步骤1: 计算 x * cos(mθ) 部分
        // 使用广播机制将 cos 值与输入张量相乘
        // x 形状: [batch_size, seq_len, embedding_dim]
        // self.cos 形状: [seq_len, embedding_dim]
        // x_cos 形状: [batch_size, seq_len, embedding_dim]
        let x_cos = x.broadcast_mul(&self.cos)?;

        // 步骤2: 准备旋转操作
        // 获取输入张量的维度信息
        let dims = x.dims();

        // 构造新的维度用于 reshape 操作
        // 将最后一维 embedding_dim 拆分为 [embedding_dim/2, 2]
        // 例如: [bs, seq_len, 32] -> [bs, seq_len, 16, 2]
        let mut new_dim = dims.to_vec();
        new_dim[dims.len() - 1] = dims[dims.len() - 1] / 2;
        new_dim.push(2);

        // reshape 张量，将最后一维拆分成两维
        // x_reshape 形状: [batch_size, seq_len, embedding_dim/2, 2]
        // 这样每一对的两个相邻元素就被分到同一组
        let x_reshape = x.reshape(new_dim)?;

        // 步骤3: 分离向量对并构建旋转操作
        // 提取每一对的第一个元素（奇数位置）
        // x1 形状: [batch_size, seq_len, embedding_dim/2, 1]
        let x1 = x_reshape.narrow(D::Minus1, 0, 1)?;

        // 提取每一对的第二个元素（偶数位置）
        // x2 形状: [batch_size, seq_len, embedding_dim/2, 1]
        let x2 = x_reshape.narrow(D::Minus1, 1, 1)?;

        // 对第二个元素取负号 (仿射变换: -1.0 * x2 + 0.0)
        // 这是 RoPE 旋转公式中 rotate(x) 操作的关键部分
        // rotate(x) = [-x2, x1, -x4, x3, ...]
        let x2 = x2.affine(-1.0, 0.0)?;

        // 步骤4: 构建旋转后的向量
        // 将取负后的 x2 和原始的 x1 按最后一维堆叠
        // rotate_stack_x 形状: [batch_size, seq_len, embedding_dim/2, 1, 2]
        // 实现了 [-x2, x1] 的排列
        let rotate_stack_x = Tensor::stack(&[&x2, &x1], D::Minus1)?;

        // 将最后两维展平，恢复原始形状
        // 从维度 -3 (即 embedding_dim/2) 到最后一维进行展平
        // rotate_flatten 形状: [batch_size, seq_len, embedding_dim]
        // 此时实现了完整的旋转: [x1, x2] -> [-x2, x1]
        let rotate_flatten = rotate_stack_x.flatten(D::Minus(3), D::Minus1)?;

        // 步骤5: 计算 rotate(x) * sin(mθ) 部分
        // 将旋转后的向量与 sin 值相乘
        // x_sin 形状: [batch_size, seq_len, embedding_dim]
        let x_sin = rotate_flatten.broadcast_mul(&self.sin)?;

        // 步骤6: 合并两部分结果
        // 最终公式: x * cos(mθ) + rotate(x) * sin(mθ)
        // rotate 形状: [batch_size, seq_len, embedding_dim]
        let rotate = x_cos.add(&x_sin)?;

        Ok(rotate)
    }
}

pub struct DotProductAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    d_sqrt: Tensor,
}

pub fn mask_filled(on_true: &Tensor, mask: &Tensor, on_false: f32) -> Result<Tensor> {
    let mask = mask.broadcast_as(on_true.shape())?;
    let on_false = Tensor::new(on_false, on_true.device())?.broadcast_as(on_true.shape())?;
    println!("{:?}", on_false);
    let filled = mask.where_cond(on_true, &on_false)?;
    Ok(filled)
}

impl DotProductAttention {
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        let w_q = linear_no_bias(in_dim, out_dim, vb.pp("w_q"))?;
        let w_k = linear_no_bias(in_dim, out_dim, vb.pp("w_k"))?;
        let w_v = linear_no_bias(in_dim, out_dim, vb.pp("w_v"))?;
        let d_sqrt = 1.0 / (out_dim as f32).sqrt();
        let d_sqrt = Tensor::new(d_sqrt, device)?;

        Ok(Self {
            w_q,
            w_k,
            w_v,
            d_sqrt,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;
        let atten_score = q.matmul(&k.t()?)?;
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?;
        Ok(atten_weight)
    }

    pub fn forward_with_mask(&self, x: &Tensor, mask: bool) -> Result<Tensor> {
        let (_, seq_len, _) = x.dims3()?;
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        // 计算 attention score: q 和 k 的转置相乘
        let mut atten_score = q.matmul(&k.t()?)?; // [batch_size, seq_len, seq_len]

        if mask {
            let mask = Tensor::tril2(seq_len, candle_core::DType::U32, x.device())?;
            println!("mask: {:?}", mask);
            atten_score = mask_filled(&atten_score, &mask, f32::NEG_INFINITY)?;
        }

        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?;
        Ok(atten_weight)
    }
}

pub struct MultiHeadAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    out_proj: Linear,
    n_head: usize,
    head_dim: usize,
    out_dim: usize,
    d_sqrt: Tensor,
}

impl MultiHeadAttention {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        n_head: usize,
        device: &Device,
    ) -> Result<Self> {
        let w_q = linear_no_bias(in_dim, out_dim, vb.pp("w_q"))?;
        let w_k = linear_no_bias(in_dim, out_dim, vb.pp("w_k"))?;
        let w_v = linear_no_bias(in_dim, out_dim, vb.pp("w_v"))?;
        let out_proj = linear_no_bias(out_dim, out_dim, vb.pp("out_proj"))?;
        let head_dim = out_dim / n_head;
        let d_sqrt = 1.0 / (head_dim as f32).sqrt();
        let d_sqrt = Tensor::new(d_sqrt, device)?;

        Ok(Self {
            w_q,
            w_k,
            w_v,
            out_proj,
            n_head,
            head_dim,
            out_dim,
            d_sqrt,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: bool) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?;
        // (bs, n_head, seq_len, head_dim)
        let q = self
            .w_q
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .w_k
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .w_v
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut atten_score = q.matmul(&k.t()?)?;
        if mask {
            let mask = Tensor::tril2(seq_len, candle_core::DType::U32, x.device())?;
            atten_score = mask_filled(&atten_score, &mask, f32::NEG_INFINITY)?;
            println!("{}", atten_score);
        }
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?; // (bs, n_head, seq_len, head_dim)
        let atten_weight = atten_weight
            .transpose(1, 2)?
            .reshape((bs, seq_len, self.out_dim))?;
        let out = self.out_proj.forward(&atten_weight)?;
        Ok(out)
    }
}

fn main() -> Result<()> {
    let device = Device::metal_if_available(0)?;

    let text = read_text("src/assets/sub_wiki_0_99.txt");
    let tokenizer = Tokenizer::from_file("src/assets/tokenizer.json")
        .map_err(|e| Error::Msg(format!("tokenizer from file error {}", e)))?;
    let vocab_size = tokenizer.get_vocab_size(true);

    let seq_len = 32;
    let stride = 32;
    let batch_size = 2;
    let token_dataset = TokenDataset::new(text, &tokenizer, seq_len, stride, &device)?;
    let mut dataloader = DataLoader::new(token_dataset, batch_size, true)?;
    let _ = dataloader.reset()?;
    let (x, _y) = dataloader.next().unwrap()?;

    let embedding_dim = 32;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let embedding = embedding(vocab_size, embedding_dim, vb.pp("embedding"))?;
    let encode = embedding.forward(&x)?;
    let pos_embedding =
        SinusoidalPositionEmbedding::new(seq_len, embedding_dim, &device)?.pos_embedding;
    let encode = encode.broadcast_add(&pos_embedding)?;
    // let attention1 =
        // DotProductAttention::new(vb.pp("attention1"), embedding_dim, embedding_dim, &device)?;
    // let output = attention1.forward(&encode)?;
    // let output = attention1.forward_with_mask(&encode, true)?;
    let n_head = 4;
    let mha = MultiHeadAttention::new(vb.pp("mha1"), embedding_dim, embedding_dim, n_head, &device)?;
    let output = mha.forward(&encode, true)?;
    println!("{:?}", output);

    Ok(())
}
