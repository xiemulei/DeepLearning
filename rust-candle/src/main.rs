use core::f32;
use std::collections::HashMap;

use candle_core::{D, DType, Device, Error, IndexOp, Result, Tensor};
use candle_nn::{
    Embedding, Init, Linear, Module, VarBuilder, VarMap, embedding, linear_no_bias, ops,
};
use rand::seq::SliceRandom;
use tokenizers::Tokenizer;

use crate::utils::{DataLoader, Dataset, file::read_text};

mod chapter2;
mod chapter3;
mod utils;

// Token 数据集结构，用于将文本转换为模型可处理的 token ID 序列
pub struct TokenDataset {
    input_ids: Tensor,  // 输入 token ID 张量
    target_ids: Tensor, // 目标 token ID 张量（用于训练）
}

impl TokenDataset {
    // 创建新的 TokenDataset 实例
    // 参数:
    //   text: 输入文本
    //   tokenizer: 分词器
    //   seq_len: 序列长度
    //   stride: 滑动窗口步长
    //   device: 计算设备
    pub fn new(
        text: String,
        tokenizer: &Tokenizer,
        seq_len: usize,
        stride: usize,
        device: &Device,
    ) -> Result<Self> {
        // 使用分词器将文本转换为 token ID
        let encode = tokenizer
            .encode(text, true) // 添加特殊标记
            .map_err(|e| Error::Msg(format!("tokenizer encode error: {}", e)))?;
        let token_ids = encode.get_ids(); // 获取 token ID 列表
        let token_len = token_ids.len(); // 获取 token 总长度
        let max_token_id = token_len - seq_len; // 计算最大起始位置

        let mut input_ids_vec = Vec::new(); // 存储输入 ID 的向量
        let mut target_ids_vec = Vec::new(); // 存储目标 ID 的向量

        // 滑动窗口生成输入-目标对
        for i in (0..max_token_id).step_by(stride) {
            // 输入: [i, i+seq_len) 的 token
            input_ids_vec.extend_from_slice(&token_ids[i..i + seq_len]);
            // 目标: [i+1, i+seq_len+1) 的 token (即输入的下一个 token)
            target_ids_vec.extend_from_slice(&token_ids[i + 1..i + seq_len + 1]);
        }

        // 计算批次大小
        let bs = input_ids_vec.len() / seq_len;
        // 创建输入和目标张量
        let input_ids = Tensor::from_vec(input_ids_vec, (bs, seq_len), device)?;
        let target_ids = Tensor::from_vec(target_ids_vec, (bs, seq_len), device)?;

        Ok(Self {
            input_ids,
            target_ids,
        })
    }
}

impl Dataset for TokenDataset {
    // 获取数据集长度（批次数量）
    fn len(&self) -> Result<usize> {
        Ok(self.input_ids.dim(0)?) // 获取第一个维度的大小（批次维度）
    }

    // 获取指定范围的批次数据
    // 参数:
    //   start: 起始索引
    //   end: 结束索引
    fn get_batch(&self, start: usize, end: usize) -> Result<(Tensor, Tensor)> {
        // 获取输入张量的指定范围
        let x_idx = self.input_ids.narrow(0, start, end - start)?;
        // 获取目标张量的指定范围
        let y_idx = self.target_ids.narrow(0, start, end - start)?;

        Ok((x_idx, y_idx))
    }

    // 随机打乱数据集
    fn shuffle(&mut self) -> Result<()> {
        let len = self.len()?; // 获取数据集长度
        // 创建索引向量 [0, 1, 2, ..., len-1]
        let mut indices: Vec<u32> = (0..len).map(|i| i as u32).collect();
        let mut rng = rand::rng(); // 获取随机数生成器
        indices.shuffle(&mut rng); // 随机打乱索引
        // 创建索引张量
        let indices_tensor = Tensor::from_vec(indices, (len,), self.input_ids.device())?;
        // 根据随机索引重新排列输入和目标张量
        self.input_ids = self.input_ids.index_select(&indices_tensor, 0)?;
        self.target_ids = self.target_ids.index_select(&indices_tensor, 0)?;

        Ok(())
    }
}

// 正弦位置嵌入结构，用于为序列中的位置提供信息
pub struct SinusoidalPositionEmbedding {
    pub pos_embedding: Tensor, // 位置嵌入张量
}

impl SinusoidalPositionEmbedding {
    // 创建新的正弦位置嵌入
    // 参数:
    //   seq_len: 序列长度
    //   hidden_dim: 隐藏层维度（必须为偶数）
    //   device: 计算设备
    pub fn new(seq_len: usize, hidden_dim: usize, device: &Device) -> Result<Self> {
        assert_eq!(hidden_dim % 2, 0, "hidden_dim must be even"); // 隐藏维度必须为偶数

        let mut pos_embedding_vec = Vec::new();
        // 为每个位置和每个维度计算正弦/余弦值
        for pos in 0..seq_len {
            // 遍历序列中的每个位置
            for i in (0..hidden_dim).step_by(2) {
                // 遍历维度，步长为2（因为成对处理）
                // 计算位置编码公式: pos / (10000^(i/d_model))
                let pos_i = pos as f32 / 10000.0_f32.powf(i as f32 / hidden_dim as f32);
                let sin = pos_i.sin(); // 计算正弦值
                let cos = pos_i.cos(); // 计算余弦值
                pos_embedding_vec.push(sin); // 添加正弦值
                pos_embedding_vec.push(cos); // 添加余弦值
            }
        }

        // 创建位置嵌入张量
        let pos_embedding = Tensor::from_vec(pos_embedding_vec, (seq_len, hidden_dim), device)?;

        Ok(Self { pos_embedding })
    }
}

// 应用正弦余弦变换到输入张量
// 这是 RoPE (Rotary Position Embedding) 的核心操作
// 实现公式: x * cos(θ) + rotate(x) * sin(θ)
pub fn apply_sin_cos(x: &Tensor, sin: &Tensor, cos: &Tensor) -> Result<Tensor> {
    let (_, _, _, head_dim) = x.dims4()?; // 获取最后一个维度（头维度）
    let half_dim = head_dim / 2; // 计算一半维度
    // 分割张量为两部分
    let x1 = x.narrow(D::Minus1, 0, half_dim)?; // 前半部分
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?; // 后半部分
    let x2 = x2.affine(-1.0, 0.0)?; // 对后半部分取负
    // 重新排列: 将 -x2 和 x1 连接起来实现旋转
    let rotate_x = Tensor::cat(&[&x2, &x1], D::Minus1)?;
    // 应用 cos 部分
    let x_cos = x.broadcast_mul(&cos)?;
    // 应用 sin 部分
    let x_sin = rotate_x.broadcast_mul(&sin)?;
    // 合并结果
    let rotate = x_cos.add(&x_sin)?;
    Ok(rotate)
}

// RoPE (Rotary Position Embedding) 结构体
// 用于实现旋转位置编码，为模型提供位置信息
pub struct RoPE {
    sin: Tensor, // 正弦值张量
    cos: Tensor, // 余弦值张量
}

impl RoPE {
    // 创建新的 RoPE 实例
    // 参数:
    //   seq_len: 序列长度
    //   embedding_dim: 嵌入维度（必须为偶数）
    //   device: 计算设备
    pub fn new(seq_len: usize, embedding_dim: usize, device: &Device) -> Result<Self> {
        assert_eq!(embedding_dim % 2, 0, "hidden_dim must be even"); // 嵌入维度必须为偶数

        // 创建位置向量 [0, 1, 2, ..., seq_len-1]
        let pos_vec = (0..seq_len).map(|i| i as f32).collect();
        // 创建角度基础向量
        let angle_base = (0..embedding_dim)
            .step_by(2) // 每两个维度计算一次
            .map(|i| 1.0_f32 / 10000.0_f32.powf(i as f32 / embedding_dim as f32))
            .collect();
        // 创建位置张量
        let pos = Tensor::from_vec(pos_vec, (seq_len, 1), device)?;
        // 创建角度基础张量
        let angle_base = Tensor::from_vec(angle_base, (1, embedding_dim / 2), device)?;
        // 计算角度张量
        let angle_tensor = pos.matmul(&angle_base)?;
        // 将角度张量复制一份，用于正弦和余弦计算
        let angle_tensor = Tensor::cat(&[&angle_tensor, &angle_tensor], 1)?;
        // 计算正弦和余弦值
        let sin = angle_tensor.sin()?;
        let cos = angle_tensor.cos()?;
        Ok(Self { sin, cos })
    }

    // 对输入张量应用 RoPE 变换（完整序列）
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rotate = apply_sin_cos(x, &self.sin, &self.cos)?;
        Ok(rotate)
    }

    // 对输入张量应用 RoPE 变换（指定位置范围）
    // 用于推理时的增量处理
    // 参数:
    //   x: 输入张量
    //   pos_idx: 起始位置索引
    pub fn apply(&self, x: &Tensor, pos_idx: usize) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?; // 获取序列长度
        let (rope_seq_len, _) = self.cos.dims2()?; // 获取 RoPE 序列长度
        // 确保 RoPE 的长度足够覆盖所需位置
        assert!(
            rope_seq_len >= (pos_idx + seq_len),
            "rope_seq_len less than pos_idx + seq_len"
        );

        // 提取指定位置范围的 cos 和 sin 值
        let cos = self.cos.narrow(0, pos_idx, seq_len)?;
        let sin = self.sin.narrow(0, pos_idx, seq_len)?;
        let rotate = apply_sin_cos(x, &sin, &cos)?;
        Ok(rotate)
    }

    // 创建新的 RoPE 实例（替代实现）
    pub fn new_reformer(seq_len: usize, embedding_dim: usize, device: &Device) -> Result<Self> {
        assert_eq!(embedding_dim % 2, 0, "hidden_dim must be even"); // 嵌入维度必须为偶数

        let mut angle = Vec::new();
        // 计算角度值
        for pos in 0..seq_len {
            for i in (0..embedding_dim).step_by(2) {
                let pos_i = pos as f32 / 10000.0_f32.powf(i as f32 / embedding_dim as f32);
                // 每个位置的每个维度都添加两次（用于正弦和余弦）
                angle.extend_from_slice(&[pos_i, pos_i]);
            }
        }

        let angle_tensor = Tensor::from_vec(angle, (seq_len, embedding_dim), device)?;
        let cos = angle_tensor.cos()?; // 计算余弦值
        let sin = angle_tensor.sin()?; // 计算正弦值

        Ok(Self { sin, cos })
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
    pub fn forward_reformer(&self, x: &Tensor) -> Result<Tensor> {
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

// 点积注意力结构体
// 实现基本的缩放点积注意力机制
pub struct DotProductAttention {
    w_q: Linear,    // 查询线性变换层
    w_k: Linear,    // 键线性变换层
    w_v: Linear,    // 值线性变换层
    d_sqrt: Tensor, // 缩放因子 (1/sqrt(d_k))，用于缩放注意力分数
}

// 使用掩码填充张量
// 将掩码位置的值替换为指定值（通常用于掩码未来信息）
pub fn mask_filled(on_true: &Tensor, mask: &Tensor, on_false: f32) -> Result<Tensor> {
    let (mask_seq_len, _) = mask.dims2()?; // 获取掩码序列长度
    let (_, _, seq_len, _) = on_true.dims4()?; // 获取输入张量序列长度
    // 确保掩码长度足够
    assert!(
        mask_seq_len >= seq_len,
        "mask seq_len less than input data seq_len"
    );
    // 提取掩码的前 seq_len x seq_len 部分
    let mask = mask.i((..seq_len, ..seq_len))?;
    // 将掩码广播到与输入张量相同的形状
    let mask = mask.broadcast_as(on_true.shape())?;
    // 创建填充值张量（广播到相同形状）
    let on_false = Tensor::new(on_false, on_true.device())?.broadcast_as(on_true.shape())?;
    // 使用条件选择：如果掩码为真则保留原值，否则使用填充值
    let filled = mask.where_cond(on_true, &on_false)?;
    Ok(filled)
}

impl DotProductAttention {
    // 创建新的点积注意力实例
    // 参数:
    //   vb: 变量构建器
    //   in_dim: 输入维度
    //   out_dim: 输出维度
    //   device: 计算设备
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        // 创建查询、键、值的线性变换层
        let w_q = linear_no_bias(in_dim, out_dim, vb.pp("w_q"))?;
        let w_k = linear_no_bias(in_dim, out_dim, vb.pp("w_k"))?;
        let w_v = linear_no_bias(in_dim, out_dim, vb.pp("w_v"))?;
        // 计算缩放因子 (1/sqrt(d_k))
        let d_sqrt = 1.0 / (out_dim as f32).sqrt();
        let d_sqrt = Tensor::new(d_sqrt, device)?;

        Ok(Self {
            w_q,
            w_k,
            w_v,
            d_sqrt,
        })
    }

    // 前向传播（无掩码）
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 线性变换得到 Q, K, V
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;
        // 计算注意力分数: Q * K^T
        let atten_score = q.matmul(&k.t()?)?;
        // 应用缩放因子
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        // 应用 Softmax 激活函数
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        // 与 V 相乘得到最终输出
        let atten_weight = softmax.matmul(&v)?;
        Ok(atten_weight)
    }

    // 前向传播（带掩码）
    // 参数:
    //   x: 输入张量
    //   mask: 是否应用掩码（通常用于因果注意力）
    pub fn forward_with_mask(&self, x: &Tensor, mask: bool) -> Result<Tensor> {
        let (_, seq_len, _) = x.dims3()?; // 获取序列长度
        // 线性变换得到 Q, K, V
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        // 计算注意力分数: q 和 k 的转置相乘
        let mut atten_score = q.matmul(&k.t()?)?; // [batch_size, seq_len, seq_len]

        if mask {
            // 创建下三角掩码（防止看到未来信息）
            let mask = Tensor::tril2(seq_len, candle_core::DType::U32, x.device())?;
            println!("mask: {:?}", mask);
            // 应用掩码，将未来位置设为负无穷
            atten_score = mask_filled(&atten_score, &mask, f32::NEG_INFINITY)?;
        }

        // 应用缩放因子
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        // 应用 Softmax
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        // 与 V 相乘
        let atten_weight = softmax.matmul(&v)?;
        Ok(atten_weight)
    }
}

// 共享缓冲区结构，用于缓存掩码和 RoPE 计算结果
// 避免重复计算，提高效率
pub struct SharedBuffer {
    buffers: HashMap<String, (Tensor, RoPE)>, // 缓存映射，键为"序列长度_维度"格式
}

impl SharedBuffer {
    // 创建新的共享缓冲区
    pub fn new() -> Result<Self> {
        let buffers: HashMap<String, (Tensor, RoPE)> = HashMap::new();
        Ok(Self { buffers })
    }

    // 获取指定序列长度和维度的掩码和 RoPE
    // 如果不存在则创建并缓存
    pub fn get(&mut self, seq_len: usize, dim: usize, device: &Device) -> Result<&(Tensor, RoPE)> {
        let key = format!("{seq_len}_{dim}"); // 创建缓存键
        if !self.buffers.contains_key(&key) {
            // 如果缓存中不存在，则创建新的掩码和 RoPE
            let mask = Tensor::tril2(seq_len, DType::U32, device)?; // 创建下三角掩码
            let rope = RoPE::new(seq_len, dim, device)?; // 创建 RoPE
            self.buffers.insert(key.clone(), (mask, rope)); // 存入缓存
        }

        // 获取缓存的值
        let value = self
            .buffers
            .get(&key)
            .ok_or(Error::Msg(format!("get mask rope key: {} None", key)))?;

        Ok(value)
    }
}

// 多头注意力结构体
// 实现标准的多头自注意力机制
pub struct MultiHeadAttention {
    w_q: Linear,      // 查询线性变换层
    w_k: Linear,      // 键线性变换层
    w_v: Linear,      // 值线性变换层
    out_proj: Linear, // 输出投影层
    n_head: usize,    // 注意力头数
    head_dim: usize,  // 每个头的维度
    out_dim: usize,   // 输出维度
    d_sqrt: Tensor,   // 缩放因子 (1/sqrt(head_dim))
}

impl MultiHeadAttention {
    // 创建新的多头注意力实例
    // 参数:
    //   vb: 变量构建器
    //   in_dim: 输入维度
    //   out_dim: 输出维度
    //   n_head: 注意力头数
    //   device: 计算设备
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        n_head: usize,
        device: &Device,
    ) -> Result<Self> {
        // 验证输出维度必须能被头数整除
        assert_eq!(out_dim % n_head, 0, "out_dim must be divisible by n_head");

        // 创建查询、键、值的线性变换层
        let w_q = linear_no_bias(in_dim, out_dim, vb.pp("w_q"))?;
        let w_k = linear_no_bias(in_dim, out_dim, vb.pp("w_k"))?;
        let w_v = linear_no_bias(in_dim, out_dim, vb.pp("w_v"))?;
        // 创建输出投影层
        let out_proj = linear_no_bias(out_dim, out_dim, vb.pp("out_proj"))?;
        // 计算每个头的维度
        let head_dim = out_dim / n_head;
        // 计算缩放因子
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

    // 前向传播（带掩码选项）
    // 参数:
    //   x: 输入张量
    //   mask: 是否应用因果掩码
    pub fn forward(&self, x: &Tensor, mask: bool) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?; // 获取批次大小和序列长度

        // 线性变换并重塑为多头格式 (bs, seq_len, n_head, head_dim)
        let q = self
            .w_q
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))? // 重塑为多头格式
            .transpose(1, 2)? // 转置为 (bs, n_head, seq_len, head_dim)
            .contiguous()?; // 确保内存连续
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

        // 计算注意力分数
        let mut atten_score = q.matmul(&k.t()?)?; // (bs, n_head, seq_len, seq_len)

        if mask {
            // 应用因果掩码（防止看到未来信息）
            let mask = Tensor::tril2(seq_len, candle_core::DType::U32, x.device())?;
            atten_score = mask_filled(&atten_score, &mask, f32::NEG_INFINITY)?;
            println!("{}", atten_score);
        }

        // 应用缩放因子
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        // 应用 Softmax
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        // 与 V 相乘
        let atten_weight = softmax.matmul(&v)?; // (bs, n_head, seq_len, head_dim)

        // 重新整形为输出格式
        let atten_weight = atten_weight
            .transpose(1, 2)? // 转置回 (bs, seq_len, n_head, head_dim)
            .reshape((bs, seq_len, self.out_dim))?; // 合并头维度
        // 通过输出投影层
        let out = self.out_proj.forward(&atten_weight)?;
        Ok(out)
    }

    // 前向传播（带 RoPE）
    // 参数:
    //   x: 输入张量
    //   mask: 是否应用掩码
    pub fn forward_with_rope(&self, x: &Tensor, mask: bool) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?; // 获取批次大小和序列长度

        // 线性变换并重塑为多头格式
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

        // 创建 RoPE 并应用到 Q, K, V
        let rope = RoPE::new(seq_len, self.head_dim, x.device())?;
        let q = rope.forward(&q)?;
        let k = rope.forward(&k)?;
        let v = rope.forward(&v)?;

        // 计算注意力分数
        let mut atten_score = q.matmul(&k.t()?)?;
        if mask {
            // 应用掩码
            let mask = Tensor::tril2(seq_len, candle_core::DType::U32, x.device())?;
            atten_score = mask_filled(&atten_score, &mask, f32::NEG_INFINITY)?;
            println!("{}", atten_score);
        }
        // 应用缩放因子
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        // (bs, n_head, seq_len, seq_len)
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?; // (bs, n_head, seq_len, head_dim)

        // 重新整形并投影输出
        let atten_weight = atten_weight
            .transpose(1, 2)?
            .reshape((bs, seq_len, self.out_dim))?;
        let out = self.out_proj.forward(&atten_weight)?;
        Ok(out)
    }

    // 前向传播（使用共享缓冲区）
    // 参数:
    //   x: 输入张量
    //   buffer: 共享缓冲区，提供预计算的掩码和 RoPE
    pub fn forward_with_buffer(&self, x: &Tensor, buffer: &mut SharedBuffer) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?; // 获取批次大小和序列长度

        // 线性变换并重塑为多头格式
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

        // 从缓冲区获取掩码和 RoPE
        let (mask, rope) = buffer.get(seq_len, self.head_dim, x.device())?;
        // 应用 RoPE 到 Q, K, V
        let q = rope.forward(&q)?;
        let k = rope.forward(&k)?;
        let v = rope.forward(&v)?;

        // 计算注意力分数
        let mut atten_score = q.matmul(&k.t()?)?;
        // 应用掩码
        atten_score = mask_filled(&atten_score, mask, f32::NEG_INFINITY)?;
        println!("{}", atten_score);
        // 应用缩放因子
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        // 应用 Softmax
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?; // (bs, n_head, seq_len, head_dim)

        // 重新整形并投影输出
        let atten_weight = atten_weight
            .transpose(1, 2)?
            .reshape((bs, seq_len, self.out_dim))?;
        let out = self.out_proj.forward(&atten_weight)?;
        Ok(out)
    }
}

// 分组注意力结构体
// 实现分组查询注意力机制，减少计算复杂度
pub struct GroupAttention {
    w_q: Linear,       // 查询线性变换层
    w_k: Linear,       // 键线性变换层
    w_v: Linear,       // 值线性变换层
    out_proj: Linear,  // 输出投影层
    n_head: usize,     // 查询头数
    n_kv_head: usize,  // 键值头数（通常少于查询头数以减少计算）
    group_size: usize, // 每组的头数 (n_head / n_kv_head)
    head_dim: usize,   // 每个头的维度
    out_dim: usize,    // 输出维度
    d_sqrt: Tensor,    // 缩放因子 (1/sqrt(head_dim))
}

impl GroupAttention {
    // 创建新的分组注意力实例
    // 参数:
    //   vb: 变量构建器
    //   in_dim: 输入维度
    //   out_dim: 输出维度
    //   n_head: 查询头数
    //   n_kv_head: 键值头数（必须能整除 n_head）
    //   device: 计算设备
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        device: &Device,
    ) -> Result<Self> {
        // 验证输出维度必须能被头数整除
        assert_eq!(out_dim % n_head, 0, "out_dim must be divisible by n_head");
        // 验证查询头数必须能被键值头数整除
        assert_eq!(
            n_head % n_kv_head,
            0,
            "n_head must be divisible by n_kv_head"
        );

        let head_dim = out_dim / n_head; // 计算每个头的维度
        // 创建查询、键、值的线性变换层
        let w_q = linear_no_bias(in_dim, out_dim, vb.pp("w_q"))?;
        let w_k = linear_no_bias(in_dim, n_kv_head * head_dim, vb.pp("w_k"))?; // 键维度为 n_kv_head * head_dim
        let w_v = linear_no_bias(in_dim, n_kv_head * head_dim, vb.pp("w_v"))?; // 值维度为 n_kv_head * head_dim
        let out_proj = linear_no_bias(out_dim, out_dim, vb.pp("out_proj"))?;
        let group_size = n_head / n_kv_head; // 计算每组的头数
        let d_sqrt = 1.0 / (head_dim as f32).sqrt(); // 计算缩放因子
        let d_sqrt = Tensor::new(d_sqrt, device)?;

        Ok(Self {
            w_q,
            w_k,
            w_v,
            out_proj,
            n_head,
            n_kv_head,
            group_size,
            head_dim,
            out_dim,
            d_sqrt,
        })
    }

    // 前向传播
    // 参数:
    //   x: 输入张量
    //   buffer: 共享缓冲区，提供预计算的掩码和 RoPE
    pub fn forward(&self, x: &Tensor, buffer: &mut SharedBuffer) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?; // 获取批次大小和序列长度
        // 从缓冲区获取掩码和 RoPE
        let (mask, rope) = buffer.get(seq_len, self.head_dim, x.device())?;

        // 查询变换: (bs, seq_len, n_head, head_dim) -> (bs, n_head, seq_len, head_dim)
        let q = self
            .w_q
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)? // 转置为 (bs, n_head, seq_len, head_dim)
            .contiguous()?;
        // 键变换: (bs, seq_len, n_kv_head, head_dim) -> (bs, n_kv_head, seq_len, head_dim)
        let k = self
            .w_k
            .forward(x)?
            .reshape((bs, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        // 值变换: (bs, seq_len, n_kv_head, head_dim) -> (bs, n_kv_head, seq_len, head_dim)
        let v = self
            .w_v
            .forward(x)?
            .reshape((bs, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // 应用 RoPE 到 Q, K, V
        let q = rope.forward(&q)?;
        let k = rope.forward(&k)?;
        // 重复 K 张量以匹配查询头数
        let k = k.repeat((1, self.group_size, 1, 1))?;
        let v = rope.forward(&v)?;
        // 重复 V 张量以匹配查询头数
        let v = v.repeat((1, self.group_size, 1, 1))?;

        // 计算注意力分数
        let mut atten_score = q.matmul(&k.t()?)?;
        // 应用掩码
        atten_score = mask_filled(&atten_score, mask, f32::NEG_INFINITY)?;
        // 应用缩放因子
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        // 应用 Softmax
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        // 与 V 相乘
        let atten_weight = softmax.matmul(&v)?; // (bs, n_head, seq_len, head_dim)

        // 重新整形为输出格式
        let atten_weight = atten_weight
            .transpose(1, 2)? // 转置回 (bs, seq_len, n_head, head_dim)
            .reshape((bs, seq_len, self.out_dim))?; // 合并头维度
        // 通过输出投影层
        let out = self.out_proj.forward(&atten_weight)?;
        Ok(out)
    }
}

// 带 KV 缓存的分组注意力结构体
// 用于推理时的增量生成，通过缓存之前的 K 和 V 值来避免重复计算
pub struct GroupAttentionWithKVCache {
    w_q: Linear,             // 查询线性变换层
    w_k: Linear,             // 键线性变换层
    w_v: Linear,             // 值线性变换层
    out_proj: Linear,        // 输出投影层
    n_head: usize,           // 查询头数
    n_kv_head: usize,        // 键值头数
    group_size: usize,       // 每组的头数 (n_head / n_kv_head)
    head_dim: usize,         // 每个头的维度
    out_dim: usize,          // 输出维度
    max_context: usize,      // 最大上下文长度
    d_sqrt: Tensor,          // 缩放因子 (1/sqrt(head_dim))
    cache_k: Option<Tensor>, // K 缓存（可选）
    cache_v: Option<Tensor>, // V 缓存（可选）
}

impl GroupAttentionWithKVCache {
    // 创建新的带 KV 缓存的分组注意力实例
    // 参数:
    //   vb: 变量构建器
    //   in_dim: 输入维度
    //   out_dim: 输出维度
    //   n_head: 查询头数
    //   n_kv_head: 键值头数
    //   max_context: 最大上下文长度
    //   device: 计算设备
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        max_context: usize,
        device: &Device,
    ) -> Result<Self> {
        // 验证维度关系
        assert_eq!(out_dim % n_head, 0, "out_dim must be divisible by n_head");
        assert_eq!(
            n_head % n_kv_head,
            0,
            "n_head must be divisible by n_kv_head"
        );

        let head_dim = out_dim / n_head; // 计算每个头的维度
        // 创建线性变换层
        let w_q = linear_no_bias(in_dim, out_dim, vb.pp("w_q"))?;
        let w_k = linear_no_bias(in_dim, n_kv_head * head_dim, vb.pp("w_k"))?;
        let w_v = linear_no_bias(in_dim, n_kv_head * head_dim, vb.pp("w_v"))?;
        let out_proj = linear_no_bias(out_dim, out_dim, vb.pp("out_proj"))?;
        let group_size = n_head / n_kv_head; // 计算组大小
        let d_sqrt = 1.0 / (head_dim as f32).sqrt(); // 计算缩放因子
        let d_sqrt = Tensor::new(d_sqrt, device)?;

        Ok(Self {
            w_q,
            w_k,
            w_v,
            out_proj,
            n_head,
            n_kv_head,
            max_context, // 最大上下文长度
            group_size,
            head_dim,
            out_dim,
            d_sqrt,
            cache_k: None, // 初始化 K 缓存为 None
            cache_v: None, // 初始化 V 缓存为 None
        })
    }

    // 前向传播
    // 参数:
    //   x: 输入张量
    //   buffer: 共享缓冲区
    //   use_cache: 是否使用 KV 缓存
    //   pos_idx: 位置索引（用于 RoPE）
    pub fn forward(
        &mut self,
        x: &Tensor,
        buffers: &mut SharedBuffer,
        use_cache: bool,
        pos_idx: usize,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?; // 获取批次大小和序列长度
        // 从缓冲区获取掩码和 RoPE（使用最大上下文长度）
        let (mask, rope) = buffers.get(self.max_context, self.head_dim, x.device())?;

        // 查询变换: (bs, seq_len, n_head, head_dim) -> (bs, n_head, seq_len, head_dim)
        let mut q = self
            .w_q
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)? // 转置为 (bs, n_head, seq_len, head_dim)
            .contiguous()?;
        // 键变换: (bs, seq_len, n_kv_head, head_dim) -> (bs, n_kv_head, seq_len, head_dim)
        let mut k = self
            .w_k
            .forward(x)?
            .reshape((bs, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        // 值变换: (bs, seq_len, n_kv_head, head_dim) -> (bs, n_kv_head, seq_len, head_dim)
        let mut v = self
            .w_v
            .forward(x)?
            .reshape((bs, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        if use_cache {
            // 使用缓存模式：应用 RoPE 到指定位置
            q = rope.apply(&q, pos_idx)?; // 应用 RoPE 到指定位置范围
            k = rope.apply(&k, pos_idx)?; // 应用 RoPE 到指定位置范围
            v = rope.apply(&v, pos_idx)?; // 应用 RoPE 到指定位置范围（已修复：之前遗漏了v）

            // 管理 KV 缓存
            if self.cache_k.is_none() {
                // 如果没有缓存，创建新的缓存
                self.cache_k = Some(k.clone());
                self.cache_v = Some(v.clone());
            } else {
                // 如果已有缓存，将当前 K/V 与缓存连接
                if let Some(cache_k_) = &self.cache_k {
                    k = Tensor::cat(&[cache_k_, &k], D::Minus2)?; // 沿着序列维度连接
                    self.cache_k = Some(k.clone());
                }
                if let Some(cache_v_) = &self.cache_v {
                    v = Tensor::cat(&[cache_v_, &v], D::Minus2)?; // 沿着序列维度连接
                    self.cache_v = Some(v.clone());
                }
            }
        } else {
            // 非缓存模式：应用 RoPE 到完整序列
            q = rope.forward(&q)?; // 应用 RoPE 到完整序列
            k = rope.forward(&k)?; // 应用 RoPE 到完整序列
            v = rope.forward(&v)?; // 应用 RoPE 到完整序列
        }

        // 重复 K 和 V 张量以匹配查询头数
        let k = k.repeat((1, self.group_size, 1, 1))?; // 重复 K
        let v = v.repeat((1, self.group_size, 1, 1))?; // 重复 V

        // 计算注意力分数
        let mut atten_score = q.matmul(&k.t()?)?;
        if seq_len != 1 {
            // 如果序列长度不为1，应用掩码（防止看到未来信息）
            atten_score = mask_filled(&atten_score, mask, f32::NEG_INFINITY)?;
        }
        // 应用缩放因子
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        // 应用 Softmax
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        // 与 V 相乘
        let atten_weight = softmax.matmul(&v)?; // (bs, n_head, seq_len, head_dim)

        // 重新整形为输出格式
        let atten_weight = atten_weight
            .transpose(1, 2)? // 转置回 (bs, seq_len, n_head, head_dim)
            .reshape((bs, seq_len, self.out_dim))?; // 合并头维度
        // 通过输出投影层
        let out = self.out_proj.forward(&atten_weight)?;
        Ok(out)
    }

    pub fn reset_kv_cache(&mut self) {
        self.cache_k = None;
        self.cache_v = None;
    }
}

pub struct RMSNorm {
    weight: Tensor,
    eps: Tensor,
}

impl RMSNorm {
    pub fn new(vb: VarBuilder, eps: f32, dim: usize) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", Init::Const(1.0))?;
        let eps = Tensor::new(eps, vb.device())?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.powf(2.0)?.mean(D::Minus1)?;
        let rms = mean
            .broadcast_add(&self.eps)?
            .sqrt()?
            .unsqueeze(D::Minus1)?;
        let x_norm = x.broadcast_div(&rms)?;
        let x_norm = x_norm.broadcast_mul(&self.weight)?;
        Ok(x_norm)
    }
}

pub struct FeedForward {
    up: Linear,
    gate: Linear,
    down: Linear,
}

impl FeedForward {
    pub fn new(vb: VarBuilder, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Result<Self> {
        let up = linear_no_bias(in_dim, hidden_dim, vb.pp("up"))?;
        let gate = linear_no_bias(in_dim, hidden_dim, vb.pp("gate"))?;
        let down = linear_no_bias(hidden_dim, out_dim, vb.pp("down"))?;

        Ok(Self { up, gate, down })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let up_x = self.up.forward(x)?;
        let gate_x = self.gate.forward(x)?.silu()?;
        let mul_cat = up_x.mul(&gate_x)?;
        let down = self.down.forward(&mul_cat)?;
        Ok(down)
    }
}

pub struct LLMConfig {
    vocab_size: usize,
    embedding_dim: usize,
    n_block: usize,
    n_head: usize,
    n_kv_head: usize,
    max_context: usize,
    hidden_dim: usize,
    eps: f32,
}

impl LLMConfig {
    pub fn default() -> Result<LLMConfig> {
        Ok(Self {
            vocab_size: 151669,
            embedding_dim: 512,
            n_block: 8,
            n_head: 8,
            n_kv_head: 4,
            max_context: 512,
            hidden_dim: 1024,
            eps: 1e-6,
        })
    }
}

pub struct AttentionBlock {
    rms_norm1: RMSNorm,
    attention: GroupAttentionWithKVCache,
    rms_norm2: RMSNorm,
    feed_forward: FeedForward,
}

impl AttentionBlock {
    pub fn new(vb: VarBuilder, config: &LLMConfig) -> Result<Self> {
        let rms_norm1 = RMSNorm::new(vb.pp("rms_norm1"), config.eps, config.embedding_dim)?;
        let attention = GroupAttentionWithKVCache::new(
            vb.pp("attention"),
            config.embedding_dim,
            config.embedding_dim,
            config.n_head,
            config.n_kv_head,
            config.max_context,
            vb.device(),
        )?;
        let rms_norm2 = RMSNorm::new(vb.pp("rms_norm2"), config.eps, config.embedding_dim)?;
        let feed_forward = FeedForward::new(
            vb.pp("feed_forward"),
            config.embedding_dim,
            config.hidden_dim,
            config.embedding_dim,
        )?;

        Ok(Self {
            rms_norm1,
            attention,
            rms_norm2,
            feed_forward,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        buffers: &mut SharedBuffer,
        use_cache: bool,
        pos_idx: usize,
    ) -> Result<Tensor> {
        let x_norm1 = self.rms_norm1.forward(x)?;
        let x_atten = self
            .attention
            .forward(&x_norm1, buffers, use_cache, pos_idx)?;
        let shortcut = x.add(&x_atten)?;
        let x_norm2 = self.rms_norm2.forward(&shortcut)?;
        let x_feed = self.feed_forward.forward(&x_norm2)?;
        let shortcut = shortcut.add(&x_feed)?;
        Ok(shortcut)
    }

    pub fn reset_kv_cache(&mut self) {
        self.attention.reset_kv_cache();
    }
}

pub struct LLM {
    embedding: Embedding,
    attention_blocks: Vec<AttentionBlock>,
    final_rms: RMSNorm,
    out_proj: Linear,
}

impl LLM {
    pub fn new(vb: VarBuilder, config: &LLMConfig) -> Result<Self> {
        let embedding = embedding(config.vocab_size, config.embedding_dim, vb.pp("embedding"))?;
        let mut attention_blocks = Vec::new();
        for i in 0..config.n_block {
            let block = AttentionBlock::new(vb.pp(format!("block_{i}")), config)?;
            attention_blocks.push(block);
        }
        let final_rms = RMSNorm::new(vb.pp("final_rms"), config.eps, config.embedding_dim)?;
        let out_proj = linear_no_bias(config.embedding_dim, config.vocab_size, vb.pp("out_proj"))?;

        Ok(Self {
            embedding,
            attention_blocks,
            final_rms,
            out_proj,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        buffers: &mut SharedBuffer,
        use_cache: bool,
        pos_idx: usize,
    ) -> Result<Tensor> {
        let mut x = self.embedding.forward(x)?;
        for block in &mut self.attention_blocks {
            x = block.forward(&x, buffers, use_cache, pos_idx)?;
        }
        let x = self.final_rms.forward(&x)?;
        let x = self.out_proj.forward(&x)?;
        Ok(x)
    }

    pub fn reset_kv_cache(&mut self) {
        for block in &mut self.attention_blocks {
            block.reset_kv_cache();
        }
    }
}

pub fn encode_str(str: &str, tokenizer: &Tokenizer, device: &Device) -> Result<Tensor> {
    let encode = tokenizer
        .encode(str, true)
        .map_err(|e| Error::Msg(format!("tokenizer encode error:{}", e)))?;
    let token_ids = encode.get_ids();
    let len = token_ids.len();
    let tensor = Tensor::from_slice(token_ids, (1, len), device)?;
    Ok(tensor)
}

pub fn decode_tokens(token_ids: &Tensor, tokenizer: &Tokenizer) -> Result<String> {
    let token_ids_vec = match token_ids.rank() {
        1 => token_ids.to_vec1()?,
        2 => token_ids.squeeze(0)?.to_vec1()?,
        _ => {
            return Err(Error::Msg(format!(
                "can't active this rank {} Tensor",
                token_ids.rank()
            )));
        }
    };

    let decode = tokenizer
        .decode(&token_ids_vec, true)
        .map_err(|e| Error::Msg(format!("tokens decode error:{}", e)))?;
    Ok(decode)
}

pub fn generate_simple(
    model: &mut LLM,
    buffers: &mut SharedBuffer,
    idx: &Tensor,
    max_generate: usize,
    max_context: usize,
) -> Result<Tensor> {
    model.reset_kv_cache();
    let mut idx = idx.clone();
    let (_, num_tokens) = idx.dims2()?;
    if num_tokens > max_context {
        let start = num_tokens - max_context;
        idx = idx.i((.., start..num_tokens))?;
    }
    let mut pos_idx = 0;
    let mut logits = model.forward(&idx, buffers, true, pos_idx)?;
    for _ in 0..max_generate {
        let (_, n_token, _) = logits.dims3()?;
        pos_idx += n_token;
        if pos_idx > max_context {
            pos_idx = 0;
        }
        logits = logits.i((.., n_token - 1, ..))?;
        let probs = ops::softmax(&logits, D::Minus1)?;
        let mut idx_next = probs.argmax(D::Minus1)?;
        if idx_next.rank() == 1 {
            idx_next = idx_next.unsqueeze(0)?;
        }
        idx = Tensor::cat(&[&idx, &idx_next], D::Minus1)?;
        logits = model.forward(&idx_next, buffers, true, pos_idx)?;
    }
    Ok(idx)
}

// 主函数：演示模型的使用流程
fn main() -> Result<()> {
    // 初始化设备（优先使用 Metal，否则使用 CPU）
    let device = Device::metal_if_available(0)?;

    // 加载文本数据和分词器
    let text = read_text("src/assets/sub_wiki_0_99.txt");
    let tokenizer = Tokenizer::from_file("src/assets/tokenizer.json")
        .map_err(|e| Error::Msg(format!("tokenizer from file error {}", e)))?;

    // 设置数据处理参数
    let batch_size = 2; // 批次大小
    let config = LLMConfig::default()?;

    // 创建数据集和数据加载器
    let token_dataset = TokenDataset::new(
        text,
        &tokenizer,
        config.max_context,
        config.max_context,
        &device,
    )?;
    let mut dataloader = DataLoader::new(token_dataset, batch_size, true)?; // 启用随机打乱
    let _ = dataloader.reset()?; // 重置数据加载器
    let (x, _y) = dataloader.next().unwrap()?; // 获取第一个批次的数据

    let varmap = VarMap::new(); // 创建变量映射
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device); // 创建变量构建器
    let mut buffers = SharedBuffer::new()?;
    let mut llm = LLM::new(vb, &config)?;
    // let _ = print_varmap(&varmap)?;
    // let output = llm.forward(&x, &mut buffers, false, 0)?;
    // println!("output: {:?}", output);
    let token_tensor = encode_str("吃饭了吗", &tokenizer, &device)?;
    let generate_token = generate_simple(
        &mut llm,
        &mut buffers,
        &token_tensor,
        100,
        config.max_context,
    )?;
    let str = decode_tokens(&generate_token, &tokenizer)?;
    println!("{}", str);

    Ok(())
}
