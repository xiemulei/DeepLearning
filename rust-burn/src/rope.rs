//! RoPE (Rotary Positional Embedding) implementation for Burn framework.
//! This module provides position encoding for transformers using rotary embeddings.

use burn::{prelude::*, tensor::Tensor};

/// RoPE (Rotary Positional Embedding) structure
///
/// Rotary embeddings encode positional information by rotating vectors in high-dimensional space.
/// This is more effective than absolute position embeddings for long sequences.
#[derive(Module, Debug)]
pub struct RoPE<B: Backend> {
    cos: Tensor<B, 2>, // [seq_len, head_dim]
    sin: Tensor<B, 2>, // [seq_len, head_dim]
}

impl<B: Backend> RoPE<B> {
    /// Create a new RoPE instance with pre-computed sin and cos values
    ///
    /// # Arguments
    /// * `seq_len` - Maximum sequence length to support
    /// * `head_dim` - Dimension of each attention head (must be even)
    /// * `device` - Device to create tensors on
    /// * `theta` - Base frequency (Qwen3 uses 1000000, default is 10000)
    ///
    /// # Returns
    /// RoPE instance with pre-computed sin and cos values
    pub fn new(seq_len: usize, head_dim: usize, device: &B::Device, theta: f64) -> Self {
        assert_eq!(
            head_dim % 2,
            0,
            "head_dim must be even for rotary embeddings"
        );

        // Create position indices [0, 1, 2, ..., seq_len-1]
        let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let pos_data = TensorData::new(positions, [seq_len, 1]);
        let pos = Tensor::<B, 2>::from_data(pos_data.convert::<f32>(), device);

        // Calculate inverse frequency: 1 / (theta^(2i/d))
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| {
                let theta_pow = theta.powf((2 * i) as f64 / head_dim as f64);
                (1.0 / theta_pow) as f32
            })
            .collect();

        let inv_freq_data = TensorData::new(inv_freq, [1, head_dim / 2]);
        let inv_freq = Tensor::<B, 2>::from_data(inv_freq_data.convert::<f32>(), device);

        // Compute angle tensor: positions * inv_freq
        let angle = pos.matmul(inv_freq);

        // Duplicate along last dimension to match head_dim
        let angle = angle.unsqueeze_dim::<3>(2).repeat(&[1, 1, 2]);
        let angle = angle.reshape([seq_len, head_dim]);

        // Compute sin and cos
        let sin = angle.clone().sin();
        let cos = angle.cos();

        Self { sin, cos }
    }

    /// Apply RoPE transformation to query and key tensors
    ///
    /// This rotates Q and K tensors according to their positions.
    /// Only Q and K need rotation, V does not.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, n_head, seq_len, head_dim]
    /// * `k` - Key tensor [batch, n_kv_head, seq_len, head_dim]
    ///
    /// # Returns
    /// Tuple of (rotated_q, rotated_k)
    pub fn apply(&self, q: Tensor<B, 4>, k: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let rotated_q = self.rotate(q);
        let rotated_k = self.rotate(k);
        (rotated_q, rotated_k)
    }

    /// Core rotation operation for RoPE
    ///
    /// Implements: x * cos(mθ) + rotate(x) * sin(mθ)
    /// where rotate(x) = [-x2, x1, -x4, x3, ...]
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, n_head, seq_len, head_dim]
    ///
    /// # Returns
    /// Rotated tensor [batch, n_head, seq_len, head_dim]
    fn rotate(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, seq_len, _head_dim] = x.dims();
        let half_dim = _head_dim / 2;

        // Split x into two halves
        let x1 = x.clone().narrow(3, 0, half_dim); // [batch, n_head, seq_len, head_dim/2]
        let x2 = x.clone().narrow(3, half_dim, half_dim); // [batch, n_head, seq_len, head_dim/2]

        // Negate second half for rotation
        let x2_neg = x2.mul_scalar(-1.0);

        // Concatenate [-x2, x1] to create rotated version
        let rotate_x = Tensor::cat(vec![x2_neg, x1], 3); // [batch, n_head, seq_len, head_dim]

        // Extract relevant sin/cos for sequence length
        let cos_seq = self.cos.clone().narrow(0, 0, seq_len); // [seq_len, head_dim]
        let sin_seq = self.sin.clone().narrow(0, 0, seq_len); // [seq_len, head_dim]

        // Reshape sin/cos to [1, 1, seq_len, head_dim] for broadcasting
        let cos_seq = cos_seq.unsqueeze_dim::<4>(0).unsqueeze_dim::<4>(0);
        let sin_seq = sin_seq.unsqueeze_dim::<4>(0).unsqueeze_dim::<4>(0);

        // Apply rotation formula: x * cos + rotate(x) * sin
        let x_cos = x.mul(cos_seq);
        let x_sin = rotate_x.mul(sin_seq);
        let rotated = x_cos.add(x_sin);

        rotated
    }

    /// Apply RoPE with position offset (for caching during inference)
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, n_head, seq_len, head_dim]
    /// * `pos_offset` - Starting position index
    ///
    /// # Returns
    /// Rotated tensor [batch, n_head, seq_len, head_dim]
    pub fn apply_with_offset(&self, x: Tensor<B, 4>, pos_offset: usize) -> Tensor<B, 4> {
        let [_, _, seq_len, _head_dim] = x.dims();
        let end_pos = pos_offset + seq_len;

        assert!(
            end_pos <= self.cos.dims()[0],
            "RoPE sequence length exceeded: requested {}, max {}",
            end_pos,
            self.cos.dims()[0]
        );

        // Extract sin/cos for relevant position range
        let cos_range = self.cos.clone().narrow(0, pos_offset, seq_len);
        let sin_range = self.sin.clone().narrow(0, pos_offset, seq_len);

        // Create temporary RoPE with extracted sin/cos
        let rope = Self {
            sin: sin_range,
            cos: cos_range,
        };

        rope.rotate(x)
    }
}
