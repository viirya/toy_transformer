use candle_core::{Module, Tensor, DType, Device, Result};
use candle_nn::{linear, Embedding, LayerNorm, Linear, VarBuilder};
use candle_nn::init::{ONE, ZERO};
use candle_nn::ops::softmax;

/// Multi-Head Self-Attention (MHSA)
struct MultiHeadSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadSelfAttention {
    fn new(vb: &VarBuilder, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            q_proj: linear(embed_dim, embed_dim, vb.clone())?,
            k_proj: linear(embed_dim, embed_dim, vb.clone())?,
            v_proj: linear(embed_dim, embed_dim, vb.clone())?,
            out_proj: linear(embed_dim, embed_dim, vb.clone())?,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, embed_dim) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        let scores = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let attn = softmax(&scores, 3)?;
        let context = attn.matmul(&v)?;

        let context = context.reshape((batch_size, seq_len, embed_dim))?;
        self.out_proj.forward(&context)
    }
}

/// Feed-Forward Network (FFN)
struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    fn new(vb: &VarBuilder, embed_dim: usize, hidden_dim: usize) -> Result<Self> {
        Ok(Self {
            fc1: linear(embed_dim, hidden_dim, vb.clone())?,
            fc2: linear(hidden_dim, embed_dim, vb.clone())?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?.relu()?;
        self.fc2.forward(&x)
    }
}

/// Transformer Block
struct TransformerBlock {
    mha: MultiHeadSelfAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerBlock {
    fn new(vb: &VarBuilder, embed_dim: usize, num_heads: usize, ffn_dim: usize) -> Result<Self> {
        let weight = vb.get_with_hints((embed_dim,), "layernorm_weight", ONE)?;
        let bias = vb.get_with_hints((embed_dim,), "layernorm_bias", ZERO)?;

        let eps = 1e-5;

        Ok(Self {
            mha: MultiHeadSelfAttention::new(vb, embed_dim, num_heads)?,
            ffn: FeedForward::new(vb, embed_dim, ffn_dim)?,
            norm1: LayerNorm::new(weight.clone(), bias.clone(), eps),
            norm2: LayerNorm::new(weight, bias, eps),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-Norm:
        // LayerNorm(x) as the input to Multi-Head Attention
        let x = (x + self.mha.forward(&self.norm1.forward(x)?)?)?;
        x.clone() + self.ffn.forward(&self.norm2.forward(&x)?)?
    }
}

struct Transformer {
    layers: Vec<TransformerBlock>,
    embedding: Embedding,
}

impl Transformer {
    fn new(vb: &VarBuilder, vocab_size: usize, embed_dim: usize, num_heads: usize, num_layers: usize, ffn_dim: usize) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(TransformerBlock::new(vb, embed_dim, num_heads, ffn_dim)?);
        }

        let embeddings = Tensor::randn(0.0f32, 5.0f32, (vocab_size, embed_dim), &Device::Cpu)?;
        Ok(Self {
            layers,
            embedding: Embedding::new(embeddings, embed_dim),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = self.embedding.forward(x)?;
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let transformer = Transformer::new(&vb, 10000, 256, 8, 6, 512)?;

    let input = Tensor::rand(0.0, 10000.0, (8, 32), &device)?
        .abs()?
        .broadcast_mul(&Tensor::from_slice(&[10000.0], (1,), &device)?)?
        .to_dtype(DType::U32)?
        .clamp(0u32, 9999u32)?;

    println!("{:?}", input.shape());
    let output = transformer.forward(&input)?;

    println!("{:?}", output.shape());
    Ok(())
}
