use candle_core::{Module, Tensor, DType, Device, Result, Var};
use candle_nn::{linear, Embedding, LayerNorm, Linear, Optimizer, VarBuilder, SGD};
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

    fn parameters(&self) -> Result<Vec<Var>> {
        Ok(vec![Var::from_tensor(self.q_proj.weight())?, Var::from_tensor(self.q_proj.bias().unwrap())?,
             Var::from_tensor(self.k_proj.weight())?, Var::from_tensor(self.k_proj.bias().unwrap())?,
             Var::from_tensor(self.v_proj.weight())?, Var::from_tensor(self.v_proj.bias().unwrap())?,
                Var::from_tensor(self.out_proj.weight())?, Var::from_tensor(self.out_proj.bias().unwrap())?])
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

    fn parameters(&self) -> Result<Vec<Var>> {
        Ok(vec![Var::from_tensor(self.fc1.weight())?, Var::from_tensor(self.fc1.bias().unwrap())?,
                Var::from_tensor(self.fc2.weight())?, Var::from_tensor(self.fc2.bias().unwrap())?])
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
        let weight1 = vb.get_with_hints((embed_dim,), "layernorm_weight1", ONE)?;
        let bias1 = vb.get_with_hints((embed_dim,), "layernorm_bias1", ZERO)?;

        let weight2 = vb.get_with_hints((embed_dim,), "layernorm_weight2", ONE)?;
        let bias2 = vb.get_with_hints((embed_dim,), "layernorm_bias2", ZERO)?;

        let eps = 1e-5;

        Ok(Self {
            mha: MultiHeadSelfAttention::new(vb, embed_dim, num_heads)?,
            ffn: FeedForward::new(vb, embed_dim, ffn_dim)?,
            norm1: LayerNorm::new(weight1, bias1, eps),
            norm2: LayerNorm::new(weight2, bias2, eps),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-Norm:
        // LayerNorm(x) as the input to Multi-Head Attention
        let x = (x + self.mha.forward(&self.norm1.forward(x)?)?)?;
        x.clone() + self.ffn.forward(&self.norm2.forward(&x)?)?
    }

    fn parameters(&self) -> Result<Vec<Var>> {
        let mut params = vec![];

        params.push(Var::from_tensor(self.norm1.weight())?);
        params.push(Var::from_tensor(self.norm1.bias().unwrap())?);
        params.push(Var::from_tensor(self.norm2.weight())?);
        params.push(Var::from_tensor(self.norm2.bias().unwrap())?);

        params.extend(self.mha.parameters()?);
        params.extend(self.ffn.parameters()?);

        Ok(params)
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

    fn embedding(&self, x: &Tensor) -> Result<Tensor> {
        self.embedding.forward(x)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = self.embedding.forward(x)?;
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }

    fn parameters(&self) -> Result<Vec<Var>> {
        let mut params = vec![Var::from_tensor(self.embedding.embeddings())?];
        for layer in &self.layers {
            params.extend(layer.parameters()?);
        }
        Ok(params)
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let transformer = Transformer::new(&vb, 10000, 256, 8, 6, 512)?;
    let vars = transformer.parameters()?;

    let input = Tensor::rand(0.0, 10000.0, (8, 32), &device)?
        .abs()?
        .broadcast_mul(&Tensor::from_slice(&[10000.0], (1,), &device)?)?
        .to_dtype(DType::U32)?
        .clamp(0u32, 9999u32)?;

    println!("input.shape: {:?}", input.shape());

    let target = Tensor::rand(0.0, 10000.0, (8, 32), &device)?
        .abs()?
        .broadcast_mul(&Tensor::from_slice(&[10000.0], (1,), &device)?)?
        .to_dtype(DType::U32)?
        .clamp(0u32, 9999u32)?;


    let mut sgd = SGD::new(vars, 0.01)?;

    for step in 0..100 {
        let output = transformer.forward(&input)?;

        println!("output.shape: {:?}", output.shape());

        let target = transformer.embedding(&target)?;

        println!("target.shape: {:?}", target.shape());

        let loss = (&output - &target)?.sqr()?.mean(2)?;

        println!("loss.shape: {:?}", loss.shape());

        sgd.backward_step(&loss)?;

        if step % 10 == 0 {
            println!("Step {}: Loss = {:?}", step, loss.to_vec2::<f32>()?);
        }
    }

    Ok(())
}
