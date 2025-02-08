use ndarray::{s, Array, Array2, Array3, Axis, concatenate};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let max = x.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    let exp_x = x - &max.insert_axis(Axis(1));
    let exp_x = exp_x.mapv(f32::exp);
    let sum_exp_x = exp_x.sum_axis(Axis(1)).insert_axis(Axis(1));
    &exp_x / sum_exp_x
}

fn attention(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
    let d_k = q.shape()[1] as f32;
    println!("q.shape: {:?}, k.shape: {:?}, v.shape: {:?}", q.shape(), k.shape(), v.shape());
    let scores = q.dot(&k.t()) / d_k.sqrt();
    println!("scores.shape: {:?}", scores.shape());
    let attn_weights = softmax(&scores);
    println!("attn_weights.shape: {:?}", attn_weights.shape());
    println!("attn_weights: {:?}", attn_weights);
    attn_weights.dot(v)
}

fn feed_forward(x: &Array2<f32>, w1: &Array2<f32>, w2: &Array2<f32>) -> Array2<f32> {
    let hidden = x.dot(w1).mapv(|x| x.max(0.0)); // ReLU
    hidden.dot(w2)
}

fn layer_norm(x: &Array2<f32>) -> Array2<f32> {
    let mean = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    let var = x.var_axis(Axis(1), 0.0).insert_axis(Axis(1));
    (x - &mean) / (var.mapv(f32::sqrt) + 1e-5)
}

fn multi_head_attention(
    x: &Array2<f32>, w_q: &Array3<f32>, w_k: &Array3<f32>, w_v: &Array3<f32>, w_o: &Array2<f32>
) -> Array2<f32> {
    let (heads, d_model, d_head) = (w_q.shape()[0], w_q.shape()[1], w_q.shape()[2]);

    let mut head_outputs = Vec::new();
    for h in 0..heads {
        let q = x.dot(&w_q.slice(s![h, .., ..]));
        let k = x.dot(&w_k.slice(s![h, .., ..]));
        let v = x.dot(&w_v.slice(s![h, .., ..]));

        let d_k = d_head as f32;
        let scores = q.dot(&k.t()) / d_k.sqrt();
        let attn_weights = softmax(&scores);
        let attn_output = attn_weights.dot(&v);

        head_outputs.push(attn_output);
    }

    let multi_head_output = concatenate(Axis(1), &head_outputs.iter().map(|o| o.view()).collect::<Vec<_>>()).unwrap();
    multi_head_output.dot(w_o)
}

fn positional_encoding(seq_len: usize, d_model: usize) -> Array2<f32> {
    let mut pe = Array::zeros((seq_len, d_model));
    for pos in 0..seq_len {
        for i in (0..d_model).step_by(2) {
            let div_term = (pos as f32) / (10000.0_f32.powf(i as f32 / d_model as f32));
            pe[[pos, i]] = div_term.sin();
            if i + 1 < d_model {
                pe[[pos, i + 1]] = div_term.cos();
            }
        }
    }
    pe
}

/// 完整的 Transformer Block
fn transformer_block(
    x: &Array2<f32>, w_q: &Array3<f32>, w_k: &Array3<f32>, w_v: &Array3<f32>, w_o: &Array2<f32>,
    w1: &Array2<f32>, w2: &Array2<f32>
) -> Array2<f32> {
    // 1. 多頭注意力
    let attn_output = multi_head_attention(x, w_q, w_k, w_v, w_o);

    // 2. 殘差 + LayerNorm
    let attn_residual = layer_norm(&(x + &attn_output));

    // 3. 前饋網絡 (FFN)
    let ffn_output = feed_forward(&attn_residual, w1, w2);

    // 4. 最終的殘差 + LayerNorm
    layer_norm(&(attn_residual + ffn_output))
}


fn main() {
    let seq_len = 4;  // 序列長度
    let d_model = 8;  // 每個 token 的維度
    let num_heads = 2;
    let d_head = d_model / num_heads;

    // 隨機初始化 embedding (假設 vocab_size = 10)
     let embedding = Array2::random((10, d_model), Uniform::new(-0.1, 0.1));

    // 隨機初始化輸入 token (0~9)
    let input_tokens = [1, 3, 7, 2];
    let input_embeds = Array::from_shape_fn((seq_len, d_model), |(i, j)| embedding[[input_tokens[i], j]]);

    // 生成位置編碼
    let pe = positional_encoding(seq_len, d_model);
    println!("pe.shape: {:?}, pe: {:?}", pe.shape(), pe);

    // 初始化 Self-Attention 參數
    // let w_q = Array::random((d_model, d_model), Uniform::new(-0.1, 0.1));
    // let w_k = Array::random((d_model, d_model), Uniform::new(-0.1, 0.1));
    // let w_v = Array::random((d_model, d_model), Uniform::new(-0.1, 0.1));

    // 初始化多頭注意力參數
    let w_q = Array::random((num_heads, d_model, d_head), Uniform::new(-0.1, 0.1));
    let w_k = Array::random((num_heads, d_model, d_head), Uniform::new(-0.1, 0.1));
    let w_v = Array::random((num_heads, d_model, d_head), Uniform::new(-0.1, 0.1));
    let w_o = Array::random((d_model, d_model), Uniform::new(-0.1, 0.1));


    // 計算 Attention
    // let q = input_embeds.dot(&w_q);
    // let k = input_embeds.dot(&w_k);
    // let v = input_embeds.dot(&w_v);
    // let attn_output = attention(&q, &k, &v);

    let input_with_pe = &input_embeds + &pe;

    /*
    // 執行多頭注意力
    let attn_output = multi_head_attention(&input_with_pe, &w_q, &w_k, &w_v, &w_o);

    // 加上殘差並做 LayerNorm
    let attn_residual = layer_norm(&(input_with_pe + &attn_output));

    // 前饋層參數
    let w1 = Array::random((d_model, d_model * 2), Uniform::new(-0.1, 0.1));
    let w2 = Array::random((d_model * 2, d_model), Uniform::new(-0.1, 0.1));

    // 前饋網絡
    let ffn_output = feed_forward(&attn_residual, &w1, &w2);

    // 最終的 Residual + LayerNorm
    let output = layer_norm(&(attn_residual + ffn_output));
     */

    // 前饋層參數
    let w1 = Array::random((d_model, d_model * 2), Uniform::new(-0.1, 0.1));
    let w2 = Array::random((d_model * 2, d_model), Uniform::new(-0.1, 0.1));

    // 執行 Transformer Block
    let output = transformer_block(&input_with_pe, &w_q, &w_k, &w_v, &w_o, &w1, &w2);

    println!("Final Transformer Output:\n{:?}", output);
}
