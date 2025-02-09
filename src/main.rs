use ndarray::{s, stack, Array, Array2, Array3, Axis, concatenate};
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

fn feed_forward(x: &Array2<f32>, w1: &Array2<f32>, w2: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    // H = ReLU(X x W1)
    let hidden = x.dot(w1).mapv(|x| x.max(0.0)); // ReLU
    // Y = H x W2
    let y = hidden.dot(w2);

    (y, hidden)
}

fn layer_norm(x: &Array2<f32>) -> Array2<f32> {
    let mean = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    let var = x.var_axis(Axis(1), 0.0).insert_axis(Axis(1));
    (x - &mean) / (var.mapv(f32::sqrt) + 1e-5)
}

fn multi_head_attention(
    x: &Array2<f32>, w_q: &Array3<f32>, w_k: &Array3<f32>, w_v: &Array3<f32>, w_o: &Array2<f32>
) -> (Array2<f32>, Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array2<f32>>) {
    let (heads, d_model, d_head) = (w_q.shape()[0], w_q.shape()[1], w_q.shape()[2]);

    let mut head_outputs = Vec::new();
    let mut all_attn_weights = Vec::new();
    let mut all_q = Vec::new();
    let mut all_k = Vec::new();
    let mut all_v = Vec::new();
    for h in 0..heads {
        let q = x.dot(&w_q.slice(s![h, .., ..]));
        let k = x.dot(&w_k.slice(s![h, .., ..]));
        let v = x.dot(&w_v.slice(s![h, .., ..]));

        let d_k = d_head as f32;
        let scores = q.dot(&k.t()) / d_k.sqrt();
        let attn_weights = softmax(&scores);
        let attn_output = attn_weights.dot(&v);

        all_q.push(q);
        all_k.push(k);
        all_v.push(v);
        all_attn_weights.push(attn_weights);
        head_outputs.push(attn_output);
    }

    let multi_head_output = concatenate(Axis(1), &head_outputs.iter().map(|o| o.view()).collect::<Vec<_>>()).unwrap();
    (multi_head_output.dot(w_o), all_attn_weights, all_q, all_k, all_v)
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
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array2<f32>>) {
    // 1. 多頭注意力
    let (attn_output, all_attn_weights, all_q, all_k, all_v) = multi_head_attention(x, w_q, w_k, w_v, w_o);

    // 2. 殘差 + LayerNorm
    let attn_residual = layer_norm(&(x + &attn_output));

    // 3. 前饋網絡 (FFN)
    let (ffn_output, hidden) = feed_forward(&attn_residual, w1, w2);

    // 4. 最終的殘差 + LayerNorm
    (layer_norm(&(attn_residual.clone() + ffn_output)), attn_output, attn_residual, hidden, all_attn_weights, all_q, all_k, all_v)
}

/// 均方誤差 (Mean Squared Error, MSE)
fn mean_squared_error(pred: &Array2<f32>, target: &Array2<f32>) -> f32 {
    let diff = pred - target;
    diff.mapv(|x| x.powi(2)).mean().unwrap()
}

/// 梯度下降更新函數 (SGD)
fn sgd_update(param: &mut Array2<f32>, grad: &Array2<f32>, lr: f32) {
    *param -= &(grad * lr);
}

fn sgd_update_3dim(param: &mut Array3<f32>, grad: &Array3<f32>, lr: f32) {
    *param -= &(grad * lr);
}

fn main() {
    let seq_len = 4;  // 序列長度
    let d_model = 8;  // 每個 token 的維度
    let num_heads = 2;
    let d_head = d_model / num_heads;

    let lr = 0.0001;
    let epochs = 10000;

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
    let mut w_q = Array::random((num_heads, d_model, d_head), Uniform::new(-0.1, 0.1));
    let mut w_k = Array::random((num_heads, d_model, d_head), Uniform::new(-0.1, 0.1));
    let mut w_v = Array::random((num_heads, d_model, d_head), Uniform::new(-0.1, 0.1));
    let mut w_o = Array::random((d_model, d_model), Uniform::new(-0.1, 0.1));

    // 計算 Attention
    // let q = input_embeds.dot(&w_q);
    // let k = input_embeds.dot(&w_k);
    // let v = input_embeds.dot(&w_v);
    // let attn_output = attention(&q, &k, &v);

    // 前饋層參數
    let mut w1 = Array::random((d_model, d_model * 2), Uniform::new(-0.1, 0.1));
    let mut w2 = Array::random((d_model * 2, d_model), Uniform::new(-0.1, 0.1));

    let input_with_pe = &input_embeds + &pe;

    // 目標輸出
    let target_output = Array::random((seq_len, d_model), Uniform::new(-0.1, 0.1));

    for epoch in 0..epochs {
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


        // 執行 Transformer Block
        let (output, attn_output, attn_residual, hidden, all_attn_weights, all_q, all_k, all_v) = transformer_block(&input_with_pe, &w_q, &w_k, &w_v, &w_o, &w1, &w2);
        // println!("output.shape: {:?}", output.shape());

        let loss = mean_squared_error(&output, &target_output);

        // 反向傳播 (Backpropagation) 訓練

        // 計算 FFN 的反向傳播
        let grad_output = 2.0 * (&output - &target_output) / (seq_len as f32);
        // println!("grad_output.shape: {:?}", grad_output.shape());

        // 計算 W2 梯度
        let grad_w2 = hidden.t().dot(&grad_output);
        // 計算 H 的梯度
        let grad_h = grad_output.dot(&w2.t());

        // 計算 ReLU 導數
        let relu_deriv = hidden.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let grad_h_relu = &grad_h * &relu_deriv;

        // 計算 W1 梯度
        let grad_w1 = attn_residual.t().dot(&grad_h_relu);

        // 計算 A' 的梯度
        // (簡化) 計算 A 的梯度
        let grad_a_residual = grad_h_relu.dot(&w1.t());
        let grad_a = grad_a_residual.clone();

        // 多頭注意力 (Multi-Head Attention, MHA) 的反向傳播
        let grad_w_o = attn_output.t().dot(&grad_a);

        let grad_attention = grad_output.dot(&w_o.t());

        let grad_attention_heads: Vec<Array2<f32>> = grad_attention
            .axis_chunks_iter(Axis(1), d_head)
            .map(|chunk| chunk.to_owned())
            .collect();

        let mut grad_w_q_heads = Vec::new();
        let mut grad_w_k_heads = Vec::new();
        let mut grad_w_v_heads = Vec::new();
        for h in 0..num_heads {
            let grad_attention_h = &grad_attention_heads[h];
            let grad_v_h = all_attn_weights[h].t().dot(grad_attention_h);
            let grad_s_h = grad_attention_h.dot(&all_v[h].t());
            let grad_q_h = grad_s_h.dot(&all_k[h]) / (d_head as f32).sqrt();
            let grad_k_h = grad_s_h.t().dot(&all_q[h]) / (d_head as f32).sqrt();

            grad_w_q_heads.push(input_with_pe.t().dot(&grad_q_h));
            grad_w_k_heads.push(input_with_pe.t().dot(&grad_k_h));
            grad_w_v_heads.push(input_with_pe.t().dot(&grad_v_h));
        }

        let grad_w_q = stack(Axis(0), &grad_w_q_heads.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
        let grad_w_k = stack(Axis(0), &grad_w_k_heads.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
        let grad_w_v = stack(Axis(0), &grad_w_v_heads.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();

        // println!("grad_w1.shape: {:?}", grad_w1.shape());
        // println!("grad_w2.shape: {:?}", grad_w2.shape());

        sgd_update(&mut w1, &grad_w1, lr);
        sgd_update(&mut w1, &grad_w1, lr);
        sgd_update(&mut w2, &grad_w2, lr);
        sgd_update_3dim(&mut w_q, &grad_w_q, lr);
        sgd_update_3dim(&mut w_k, &grad_w_k, lr);
        sgd_update_3dim(&mut w_v, &grad_w_v, lr);
        sgd_update(&mut w_o, &grad_w_o, lr);

        println!("Epoch: {}, Loss: {:.6}", epoch, loss);
    }

    // println!("Final Transformer Output:\n{:?}", output);
}
