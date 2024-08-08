use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{SafeTensors, View};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        // let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| {
        // ...
        // };

        // LLamaParams {
        //     embedding_table: get_tensor(...),
        //     ...
        // }

        let get_tensor= |name: &str| {
            match safetensor.tensor(name) {
                Ok(tv) => {
                    let div = std::mem::size_of::<f32>() / std::mem::size_of::<u8>();
                    let float_slice: &[f32] = unsafe {
                        std::slice::from_raw_parts(tv.data().as_ptr() as *const f32, tv.data_len() / div)
                    };
                    Tensor::new(float_slice.to_vec(), &tv.shape().to_vec())
                },
                Err(_) => {
                    Tensor::<f32>::default(&Vec::new())
                }
            }
        };

        let nr_layer = config.num_hidden_layers;
        Self {
            embedding_table: get_tensor("lm_head.weight"),

            rms_att_w: (0..nr_layer).map(|x| {
                get_tensor(&format!("model.layers.{x}.input_layernorm.weight"))
            }).collect(),
            wq: (0..nr_layer).map(|x| {
                get_tensor(&format!("model.layers.{x}.self_attn.q_proj.weight"))
            }).collect(),
            wk: (0..nr_layer).map(|x| {
                get_tensor(&format!("model.layers.{x}.self_attn.k_proj.weight"))
            }).collect(),
            wv: (0..nr_layer).map(|x| {
                get_tensor(&format!("model.layers.{x}.self_attn.v_proj.weight"))
            }).collect(),
            wo: (0..nr_layer).map(|x| {
                get_tensor(&format!("model.layers.{x}.self_attn.o_proj.weight"))
            }).collect(),

            rms_ffn_w: (0..nr_layer).map(|x| {
                get_tensor(&format!("model.layers.{x}.post_attention_layernorm.weight"))
            }).collect(),
            w_up: (0..nr_layer).map(|x| {
                get_tensor(&format!("model.layers.{x}.mlp.up_proj.weight"))
            }).collect(),
            w_gate: (0..nr_layer).map(|x| {
                get_tensor(&format!("model.layers.{x}.mlp.gate_proj.weight"))
            }).collect(),
            w_down: (0..nr_layer).map(|x| {
                get_tensor(&format!("model.layers.{x}.mlp.down_proj.weight"))
            }).collect(),

            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
