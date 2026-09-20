use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize)]
pub struct DacConfig {
    pub codebook_dim: usize,
    pub codebook_loss_weight: f64,
    pub codebook_size: usize,
    pub commitment_loss_weight: f64,
    pub decoder_hidden_size: usize,
    pub downsampling_ratios: Vec<usize>,
    pub encoder_hidden_size: usize,
    pub hidden_size: usize,
    pub hop_length: usize,
    pub model_type: String,
    pub n_codebooks: usize,
    pub quantizer_dropout: usize,
    pub sampling_rate: usize,
    pub upsampling_ratios: Vec<usize>,
}

fn get_config() -> DacConfig {
    serde_json::from_value::<DacConfig>(json!(
          {
      "codebook_dim": 8,
      "codebook_loss_weight": 1.0,
      "codebook_size": 1024,
      "commitment_loss_weight": 0.25,
      "decoder_hidden_size": 1024,
      "downsampling_ratios": [
        8,
        5,
        4,
        2,
        3
      ],
      "encoder_hidden_size": 64,
      "hidden_size": 256,
      "hop_length": 960,
      "model_type": "dac",
      "n_codebooks": 9,
      "quantizer_dropout": 0,
      "sampling_rate": 16000,
      "upsampling_ratios": [
        8,
        5,
        4,
        2,
        3
      ]
    }
      ))
    .unwrap()
}
