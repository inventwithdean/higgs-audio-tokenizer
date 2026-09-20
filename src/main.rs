#![recursion_limit = "256"]

use burn::backend::Wgpu;
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::tokenizer::HiggsAudioV2TokenizerModelConfig;
mod config;
mod dac;
mod hubert;
mod residual_vector_quantization;
mod semantic_encoder;
mod tokenizer;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = Default::default();
    let model_config = config::get_config();

    let mut tokenizer =
        HiggsAudioV2TokenizerModelConfig::new().init::<MyBackend>(&model_config, &device);

    println!("{:?}", tokenizer);

    // let remapper = KeyRemapper::new();

    let mut store = SafetensorsStore::from_file("higgs_audio_tokenizer.safetensors")
        .with_from_adapter(PyTorchToBurnAdapter);

    let result = tokenizer.load_from(&mut store).unwrap();
    println!("{}", result);

    println!("Applied: {} tensors", result.applied.len());
    println!("Missing {:?}", result.missing);
    println!("Errors: {:?}", result.errors);

    if result.is_success() {
        println!("All tensors loaded successfully!");
    }
}
