use burn::{Tensor, backend::Wgpu, tensor::TensorData};
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use safetensors::SafeTensors;

use crate::hubert::{config::get_config, model::HubertModelConfig};
mod hubert;
mod dac;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = Default::default();
    let hubert_config = get_config();
    let mut hubert_model = HubertModelConfig::new().init::<MyBackend>(&hubert_config, &device);
    println!("{:?}", hubert_model);

    let remapper = KeyRemapper::new()
        .add_pattern(r"^semantic_model\.", "")
        .unwrap();

    let mut store = SafetensorsStore::from_file("higgs_audio_tokenizer.safetensors")
        .remap(remapper)
        .with_from_adapter(PyTorchToBurnAdapter);

    let result = hubert_model.load_from(&mut store).unwrap();
    println!("{}", result);

    println!("Applied: {} tensors", result.applied.len());
    println!("Missing {:?}", result.missing);
    println!("Errors: {:?}", result.errors);

    if result.is_success() {
        println!("All tensors loaded successfully!");
    }

    // Testing a input after passing it via AutoFeatureExtractor from Python
    // Save it using save_file from safetensors library. input_values are of shape (1, T)
    // {"input_values": input_values}

    let bytes = std::fs::read("test_input.safetensors").unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();

    let view = st.tensor("input_values").unwrap();

    let float_data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let shape = view.shape().to_vec();
    let data = TensorData::new(float_data, shape);
    let input_values: Tensor<MyBackend, 2> = Tensor::from_data(data, &device);
    let [batch_size, time] = input_values.dims();
    let input_values = input_values.reshape([batch_size, 1, time]);

    let hidden_states = hubert_model.forward(input_values);
    println!("Rust output Shape: {:?}", hidden_states.dims());

    // Outputs match yayy!!
    let first_10_values = hidden_states.slice([0..1, 0..1, 0..10]);
    println!("First 10 values: \n{}", first_10_values);
}
