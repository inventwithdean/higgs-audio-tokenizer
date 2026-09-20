use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::backend::Backend,
};

use crate::dac::{
    blocks::{DacEncoderBlock, DacEncoderBlockConfig, Snake1d, Snake1dConfig},
    config::DacConfig,
};

#[derive(Module, Debug)]
pub struct DacEncoder<B: Backend> {
    conv1: Conv1d<B>,
    block: Vec<DacEncoderBlock<B>>,
    snake1: Snake1d<B>,
    conv2: Conv1d<B>,
}

impl<B: Backend> DacEncoder<B> {
    pub fn forward(&self, mut hidden_state: Tensor<B, 3>) -> Tensor<B, 3> {
        hidden_state = self.conv1.forward(hidden_state);
        for layer in &self.block {
            hidden_state = layer.forward(hidden_state);
        }
        hidden_state = self.snake1.forward(hidden_state);
        self.conv2.forward(hidden_state)
    }
}

#[derive(Config, Debug)]
pub struct DacEncoderConfig {}

impl DacEncoderConfig {
    pub fn init<B: Backend>(&self, config: &DacConfig, device: &B::Device) -> DacEncoder<B> {
        let d_model =
            config.encoder_hidden_size * 2_usize.pow(config.downsampling_ratios.len() as u32);
        let mut blocks = vec![];

        for (stride_index, stride) in config.downsampling_ratios.iter().enumerate() {
            blocks.push(DacEncoderBlockConfig::new(*stride, stride_index + 1).init(config, device));
        }

        DacEncoder {
            conv1: Conv1dConfig::new(1, config.encoder_hidden_size, 7)
                .with_padding(PaddingConfig1d::Explicit(3, 3))
                .init(device),
            block: blocks,
            snake1: Snake1dConfig::new(d_model).init(device),
            conv2: Conv1dConfig::new(d_model, config.hidden_size, 3)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .init(device),
        }
    }
}
