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
    blocks::{DacDecoderBlock, DacDecoderBlockConfig, Snake1d, Snake1dConfig},
    config::DacConfig,
};

// Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py#L477
// DAC implemented in HiggsAudioV2Tokenizer is slightly different from the HF version
// No tanh activation in the final decoder output

#[derive(Module, Debug)]
pub struct DacDecoder<B: Backend> {
    conv1: Conv1d<B>,
    block: Vec<DacDecoderBlock<B>>,
    snake1: Snake1d<B>,
    conv2: Conv1d<B>,
}

impl<B: Backend> DacDecoder<B> {
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
pub struct DacDecoderConfig {}

impl DacDecoderConfig {
    pub fn init<B: Backend>(&self, config: &DacConfig, device: &B::Device) -> DacDecoder<B> {
        let input_channel = config.hidden_size;
        let channels = config.decoder_hidden_size;

        let mut blocks = vec![];
        for (stride_index, stride) in config.upsampling_ratios.iter().enumerate() {
            blocks.push(DacDecoderBlockConfig::new(*stride, stride_index).init(config, device));
        }

        let output_dim =
            config.decoder_hidden_size / 2_usize.pow(config.upsampling_ratios.len() as u32);
        DacDecoder {
            conv1: Conv1dConfig::new(input_channel, channels, 7)
                .with_padding(PaddingConfig1d::Explicit(3, 3))
                .init(device),
            block: blocks,
            snake1: Snake1dConfig::new(output_dim).init(device),
            conv2: Conv1dConfig::new(output_dim, 1, 7)
                .with_padding(PaddingConfig1d::Explicit(3, 3))
                .init(device),
        }
    }
}
