use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{
        PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig},
    },
    tensor::{backend::Backend, s},
};

use crate::dac::config::DacConfig;

#[derive(Module, Debug)]
pub struct Snake1d<B: Backend> {
    alpha: Param<Tensor<B, 3>>,
}

impl<B: Backend> Snake1d<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        // hidden_states: (B, C, T)
        let alpha = self.alpha.val();
        let alpha_recip = (alpha.clone() + 1e-9).recip();
        let sin_sq = (alpha * hidden_states.clone()).sin().powf_scalar(2.0);
        hidden_states + (alpha_recip * sin_sq)
    }
}

#[derive(Config, Debug)]
pub struct Snake1dConfig {
    hidden_dim: usize,
}

impl Snake1dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Snake1d<B> {
        Snake1d {
            alpha: Param::from_tensor(Tensor::ones([1, self.hidden_dim, 1], device)),
        }
    }
}

#[derive(Module, Debug)]
pub struct DacResidualUnit<B: Backend> {
    snake1: Snake1d<B>,
    conv1: Conv1d<B>,
    snake2: Snake1d<B>,
    conv2: Conv1d<B>,
}

impl<B: Backend> DacResidualUnit<B> {
    pub fn forward(&self, mut hidden_state: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output_tensor = hidden_state.clone();
        output_tensor = self.conv1.forward(self.snake1.forward(output_tensor));
        output_tensor = self.conv2.forward(self.snake2.forward(output_tensor));

        let padding = ((hidden_state.dims()[2] - output_tensor.dims()[2]) / 2) as i32;
        if padding > 0 {
            hidden_state = hidden_state.slice(s![.., .., padding..-padding]);
        }
        hidden_state + output_tensor
    }
}

#[derive(Config, Debug)]
pub struct DacResidualUnitConfig {
    dimension: usize,
    dilation: usize,
}

impl DacResidualUnitConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DacResidualUnit<B> {
        let pad = ((7 - 1) * self.dilation) / 2;
        DacResidualUnit {
            snake1: Snake1dConfig::new(self.dimension).init(device),
            conv1: Conv1dConfig::new(self.dimension, self.dimension, 7)
                .with_dilation(self.dilation)
                .with_padding(PaddingConfig1d::Explicit(pad, pad))
                .init(device),
            snake2: Snake1dConfig::new(self.dimension).init(device),
            conv2: Conv1dConfig::new(self.dimension, self.dimension, 1).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct DacEncoderBlock<B: Backend> {
    res_unit1: DacResidualUnit<B>,
    res_unit2: DacResidualUnit<B>,
    res_unit3: DacResidualUnit<B>,
    snake1: Snake1d<B>,
    conv1: Conv1d<B>,
}

impl<B: Backend> DacEncoderBlock<B> {
    pub fn forward(&self, mut hidden_state: Tensor<B, 3>) -> Tensor<B, 3> {
        hidden_state = self.res_unit1.forward(hidden_state);
        hidden_state = self.res_unit2.forward(hidden_state);
        hidden_state = self.snake1.forward(self.res_unit3.forward(hidden_state));
        self.conv1.forward(hidden_state)
    }
}

#[derive(Config, Debug)]
pub struct DacEncoderBlockConfig {
    stride: usize,
    stride_index: usize,
}

impl DacEncoderBlockConfig {
    pub fn init<B: Backend>(&self, config: &DacConfig, device: &B::Device) -> DacEncoderBlock<B> {
        let dimension = config.encoder_hidden_size * 2_usize.pow(self.stride_index as u32);
        let padding = (self.stride as f32 / 2.0).ceil() as usize;
        DacEncoderBlock {
            res_unit1: DacResidualUnitConfig::new(dimension / 2, 1).init(device),
            res_unit2: DacResidualUnitConfig::new(dimension / 2, 3).init(device),
            res_unit3: DacResidualUnitConfig::new(dimension / 2, 9).init(device),
            snake1: Snake1dConfig::new(dimension / 2).init(device),
            conv1: Conv1dConfig::new(dimension / 2, dimension, 2 * self.stride)
                .with_stride(self.stride)
                .with_padding(PaddingConfig1d::Explicit(padding, padding))
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct DacDecoderBlock<B: Backend> {
    snake1: Snake1d<B>,
    conv_t1: ConvTranspose1d<B>,
    res_unit1: DacResidualUnit<B>,
    res_unit2: DacResidualUnit<B>,
    res_unit3: DacResidualUnit<B>,
}

impl<B: Backend> DacDecoderBlock<B> {
    pub fn forward(&self, mut hidden_state: Tensor<B, 3>) -> Tensor<B, 3> {
        hidden_state = self.snake1.forward(hidden_state);
        hidden_state = self.conv_t1.forward(hidden_state);
        hidden_state = self.res_unit1.forward(hidden_state);
        hidden_state = self.res_unit2.forward(hidden_state);
        self.res_unit3.forward(hidden_state)
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py#L477
// Need to change the output padding of ConvTranspose1D as HiggsAudioV2Tokenizer's DAC implementation is slightly different from the HF version
#[derive(Config, Debug)]
pub struct DacDecoderBlockConfig {
    stride: usize,
    stride_index: usize,
}

impl DacDecoderBlockConfig {
    pub fn init<B: Backend>(&self, config: &DacConfig, device: &B::Device) -> DacDecoderBlock<B> {
        let input_dim = config.decoder_hidden_size / 2_usize.pow(self.stride_index as u32);
        let output_dim = config.decoder_hidden_size / 2_usize.pow(self.stride_index as u32 + 1);
        let padding = (self.stride as f64 / 2.0).ceil() as usize;
        let out_padding = self.stride % 2;
        DacDecoderBlock {
            snake1: Snake1dConfig::new(input_dim).init(device),
            conv_t1: ConvTranspose1dConfig::new([input_dim, output_dim], 2 * self.stride)
                .with_stride(self.stride)
                .with_padding(padding)
                .with_padding_out(out_padding)
                .init(device),
            res_unit1: DacResidualUnitConfig::new(output_dim, 1).init(device),
            res_unit2: DacResidualUnitConfig::new(output_dim, 3).init(device),
            res_unit3: DacResidualUnitConfig::new(output_dim, 9).init(device),
        }
    }
}
