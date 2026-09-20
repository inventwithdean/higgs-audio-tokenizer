// Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/hubert/modular_hubert.py
use burn::{Tensor, config::Config, module::Module, tensor::backend::Backend};

use crate::hubert::{
    config::HubertConfig,
    wav2vec2::{
        Wav2Vec2Encoder, Wav2Vec2EncoderConfig, Wav2Vec2FeatureEncoder,
        Wav2Vec2FeatureEncoderConfig, Wav2Vec2FeatureProjection, Wav2Vec2FeatureProjectionConfig,
    },
};

#[derive(Module, Debug)]
pub struct HubertModel<B: Backend> {
    feature_extractor: Wav2Vec2FeatureEncoder<B>,
    feature_projection: Wav2Vec2FeatureProjection<B>,
    encoder: Wav2Vec2Encoder<B>,
}

impl<B: Backend> HubertModel<B> {
    pub fn forward(&self, input_values: Tensor<B, 3>) -> Vec<Tensor<B, 3>> {
        // input_values: (B, C, T)
        let mut extract_features = self.feature_extractor.forward(input_values);
        extract_features = extract_features.transpose(); // (B, T, C)
        let hidden_states = self.feature_projection.forward(extract_features);
        self.encoder.forward(hidden_states)
    }
}

#[derive(Config, Debug)]
pub struct HubertModelConfig {}

impl HubertModelConfig {
    pub fn init<B: Backend>(&self, config: &HubertConfig, device: &B::Device) -> HubertModel<B> {
        HubertModel {
            feature_extractor: Wav2Vec2FeatureEncoderConfig::new().init(config, device),
            feature_projection: Wav2Vec2FeatureProjectionConfig::new().init(config, device),
            encoder: Wav2Vec2EncoderConfig::new().init(config, device),
        }
    }
}
