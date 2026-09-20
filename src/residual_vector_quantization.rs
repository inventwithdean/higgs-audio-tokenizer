use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::{Int, backend::Backend, module::embedding, s},
};

use crate::config::HiggsAudioV2TokenizerConfig;

#[derive(Module, Debug)]
pub struct HiggsAudioV2TokenizerEuclideanCodebook<B: Backend> {
    embed: Param<Tensor<B, 2>>,
}

impl<B: Backend> HiggsAudioV2TokenizerEuclideanCodebook<B> {
    pub fn quantize(&self, hidden_states: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        let embed = self.embed.val().t();
        let scaled_states = hidden_states.clone().powf_scalar(2.0).sum_dim(1);
        let dist = scaled_states - hidden_states.matmul(embed.clone()).mul_scalar(2.0)
            + embed.powf_scalar(2.0).sum_dim(0);
        // Return indices only
        (-dist).max_dim_with_indices(1).1
    }

    pub fn encode(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 2, Int> {
        let [b, t, c] = hidden_states.dims();
        let hidden_states = hidden_states.reshape([b * t, c]);
        let embed_ind = self.quantize(hidden_states);
        embed_ind.reshape([b, t]) // (b, t, 1) reshaped to (b, t)
    }

    pub fn decode(&self, embed_ind: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        embedding(self.embed.val(), embed_ind)
    }
}

#[derive(Config, Debug)]
pub struct HiggsAudioV2TokenizerEuclideanCodebookConfig {}

impl HiggsAudioV2TokenizerEuclideanCodebookConfig {
    pub fn init<B: Backend>(
        &self,
        config: &HiggsAudioV2TokenizerConfig,
        device: &B::Device,
    ) -> HiggsAudioV2TokenizerEuclideanCodebook<B> {
        HiggsAudioV2TokenizerEuclideanCodebook {
            embed: Param::from_tensor(Tensor::zeros(
                [config.codebook_size, config.codebook_dim],
                device,
            )),
        }
    }
}

#[derive(Module, Debug)]
pub struct HiggsAudioV2TokenizerVectorQuantization<B: Backend> {
    codebook: HiggsAudioV2TokenizerEuclideanCodebook<B>,
    project_in: Linear<B>,
    project_out: Linear<B>,
}

impl<B: Backend> HiggsAudioV2TokenizerVectorQuantization<B> {
    pub fn encode(&self, mut hidden_states: Tensor<B, 3>) -> Tensor<B, 2, Int> {
        hidden_states = hidden_states.transpose();
        hidden_states = self.project_in.forward(hidden_states);
        self.codebook.encode(hidden_states)
    }

    pub fn decode(&self, embed_ind: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut quantize = self.codebook.decode(embed_ind);
        quantize = self.project_out.forward(quantize);
        quantize.transpose()
    }
}

#[derive(Config, Debug)]
pub struct HiggsAudioV2TokenizerVectorQuantizationConfig {}

impl HiggsAudioV2TokenizerVectorQuantizationConfig {
    pub fn init<B: Backend>(
        &self,
        config: &HiggsAudioV2TokenizerConfig,
        device: &B::Device,
    ) -> HiggsAudioV2TokenizerVectorQuantization<B> {
        let hidden_size =
            config.acoustic_model_config.hidden_size + config.semantic_model_config.hidden_size;
        HiggsAudioV2TokenizerVectorQuantization {
            codebook: HiggsAudioV2TokenizerEuclideanCodebookConfig::new().init(config, device),
            project_in: LinearConfig::new(hidden_size, config.codebook_dim).init(device),
            project_out: LinearConfig::new(config.codebook_dim, hidden_size).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct HiggsAudioV2TokenizerResidualVectorQuantization<B: Backend> {
    quantizers: Vec<HiggsAudioV2TokenizerVectorQuantization<B>>,
}

impl<B: Backend> HiggsAudioV2TokenizerResidualVectorQuantization<B> {
    pub fn encode(&self, embeddings: Tensor<B, 3>) -> Tensor<B, 3, Int> {
        // Hardcoding here.
        let num_quantizers = 8;
        let mut residual = embeddings.clone();
        let mut all_indices = vec![];
        for quantizer in &self.quantizers[..num_quantizers] {
            let indices = quantizer.encode(residual.clone());
            let quantized = quantizer.decode(indices.clone());
            residual = residual - quantized;
            all_indices.push(indices);
        }

        // Stacks to shape (B, num_quantizers, T)
        Tensor::stack(all_indices, 1)
    }

    pub fn decode(&self, codes: Tensor<B, 3, Int>) -> Tensor<B, 3> {
        // codes: (B, num_quantizers, T)
        let [_b, num_quantizers, _t] = codes.dims();
        let mut quantized_out: Option<Tensor<B, 3>> = None;

        for i in 0..num_quantizers {
            let quantizer = &self.quantizers[i];
            let quantized =
                quantizer.decode(codes.clone().slice(s![.., i..i + 1, ..]).squeeze_dim(1));
            quantized_out = match quantized_out {
                Some(acc) => Some(acc + quantized),
                None => Some(quantized),
            };
        }

        quantized_out.expect("Codes tensor should have at least 1 quantizer dimension.")
    }
}

#[derive(Config, Debug)]
pub struct HiggsAudioV2TokenizerResidualVectorQuantizationConfig {}

impl HiggsAudioV2TokenizerResidualVectorQuantizationConfig {
    pub fn init<B: Backend>(
        &self,
        config: &HiggsAudioV2TokenizerConfig,
        device: &B::Device,
    ) -> HiggsAudioV2TokenizerResidualVectorQuantization<B> {
        let target_bandwidth = config
            .target_bandwidths
            .last()
            .expect("target_bandwidths should not be empty!")
            .to_owned();
        let hop_length: f64 = config
            .acoustic_model_config
            .downsampling_ratios
            .iter()
            .product::<usize>() as f64;
        let frame_rate = (config.sample_rate as f64 / hop_length).ceil() as usize;
        let codebook_nbits = (config.codebook_size as f64).log2().ceil() as usize;
        let num_quantizers = (1000.0 * target_bandwidth) as usize / (frame_rate * codebook_nbits);
        let mut quantizers = vec![];
        for _ in 0..num_quantizers {
            quantizers
                .push(HiggsAudioV2TokenizerVectorQuantizationConfig::new().init(config, device));
        }

        HiggsAudioV2TokenizerResidualVectorQuantization { quantizers }
    }
}
