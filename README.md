# Higgs Audio V2 Tokenizer

A pure Rust implementation of the **Higgs Audio V2 Tokenizer** neural audio codec, built using the [Burn](https://burn.dev/) deep learning framework. This project provides a fully functional pipeline to encode raw audio into discrete tokens (Residual Vector Quantization) and decode them back into high-fidelity audio waveforms.

## Overview

This repository ports the Boson AI's [`HiggsAudioV2Tokenizer`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py) architecture to Rust. It leverages Burn's modular neural network APIs and hardware-agnostic backends (using WGPU for GPU acceleration by default) to perform robust audio compression and reconstruction.

## Speed Comparison (end-to-end)

| Burn (wgpu) | PyTorch CUDA | PyTorch CPU
| --- | --- | --- |
| ~496ms | ~663ms | ~7640ms

> Tested on RTX 4060Ti 8G

#### Rust's binary takes virtually no time to start, while Python has to load heavy PyTorch binaries. So let's not compare cold starts.


## Features

* **End-to-End Pipeline:** Full encoding (audio -> tokens) and decoding (tokens -> audio) capabilities.
* **Hardware Agnostic:** Powered by Burn's `Wgpu` backend, allowing the model to run efficiently on any supported GPU.
* **Safetensors Integration:** Directly loads pre-trained PyTorch weights via `.safetensors` utilizing `PyTorchToBurnAdapter`.
* **Native Audio Resampling:** Built-in audio resampling using the `rubato` crate to match the model's required 24kHz sample rate.
* **Complex Model Architecture:** Accurately implements Descript Audio Codec (DAC), HuBERT (Wav2Vec2), and Residual Vector Quantization (RVQ).

## Architecture Breakdown

The `HiggsAudioV2TokenizerModel` consists of several interconnected modules:

* **Acoustic Encoder/Decoder (DAC):** Handles the high-fidelity representation of the audio waveform. Implements custom Snake1d activations and 1D convolutions.

* **Semantic Model (HuBERT):** Extracts high-level semantic meaning from the audio. Operates at a downsampled 16kHz rate and utilizes Wav2Vec2 components (Convolutions, Self-Attention, Feed-Forward).

* **Residual Vector Quantization (RVQ):** Quantizes the fused acoustic and semantic embeddings into discrete codebook indices (tokens) for extreme compression.

* **Feature Fusion:** Projects and combines both acoustic and semantic embeddings into a unified bottleneck.



## Prerequisites

* **Rust Toolchain:** Latest stable Rust compiler and `cargo`.
* **Pre-trained Weights:** A valid [`model.safetensors`](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/blob/main/model.safetensors) file containing the Higgs Audio V2 Tokenizer weights placed in the root directory.
* **Test Audio:** A valid `test.wav` file placed in the root directory.


## Usage

1. **Clone the repository:**
```
git clone https://github.com/inventwithdean/higgs-audio-tokenizer.git
cd higgs-audio-tokenizer
```

2. **Ensure assets are present:**
Place your `test.wav` and `model.safetensors` in the project root.

3. **Run the application:**
```
cargo run --release
```


### Expected Output

The application will:

1. Initialize the WGPU backend and load the model configuration.

2. Stream weights from `model.safetensors` into the Burn module.

3. Read `test.wav`, format it, and resample the audio to the 24,000 Hz base rate expected by the acoustic model.

4. Run a forward pass of the `encode` method to extract discrete tokens.

5. Run a forward pass of the `decode` method to reconstruct the audio.

6. Save the result to `reconstructed.wav` in a 32-bit float format.

TODO: Convert this to a usable crate

## License

Code distributed under the MIT License.