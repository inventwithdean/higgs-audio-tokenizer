use burn::{Tensor, backend::Wgpu};
mod hubert;

fn main() {
    type B = Wgpu<f32, i32>;
    let device = Default::default();
    let tensor = Tensor::<B, 2>::ones([32, 512], &device);
    println!("Hello, world!");
}
