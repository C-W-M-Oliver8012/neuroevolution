pub mod matrix;
pub mod models;
pub mod nn;

use crate::nn::layers::conv2d;

fn main() {
    let conv2d = conv2d::new_gaussian_noise(3, 3, 1, 2, 2);
    matrix::print(&conv2d.filters);
    conv2d::print(&conv2d);
}
