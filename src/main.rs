pub mod matrix;
pub mod models;
pub mod nn;

use crate::nn::layers::conv2d;
use std::time::Instant;

fn main() {
    let mut input: Vec<matrix::Matrix> = Vec::new();

    for _ in 0..256 {
        input.push(matrix::new_gaussian_noise(14, 14));
    }

    let conv1 = conv2d::new_gaussian_noise(256, 256, 1, 2, 2);
    let conv2 = conv2d::new_gaussian_noise(256, 256, 1, 2, 2);
    let conv3 = conv2d::new_gaussian_noise(256, 256, 1, 2, 2);
    let conv4 = conv2d::new_gaussian_noise(256, 256, 1, 2, 2);
    let conv5 = conv2d::new_gaussian_noise(256, 256, 1, 2, 2);
    let conv6 = conv2d::new_gaussian_noise(256, 256, 1, 2, 2);
    //matrix::print(&conv2d.filters);
    //conv2d::print(&conv2d);

    let now = Instant::now();

    let mut _output = conv2d::feedforward(&conv1, &input);
    _output = conv2d::feedforward(&conv2, &_output);
    _output = conv2d::feedforward(&conv3, &_output);
    _output = conv2d::feedforward(&conv4, &_output);
    _output = conv2d::feedforward(&conv5, &_output);
    _output = conv2d::feedforward(&conv6, &_output);

    println!("{}, {}", _output[0].rows, _output[0].columns);

    println!("{}", now.elapsed().as_micros() as f64 / 1000000.0);
}
