pub mod matrix;
pub mod models;
pub mod nn;

use crate::nn::activations;
use crate::nn::layers::conv2d;
use std::time::Instant;

fn main() {
    let mut input: Vec<matrix::Matrix> = Vec::new();

    for _ in 0..3 {
        input.push(matrix::new_gaussian_noise(3, 3));
    }

    let conv1 = conv2d::new_gaussian_noise(3, 32, 3, 3);
    let conv2 = conv2d::new_gaussian_noise(32, 32, 3, 3);
    let conv3 = conv2d::new_gaussian_noise(32, 32, 3, 3);
    let conv4 = conv2d::new_gaussian_noise(32, 32, 3, 3);

    let start = Instant::now();

    let mut output = conv2d::feedforward(&conv1, &input, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.25, 0.001);
    }

    output = conv2d::feedforward(&conv2, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.25, 0.001);
    }

    output = conv2d::feedforward(&conv3, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.25, 0.001);
    }

    output = conv2d::feedforward(&conv4, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.25, 0.001);
    }

    println!("{}", start.elapsed().as_micros() as f64 / 1000000.0);

    for output_matrix in output.iter_mut() {
        matrix::print(output_matrix);
    }
}
