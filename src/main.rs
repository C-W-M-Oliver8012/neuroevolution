pub mod matrix;
pub mod models;
pub mod nn;

use crate::nn::activations;
use crate::nn::layers::conv2d;
use std::time::Instant;

fn main() {
    let mut input: Vec<matrix::Matrix> = Vec::new();

    for _ in 0..3 {
        input.push(matrix::new(3, 3));
    }

    let conv1 = conv2d::new_gaussian_noise(3, 32, (3, 3));
    let conv2 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv3 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv4 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv5 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv6 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv7 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv8 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv9 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv10 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv11 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv12 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv13 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv14 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv15 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv16 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv17 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv18 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv19 = conv2d::new_gaussian_noise(32, 32, (3, 3));
    let conv20 = conv2d::new_gaussian_noise(32, 32, (3, 3));

    let start = Instant::now();

    let mut output = conv2d::feedforward(&conv1, &input, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv2, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv3, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv4, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv5, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv6, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv7, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv8, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv9, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv10, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv11, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv12, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv13, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv14, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv15, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv16, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv17, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv18, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv19, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    output = conv2d::feedforward(&conv20, &output, (1, 1), (1, 1, 1, 1));
    for output_matrix in output.iter_mut() {
        *output_matrix = activations::parameterized_relu(output_matrix, 0.15, 0.001);
    }

    println!("{}", start.elapsed().as_micros() as f64 / 1000000.0);

    for output_matrix in output.iter_mut() {
        matrix::print(output_matrix);
    }
}
