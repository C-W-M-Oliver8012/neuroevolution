pub mod matrix;
pub mod models;
pub mod nn;

//use crate::nn::activations;
use crate::nn::layers::conv2d;
//use std::time::Instant;

fn main() {
    /*
    let mut input: Vec<matrix::Matrix> = Vec::new();

    for _ in 0..3 {
        input.push(matrix::new_gaussian_noise(12, 12));
    }

    let conv1 = conv2d::new_gaussian_noise(3, 32, 3, 3);
    let conv2 = conv2d::new_gaussian_noise(32, 32, 3, 3);
    let conv3 = conv2d::new_gaussian_noise(32, 32, 3, 3);
    let conv4 = conv2d::new_gaussian_noise(32, 32, 3, 3);

    let start = Instant::now();

    let mut output = conv2d::feedforward(&conv1, &input, (1, 1), (0, 0, 0, 0));
    for i in 0..output.len() {
        output[i] = activations::parameterized_relu(&output[i], 0.2, 0.001);
    }

    output = conv2d::feedforward(&conv2, &output, (1, 1), (0, 0, 0, 0));
    for i in 0..output.len() {
        output[i] = activations::parameterized_relu(&output[i], 0.2, 0.001);
    }

    output = conv2d::feedforward(&conv3, &output, (1, 1), (0, 0, 0, 0));
    for i in 0..output.len() {
        output[i] = activations::parameterized_relu(&output[i], 0.2, 0.001);
    }

    output = conv2d::feedforward(&conv4, &output, (1, 1), (0, 0, 0, 0));
    for i in 0..output.len() {
        output[i] = activations::parameterized_relu(&output[i], 0.2, 0.001);
    }

    println!("{}", start.elapsed().as_micros() as f64 / 1000000.0);
    
    for i in 0..output.len() {
        matrix::print(&output[i]);
    }
    */

    let mut input = vec![matrix::new(3, 3), matrix::new(3, 3)];
    input[0].value = vec![3.0, 2.0, 1.0, 9.0, 8.0, 4.0, 0.0, 1.0, 8.0];
    input[1].value = vec![3.0, 2.0, 1.0, 9.0, 8.0, 4.0, 0.0, 1.0, 8.0];
    matrix::print(&input[0]);

    let filter_size: (usize, usize) = (2, 2);
    let stride_size: (usize, usize) = (1, 1);

    let window_size = conv2d::get_window_size((3, 3), filter_size, stride_size, (0, 0, 0, 0));
    let a = conv2d::im2col(&input, window_size, filter_size, 2, stride_size, (0, 0));
    matrix::print(&a);

    let window_size = conv2d::get_window_size((3, 3), filter_size, stride_size, (1, 1, 1, 1));
    let a = conv2d::im2col(&input, window_size, filter_size, 2, stride_size, (1, 1));
    matrix::print(&a);
}