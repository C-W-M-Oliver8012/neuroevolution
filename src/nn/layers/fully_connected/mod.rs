pub mod test;

use crate::matrix;
use std::fs;

#[derive(Clone)]
pub struct FullyConnected {
    pub weights: matrix::Matrix,
    pub bias: matrix::Matrix,
}

pub fn new(inputs: usize, outputs: usize) -> FullyConnected {
    FullyConnected {
        weights: matrix::new(inputs, outputs),
        bias: matrix::new(1, outputs),
    }
}

pub fn new_gaussian_noise(inputs: usize, outputs: usize) -> FullyConnected {
    FullyConnected {
        weights: matrix::new_gaussian_noise(inputs, outputs),
        bias: matrix::new_gaussian_noise(1, outputs),
    }
}

pub fn print(a: &FullyConnected) {
    println!("Fully Connected Layer");
    println!("Weights");
    matrix::print(&a.weights);
    println!("Bias");
    matrix::print(&a.bias);
}

pub fn feedforward(fully_connected: &FullyConnected, input: &matrix::Matrix) -> matrix::Matrix {
    let mut output = matrix::multiply(input, &fully_connected.weights).unwrap();
    output = matrix::add(&output, &fully_connected.bias).unwrap();
    output
}

pub fn add(a: &FullyConnected, b: &FullyConnected) -> FullyConnected {
    let mut c = a.clone();
    c.weights = matrix::add(&a.weights, &b.weights).unwrap();
    c.bias = matrix::add(&a.bias, &b.bias).unwrap();
    c
}

pub fn scalar(a: &FullyConnected, s: f32) -> FullyConnected {
    let mut b = a.clone();
    b.weights = matrix::scalar(&b.weights, s);
    b.bias = matrix::scalar(&b.bias, s);
    b
}

pub fn save(a: &FullyConnected, dir_name: &str) {
    fs::create_dir_all(dir_name).unwrap();
    matrix::save(&a.weights, (dir_name.to_owned() + "/weights.bin").as_str());
    matrix::save(&a.bias, (dir_name.to_owned() + "/bias.bin").as_str());
}

pub fn load(a: &FullyConnected, dir_name: &str) -> FullyConnected {
    let mut b = a.clone();
    b.weights = matrix::load(&b.weights, (dir_name.to_owned() + "/weights.bin").as_str());
    b.bias = matrix::load(&b.bias, (dir_name.to_owned() + "/bias.bin").as_str());
    b
}
