pub mod test;

use crate::matrix;
use crate::nn::activations::Activate;
use std::fs;

#[derive(Clone)]
pub struct FullyConnected<T: Activate> {
    pub weights: matrix::Matrix,
    pub bias: matrix::Matrix,
    pub activation: T,
}

pub fn new<T: Activate>(inputs: usize, outputs: usize, activation: T) -> FullyConnected<T> {
    FullyConnected {
        weights: matrix::new(inputs, outputs),
        bias: matrix::new(1, outputs),
        activation,
    }
}

pub fn new_gaussian_noise<T: Activate>(
    inputs: usize,
    outputs: usize,
    activation: T,
) -> FullyConnected<T> {
    FullyConnected {
        weights: matrix::new_gaussian_noise(inputs, outputs),
        bias: matrix::new_gaussian_noise(1, outputs),
        activation,
    }
}

pub fn print<T: Activate>(a: &FullyConnected<T>) {
    println!("Fully Connected Layer");
    println!("Weights");
    matrix::print(&a.weights);
    println!("Bias");
    matrix::print(&a.bias);
}

pub fn feedforward<T: Activate>(
    fully_connected: &FullyConnected<T>,
    input: &matrix::Matrix,
) -> matrix::Matrix {
    let mut output = matrix::multiply(input, &fully_connected.weights);
    output = matrix::add(&output, &fully_connected.bias);
    output = fully_connected.activation.activate(&output);

    output
}

pub fn add<T: Activate + Clone>(a: &FullyConnected<T>, b: &FullyConnected<T>) -> FullyConnected<T> {
    let mut c = a.clone();
    c.weights = matrix::add(&a.weights, &b.weights);
    c.bias = matrix::add(&a.bias, &b.bias);

    c
}

pub fn scalar<T: Activate + Clone>(a: &FullyConnected<T>, s: f32) -> FullyConnected<T> {
    let mut b = a.clone();
    b.weights = matrix::scalar(&b.weights, s);
    b.bias = matrix::scalar(&b.bias, s);

    b
}

pub fn save<T: Activate>(a: &FullyConnected<T>, dir_name: &str) {
    fs::create_dir_all(dir_name).unwrap();
    matrix::save(&a.weights, (dir_name.to_owned() + "/weights.bin").as_str());
    matrix::save(&a.bias, (dir_name.to_owned() + "/bias.bin").as_str());
}

pub fn load<T: Activate + Clone>(a: &FullyConnected<T>, dir_name: &str) -> FullyConnected<T> {
    let mut b = a.clone();
    b.weights = matrix::load(&b.weights, (dir_name.to_owned() + "/weights.bin").as_str());
    b.bias = matrix::load(&b.bias, (dir_name.to_owned() + "/bias.bin").as_str());

    b
}
