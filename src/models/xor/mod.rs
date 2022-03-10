use crate::matrix;
use crate::nn::activations;
use crate::nn::layers::fully_connected;
use std::fs;

#[derive(Clone)]
pub struct XorModel {
    pub fc1: fully_connected::FullyConnected,
    pub fc2: fully_connected::FullyConnected,
}

pub fn new() -> XorModel {
    XorModel {
        fc1: fully_connected::new(2, 10),
        fc2: fully_connected::new(10, 2),
    }
}

pub fn new_gaussian_noise() -> XorModel {
    XorModel {
        fc1: fully_connected::new_gaussian_noise(2, 10),
        fc2: fully_connected::new_gaussian_noise(10, 2),
    }
}

pub fn print(a: &XorModel) {
    println!("XorModel");
    fully_connected::print(&a.fc1);
    fully_connected::print(&a.fc2);
}

pub fn feedforward(a: &XorModel, input: &matrix::Matrix) -> matrix::Matrix {
    let mut output = fully_connected::feedforward(&a.fc1, input);
    output = activations::parameterized_relu(&output, 0.25);

    output = fully_connected::feedforward(&a.fc2, &output);
    output = activations::parameterized_relu(&output, 0.25);
    output
}

pub fn add(a: &XorModel, b: &XorModel) -> XorModel {
    let mut c = a.clone();
    c.fc1 = fully_connected::add(&a.fc1, &b.fc1);
    c.fc2 = fully_connected::add(&a.fc2, &b.fc2);
    c
}

pub fn scalar(a: &XorModel, s: f32) -> XorModel {
    let mut c = a.clone();
    c.fc1 = fully_connected::scalar(&a.fc1, s);
    c.fc2 = fully_connected::scalar(&a.fc2, s);
    c
}

pub fn save(a: &XorModel, dir_name: &str) {
    fs::create_dir_all(dir_name).unwrap();
    fully_connected::save(&a.fc1, (dir_name.to_owned() + "/fc1").as_str());
    fully_connected::save(&a.fc2, (dir_name.to_owned() + "/fc2").as_str());
}

pub fn load(dir_name: &str) -> XorModel {
    let mut a = new();
    a.fc1 = fully_connected::load(&a.fc1, (dir_name.to_owned() + "/fc1").as_str());
    a.fc2 = fully_connected::load(&a.fc2, (dir_name.to_owned() + "/fc2").as_str());
    a
}
