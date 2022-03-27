use crate::matrix;
use crate::nn::activations::param_relu;
use crate::nn::layers::conv2d;
use crate::nn::layers::fully_connected;
use std::fs;

#[derive(Clone)]
pub struct TicTacToe {
    pub conv1: conv2d::Conv2D<param_relu::ParamRelu>,
    pub conv2: conv2d::Conv2D<param_relu::ParamRelu>,
    pub conv3: conv2d::Conv2D<param_relu::ParamRelu>,
    pub fc1: fully_connected::FullyConnected<param_relu::ParamRelu>,
    pub fc2: fully_connected::FullyConnected<param_relu::ParamRelu>,
}

pub fn new() -> TicTacToe {
    TicTacToe {
        conv1: conv2d::new(3, 64, (3, 3), param_relu::new(1.0, 0.001)),
        conv2: conv2d::new(64, 64, (3, 3), param_relu::new(1.0, 0.001)),
        conv3: conv2d::new(64, 64, (3, 3), param_relu::new(1.0, 0.001)),
        fc1: fully_connected::new(576, 100, param_relu::new(0.25, 0.001)),
        fc2: fully_connected::new(100, 9, param_relu::new(0.25, 0.001)),
    }
}

pub fn new_gaussian_noise() -> TicTacToe {
    TicTacToe {
        conv1: conv2d::new_gaussian_noise(3, 64, (3, 3), param_relu::new(1.0, 0.001)),
        conv2: conv2d::new_gaussian_noise(64, 64, (3, 3), param_relu::new(1.0, 0.001)),
        conv3: conv2d::new_gaussian_noise(64, 64, (3, 3), param_relu::new(1.0, 0.001)),
        fc1: fully_connected::new_gaussian_noise(576, 100, param_relu::new(0.25, 0.001)),
        fc2: fully_connected::new_gaussian_noise(100, 9, param_relu::new(0.25, 0.001)),
    }
}

pub fn feedforward(ttt: &TicTacToe, input: &[matrix::Matrix]) -> matrix::Matrix {
    // conv1
    let mut conv_output = conv2d::feedforward(&ttt.conv1, input, (1, 1), (1, 1, 1, 1));

    // conv2
    conv_output = conv2d::feedforward(&ttt.conv2, &conv_output, (1, 1), (1, 1, 1, 1));

    // conv3
    conv_output = conv2d::feedforward(&ttt.conv3, &conv_output, (1, 1), (1, 1, 1, 1));

    let mut output = matrix::new(1, 576);
    output.value = vec![];

    // flatten matrix
    for conv_output_matrix in conv_output.iter() {
        for val in conv_output_matrix.value.iter() {
            output.value.push(*val);
        }
    }

    assert!(output.value.len() == 576);

    output = fully_connected::feedforward(&ttt.fc1, &output);
    output = fully_connected::feedforward(&ttt.fc2, &output);

    output
}

pub fn add(a: &TicTacToe, b: &TicTacToe) -> TicTacToe {
    let mut c = a.clone();

    c.conv1 = conv2d::add(&c.conv1, &b.conv1);
    c.conv2 = conv2d::add(&c.conv2, &b.conv2);
    c.conv3 = conv2d::add(&c.conv3, &b.conv3);
    c.fc1 = fully_connected::add(&c.fc1, &b.fc1);
    c.fc2 = fully_connected::add(&c.fc2, &b.fc2);

    c
}

pub fn scalar(a: &TicTacToe, s: f32) -> TicTacToe {
    let mut b = a.clone();

    b.conv1 = conv2d::scalar(&b.conv1, s);
    b.conv2 = conv2d::scalar(&b.conv2, s);
    b.conv3 = conv2d::scalar(&b.conv3, s);
    b.fc1 = fully_connected::scalar(&b.fc1, s);
    b.fc2 = fully_connected::scalar(&b.fc2, s);

    b
}

pub fn save(a: &TicTacToe, dir_name: &str) {
    fs::create_dir_all(dir_name).unwrap();
    conv2d::save(&a.conv1, (dir_name.to_owned() + "/conv1").as_str());
    conv2d::save(&a.conv2, (dir_name.to_owned() + "/conv2").as_str());
    conv2d::save(&a.conv3, (dir_name.to_owned() + "/conv3").as_str());
    fully_connected::save(&a.fc1, (dir_name.to_owned() + "/fc1").as_str());
    fully_connected::save(&a.fc2, (dir_name.to_owned() + "/fc2").as_str());
}

pub fn load(dir_name: &str) -> TicTacToe {
    let mut a = new();
    a.conv1 = conv2d::load(&a.conv1, (dir_name.to_owned() + "/conv1").as_str());
    a.conv2 = conv2d::load(&a.conv2, (dir_name.to_owned() + "/conv2").as_str());
    a.conv3 = conv2d::load(&a.conv3, (dir_name.to_owned() + "/conv3").as_str());
    a.fc1 = fully_connected::load(&a.fc1, (dir_name.to_owned() + "/fc1").as_str());
    a.fc2 = fully_connected::load(&a.fc2, (dir_name.to_owned() + "/fc2").as_str());

    a
}
