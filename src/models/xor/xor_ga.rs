use crate::matrix;
use crate::models::xor;
use std::sync::mpsc;
use std::thread;

fn fitness(model: &xor::XorModel) -> (f32, bool) {
    let mut did_pass = false;
    let mut score: f32 = 0.0;

    // 0.0, 0.0 -> 0.0
    let mut input = matrix::new(1, 2);
    input.value = vec![0.0, 0.0];
    let mut output = xor::feedforward(model, &input);
    score += (output.value[0] - output.value[1]).min(1.0);

    // 1.0, 0.0 -> 1.0
    input.value = vec![1.0, 0.0];
    output = xor::feedforward(model, &input);
    score += (output.value[1] - output.value[0]).min(1.0);

    // 0.0, 1.0 -> 1.0
    input.value = vec![0.0, 1.0];
    output = xor::feedforward(model, &input);
    score += (output.value[1] - output.value[0]).min(1.0);

    // 1.0, 1.0 -> 1.0
    input.value = vec![1.0, 1.0];
    output = xor::feedforward(model, &input);
    score += (output.value[0] - output.value[1]).min(1.0);

    if score >= 4.0 {
        did_pass = true;
    }

    (score, did_pass)
}

pub fn print_outputs(model: &xor::XorModel) {
    let mut input = matrix::new(1, 2);

    input.value = vec![0.0, 0.0];
    let mut output = xor::feedforward(model, &input);
    println!("0.0, 0.0 => {}, {}", output.value[0], output.value[1]);

    input.value = vec![1.0, 0.0];
    output = xor::feedforward(model, &input);
    println!("1.0, 0.0 => {}, {}", output.value[0], output.value[1]);

    input.value = vec![0.0, 1.0];
    output = xor::feedforward(model, &input);
    println!("0.0, 1.0 => {}, {}", output.value[0], output.value[1]);

    input.value = vec![1.0, 1.0];
    output = xor::feedforward(model, &input);
    println!("1.0, 1.0 => {}, {}", output.value[0], output.value[1]);
}

pub fn train() {
    
}
