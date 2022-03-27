pub mod matrix;
pub mod models;
pub mod nn;

use crate::models::tic_tac_toe;
use std::time::Instant;

fn main() {
    let mut input: Vec<matrix::Matrix> = Vec::new();
    for _ in 0..3 {
        input.push(matrix::new_gaussian_noise(3, 3));
    }

    let mut ttt = tic_tac_toe::new_gaussian_noise();

    let now = Instant::now();
    let mut output = tic_tac_toe::feedforward(&ttt, &input);
    println!("{}", now.elapsed().as_micros() as f64 / 1000000.0);

    matrix::print(&output);

    let mut a = tic_tac_toe::new_gaussian_noise();
    a = tic_tac_toe::scalar(&a, 0.1);
    ttt = tic_tac_toe::add(&ttt, &a);

    output = tic_tac_toe::feedforward(&ttt, &input);

    matrix::print(&output);

    println!("{}", now.elapsed().as_micros() as f64 / 1000000.0);
}
