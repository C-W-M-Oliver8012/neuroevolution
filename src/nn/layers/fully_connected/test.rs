#[cfg(test)]
use crate::matrix;
#[cfg(test)]
use crate::nn::layers::fully_connected;
#[cfg(test)]
use std::fs;

#[test]
fn new_test() {
    let a = fully_connected::new(2, 4);

    assert_eq!(a.weights.rows, 2);
    assert_eq!(a.weights.columns, 4);
    assert_eq!(a.bias.rows, 1);
    assert_eq!(a.bias.columns, 4);
    assert_eq!(a.weights.value, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(a.bias.value, [0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn new_gaussian_noise_test() {
    let a = fully_connected::new_gaussian_noise(2, 4);

    assert_eq!(a.weights.rows, 2);
    assert_eq!(a.weights.columns, 4);
    assert_eq!(a.bias.rows, 1);
    assert_eq!(a.bias.columns, 4);
    assert_ne!(a.weights.value, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_ne!(a.bias.value, [0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn print_test() {
    let a = fully_connected::new_gaussian_noise(2, 4);
    // pass = does not panic
    fully_connected::print(&a);
}

#[test]
fn feedforward_test() {
    let mut a = fully_connected::new(2, 4);
    a.weights.value = vec![2.0, 3.0, -4.0, 5.0, -7.0, 8.0, -1.0, 2.0];
    a.bias.value = vec![1.0, -1.0, -2.0, 1.0];

    let mut input = matrix::new(1, 2);
    input.value = vec![2.0, 1.0];

    let mut expected_output = matrix::multiply(&input, &a.weights).unwrap();
    expected_output = matrix::add(&expected_output, &a.bias).unwrap();

    let output = fully_connected::feedforward(&a, &input);

    assert_eq!(output.rows, 1);
    assert_eq!(output.columns, 4);
    assert_eq!(output.value, expected_output.value);
}

#[test]
fn add_test() {
    let mut a = fully_connected::new(2, 4);
    let mut b = fully_connected::new(2, 4);
    a.weights.value = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    a.bias.value = vec![1.0, 1.0, 1.0, 1.0];
    b.weights.value = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    b.bias.value = vec![1.0, 1.0, 1.0, 1.0];

    let c = fully_connected::add(&a, &b);
    assert_eq!(c.weights.value, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    assert_eq!(c.bias.value, [2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn scalar_test() {
    let mut a = fully_connected::new(2, 4);
    a.weights.value = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    a.bias.value = vec![1.0, 1.0, 1.0, 1.0];

    let b = fully_connected::scalar(&a, 2.0);
    assert_eq!(b.weights.value, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    assert_eq!(b.bias.value, [2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn save_load_test() {
    let mut a = fully_connected::new(2, 3);
    a.weights.value = vec![1.0, 3.0, 4.0, -5.0, 2.0, -9.0];
    a.bias.value = vec![1.0, 2.0, -4.0];

    fully_connected::save(&a, "fc");
    let b = fully_connected::load(&a, "fc");
    fs::remove_dir_all("fc").unwrap();

    assert_eq!(b.weights.rows, 2);
    assert_eq!(b.weights.columns, 3);
    assert_eq!(b.bias.rows, 1);
    assert_eq!(b.bias.columns, 3);
    assert_eq!(b.weights.value, [1.0, 3.0, 4.0, -5.0, 2.0, -9.0]);
    assert_eq!(b.bias.value, [1.0, 2.0, -4.0]);
}
