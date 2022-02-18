#[cfg(test)]
use crate::matrix;
#[cfg(test)]
use crate::nn::activations;

#[test]
fn normalize_test() {
    let mut a = matrix::new(2, 3);
    a.value = vec![2.0, 4.0, -4.0, 8.0, -2.0, 6.0];

    let mean = matrix::mean(&a);
    let variance = matrix::variance(&a, mean);
    let std = variance.sqrt();
    let expected_output = [
        (a.value[0] - mean) / std,
        (a.value[1] - mean) / std,
        (a.value[2] - mean) / std,
        (a.value[3] - mean) / std,
        (a.value[4] - mean) / std,
        (a.value[5] - mean) / std,
    ];

    let b = activations::normalize(&a, 0.0, 1.0);
    assert_eq!(b.rows, 2);
    assert_eq!(b.columns, 3);
    assert_eq!(b.value, expected_output);
}

#[test]
fn leaky_relu_test() {
    let mut a = matrix::new(2, 3);
    a.value = vec![2.0, 4.0, -4.0, 8.0, -2.0, 6.0];

    let expected_output = [
        a.value[0],
        a.value[1],
        a.value[2] * 0.25,
        a.value[3],
        a.value[4] * 0.25,
        a.value[5],
    ];

    let b = activations::leaky_relu(&a, 0.25);
    assert_eq!(b.rows, 2);
    assert_eq!(b.columns, 3);
    assert_eq!(b.value, expected_output);
}
