#[cfg(test)]
use crate::matrix;
#[cfg(test)]
use crate::nn::activations::param_relu;
#[cfg(test)]
use crate::nn::activations::Activate;

#[test]
fn param_relu_test() {
    let pr = param_relu::new(0.5, 0.25);

    let mut a = matrix::new(2, 3);
    a.value = vec![2.0, 4.0, -4.0, 8.0, -2.0, 6.0];

    let expected_output = [
        a.value[0] * 0.5,
        a.value[1] * 0.5,
        a.value[2] * 0.25,
        a.value[3] * 0.5,
        a.value[4] * 0.25,
        a.value[5] * 0.5,
    ];

    let b = pr.activate(&a);
    assert_eq!(b.rows, 2);
    assert_eq!(b.columns, 3);
    assert_eq!(b.value, expected_output);
}
