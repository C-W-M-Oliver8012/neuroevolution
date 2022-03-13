#[cfg(test)]
use crate::matrix;
#[cfg(test)]
use std::fs;

#[test]
fn new_test() {
    let a = matrix::new(2, 3);

    assert_eq!(a.rows, 2);
    assert_eq!(a.columns, 3);
    assert_eq!(a.value, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn new_gaussian_noise_test() {
    let a = matrix::new_gaussian_noise(2, 3);

    assert_eq!(a.rows, 2);
    assert_eq!(a.columns, 3);
    assert_ne!(a.value, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn print_test() {
    let a = matrix::new(2, 2);
    // pass = does not panic
    matrix::print(&a);
}

#[test]
fn multiply_test() {
    let mut a = matrix::new(2, 3);
    let mut b = matrix::new(3, 3);
    a.value = vec![4.0, 6.0, 1.0, 9.0, 7.0, 3.0];
    b.value = vec![2.0, 8.0, 3.0, 3.0, 2.0, 8.0, 7.0, 2.0, 9.0];

    let c = matrix::multiply(&a, &b);
    assert_eq!(c.rows, 2);
    assert_eq!(c.columns, 3);
    assert_eq!(c.value, [37.0, 93.0, 70.0, 60.0, 93.0, 87.0]);
}

#[test]
#[should_panic]
fn multiply_panic_test() {
    let mut a = matrix::new(2, 3);
    let mut b = matrix::new(3, 3);
    a.value = vec![4.0, 6.0, 1.0, 9.0, 7.0, 3.0];
    b.value = vec![2.0, 8.0, 3.0, 3.0, 2.0, 8.0, 7.0, 2.0, 9.0];
    let _ = matrix::multiply(&b, &a);
}

#[test]
fn add_test() {
    let mut a = matrix::new(2, 3);
    let mut b = matrix::new(2, 3);
    a.value = vec![4.0, 6.0, 1.0, 9.0, 7.0, 3.0];
    b.value = vec![2.0, 8.0, 3.0, 3.0, 2.0, 8.0];

    let c = matrix::add(&a, &b);
    assert_eq!(c.rows, 2);
    assert_eq!(c.columns, 3);
    assert_eq!(c.value, [6.0, 14.0, 4.0, 12.0, 9.0, 11.0]);
}

#[test]
#[should_panic]
fn add_panic_test() {
    let mut a = matrix::new(2, 3);
    let mut b = matrix::new(2, 2);
    a.value = vec![4.0, 6.0, 1.0, 9.0, 7.0, 3.0];
    b.value = vec![2.0, 8.0, 3.0, 3.0];

    let _ = matrix::add(&a, &b);
}

#[test]
fn scalar_test() {
    let mut a = matrix::new(2, 3);
    a.value = vec![4.0, 6.0, 1.0, 9.0, 7.0, 3.0];

    let b = matrix::scalar(&a, 1.3);
    assert_eq!(b.rows, 2);
    assert_eq!(b.columns, 3);
    assert_eq!(
        b.value,
        [
            4.0 * 1.3,
            6.0 * 1.3,
            1.0 * 1.3,
            9.0 * 1.3,
            7.0 * 1.3,
            3.0 * 1.3
        ]
    );
}

#[test]
fn element_wise_add_test() {
    let mut a = matrix::new(2, 3);
    a.value = vec![4.0, 6.0, 1.0, 9.0, 7.0, 3.0];

    let e: f32 = 5.0;
    let b = matrix::element_wise_add(&a, e);
    assert_eq!(b.rows, 2);
    assert_eq!(b.columns, 3);
    assert_eq!(b.value, [9.0, 11.0, 6.0, 14.0, 12.0, 8.0]);
}

#[test]
fn mean_test() {
    let mut a = matrix::new(2, 2);
    a.value = vec![2.0, 4.0, 1.0, 7.0];

    assert_eq!(matrix::mean(&a), 3.5);
}

#[test]
fn variance_test() {
    let mut a = matrix::new(2, 2);
    a.value = vec![2.0, 4.0, 1.0, 7.0];

    assert_eq!(matrix::variance(&a, matrix::mean(&a)), 5.25);
}

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

    let b = matrix::normalize(&a, 0.0, 1.0);
    assert_eq!(b.rows, 2);
    assert_eq!(b.columns, 3);
    assert_eq!(b.value, expected_output);
}

#[test]
fn save_load_test() {
    let mut a = matrix::new(2, 3);
    a.value = vec![3.987, 4.123, -5.245, 6.78, 9.32, -5.47];

    matrix::save(&a, "save_load_test.bin");
    let b = matrix::load(&a, "save_load_test.bin");
    fs::remove_file("save_load_test.bin").unwrap();

    assert_eq!(b.rows, 2);
    assert_eq!(b.columns, 3);
    assert_eq!(b.value, [3.987, 4.123, -5.245, 6.78, 9.32, -5.47]);
}
