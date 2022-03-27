#[cfg(test)]
use crate::matrix;
#[cfg(test)]
use crate::nn::activations::no_activation;
#[cfg(test)]
use crate::nn::activations::Activate;

#[test]
fn no_activation_test() {
    let na = no_activation::new();

    let mut a = matrix::new(2, 2);
    a.value = vec![1.0, 2.0, 3.0, 4.0];

    assert!(a.rows == 2);
    assert!(a.columns == 2);
    assert!(a.value == [1.0, 2.0, 3.0, 4.0]);

    let b = na.activate(&a);

    assert!(b.rows == 2);
    assert!(b.columns == 2);
    assert!(b.value == [1.0, 2.0, 3.0, 4.0]);
}
