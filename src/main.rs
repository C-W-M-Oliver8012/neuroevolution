pub mod matrix;
pub mod models;
pub mod nn;

use crate::nn::layers::conv;

fn main() {
    let mut a: Vec<matrix::Matrix> = vec![matrix::new(3, 3), matrix::new(3, 3)];
    a[0].value = vec![3.0, 2.0, 1.0, 9.0, 8.0, 4.0, 0.0, 1.0, 8.0];
    a[1].value = vec![3.0, 2.0, 1.0, 9.0, 8.0, 4.0, 0.0, 1.0, 8.0];
    matrix::print(&a[0]);
    matrix::print(&a[1]);

    let b = conv::im2col(&a, 2, 2, 2, 1);
    matrix::print(&b);
}
