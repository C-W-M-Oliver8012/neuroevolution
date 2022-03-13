pub mod matrix;
pub mod models;
pub mod nn;

use crate::nn::layers::conv;

/*
1, 2, 3, 4,
5, 6, 7, 8,
9, 10, 11, 12,
*/

fn main() {
    /*
    let mut a: Vec<matrix::Matrix> = vec![matrix::new(3, 3), matrix::new(3, 3)];
    a[0].value = vec![3.0, 2.0, 1.0, 9.0, 8.0, 4.0, 0.0, 1.0, 8.0];
    a[1].value = vec![3.0, 2.0, 1.0, 9.0, 8.0, 4.0, 0.0, 1.0, 8.0];
    matrix::print(&a[0]);
    matrix::print(&a[1]);

    let window_size = conv::get_window_size(a[0].rows, a[0].columns, 2, 2, 1);
    let b = conv::im2col(&a, window_size.0, window_size.1, 2, 2, 2);
    matrix::print(&b);
    */

    let mut a = matrix::new(3, 4);
    a.value = vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
    matrix::print(&a);

    let b = conv::row2im(&a, 2, 2);
    for i in 0..b.len() {
        matrix::print(&b[i]);
    }
}
