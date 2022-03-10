pub mod matrix;
pub mod models;
pub mod nn;

use crate::nn::layers::conv;

fn main() {
    /*
    let conv = conv::new_gaussian_noise(2, 1, 1, 2, 2);

    for i in 0..conv.filters.len() {
        for j in 0..conv.filters[i].len() {
            matrix::print(&conv.filters[i][j]);
            println!();
        }
        println!("-----------------------------------------------");
    }
    */

    let mut a = matrix::new(3, 3);
    a.value = vec![
        3.0, 2.0, 1.0, 9.0, 8.0, 4.0, 0.0, 1.0, 8.0,
    ];
    matrix::print(&a);

    let b = conv::reshape(&a, 2, 2, 1);
    matrix::print(&b.0);
    println!("{}, {}", b.1, b.2);
}
