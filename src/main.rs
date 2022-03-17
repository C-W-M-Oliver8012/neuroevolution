pub mod matrix;
pub mod models;
pub mod nn;

fn main() {
    let a = matrix::new_gaussian_noise(2000, 2000);
    let b = matrix::new_gaussian_noise(2000, 2000);
    let _c = matrix::multiply(&a, &b);
}
