pub mod matrix;
pub mod models;
pub mod nn;

fn main() {
    models::xor::xor_ga::train();
}
