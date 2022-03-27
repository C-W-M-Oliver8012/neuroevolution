pub mod test;

use crate::matrix;
use crate::nn::activations::Activate;

#[derive(Clone)]
pub struct NoActivation {}

pub fn new() -> NoActivation {
    NoActivation {}
}

impl Activate for NoActivation {
    fn activate(&self, a: &matrix::Matrix) -> matrix::Matrix {
        a.clone()
    }

    fn print(&self) {
        println!("No Activation");
    }
}
