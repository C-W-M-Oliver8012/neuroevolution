pub mod no_activation;
pub mod param_relu;

use crate::matrix;

pub trait Activate {
    fn activate(&self, a: &matrix::Matrix) -> matrix::Matrix;
}
