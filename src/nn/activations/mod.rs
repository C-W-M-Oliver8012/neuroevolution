pub mod test;

use crate::matrix;

pub fn leaky_relu(a: &matrix::Matrix, negative_slope: f32) -> matrix::Matrix {
    let mut b = a.clone();
    for i in 0..b.rows * b.columns {
        if b.value[i] < 0.0 {
            b.value[i] *= negative_slope;
        }
    }
    b
}
