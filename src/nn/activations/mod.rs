pub mod test;

use crate::matrix;

pub fn parameterized_relu(
    a: &matrix::Matrix,
    positive_slope: f32,
    negative_slope: f32,
) -> matrix::Matrix {
    let mut b = a.clone();
    for i in 0..b.rows * b.columns {
        if b.value[i] < 0.0 {
            b.value[i] *= negative_slope;
        } else {
            b.value[i] *= positive_slope;
        }
    }

    b
}
