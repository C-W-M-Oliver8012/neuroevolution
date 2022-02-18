pub mod test;

use crate::matrix;

pub fn normalize(a: &matrix::Matrix, mean: f32, std: f32) -> matrix::Matrix {
    let current_mean = matrix::mean(a);
    let current_variance = matrix::variance(a, current_mean);
    let current_std = current_variance.sqrt();

    let mut b = a.clone();
    for i in 0..b.rows * b.columns {
        b.value[i] = b.value[i] - current_mean + mean;
        if current_std != 0.0 {
            b.value[i] /= current_std;
            b.value[i] *= std;
        }
    }
    b
}

pub fn leaky_relu(a: &matrix::Matrix, negative_slope: f32) -> matrix::Matrix {
    let mut b = a.clone();
    for i in 0..b.rows * b.columns {
        if b.value[i] < 0.0 {
            b.value[i] *= negative_slope;
        }
    }
    b
}
