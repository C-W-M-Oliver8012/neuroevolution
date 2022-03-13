use crate::matrix;
use rand::prelude::*;
use rand_distr::StandardNormal;

#[derive(Clone)]
pub struct Conv {
    pub num_channels: usize,
    pub num_filters: usize,
    pub stride: usize,
    pub bias: Vec<f32>,
    pub filters: matrix::Matrix,
    pub filter_rows: usize,
    pub filter_columns: usize,
}

pub fn new(
    num_channels: usize,
    num_filters: usize,
    stride: usize,
    filter_rows: usize,
    filter_columns: usize,
) -> Conv {
    let mut conv = Conv {
        num_channels,
        num_filters,
        stride,
        bias: Vec::with_capacity(num_filters),
        filters: matrix::new(num_filters, filter_rows * filter_columns * num_channels),
        filter_rows,
        filter_columns,
    };

    for _ in 0..num_filters {
        conv.bias.push(0.0);
    }

    conv
}

pub fn new_gaussian_noise(
    num_channels: usize,
    num_filters: usize,
    stride: usize,
    filter_rows: usize,
    filter_columns: usize,
) -> Conv {
    let mut conv = Conv {
        num_channels,
        num_filters,
        stride,
        bias: Vec::with_capacity(num_filters),
        filters: matrix::new_gaussian_noise(
            num_filters,
            filter_rows * filter_columns * num_channels,
        ),
        filter_rows,
        filter_columns,
    };

    for _ in 0..num_filters {
        conv.bias.push(thread_rng().sample(StandardNormal));
    }

    conv
}

// Output height = (Input hiehgt + padding height top + padding height bottom - kernel height) / (stride height) + 1
// Output width = (Input width + padding width right + padding width left - kernel width) / (stide width) + 1
// Output depth = Number of kernels

pub fn get_window_size(
    input_rows: usize,
    input_columns: usize,
    filter_rows: usize,
    filter_columns: usize,
    stride: usize,
) -> (usize, usize) {
    let window_rows = (input_rows - filter_rows) / stride + 1;
    let window_columns = (input_columns - filter_columns) / stride + 1;

    (window_rows, window_columns)
}

pub fn im2col(
    a: &[matrix::Matrix],
    window_rows: usize,
    window_columns: usize,
    filter_rows: usize,
    filter_columns: usize,
    num_channels: usize,
) -> matrix::Matrix {
    assert!(num_channels == a.len());

    let mut b = matrix::new(
        filter_rows * filter_columns * num_channels,
        window_rows * window_columns,
    );

    let mut inc: usize = 0;
    for i in 0..window_rows {
        for j in 0..window_columns {
            for n in 0..num_channels {
                assert!(a[0].rows == a[n].rows);
                assert!(a[0].columns == a[n].columns);

                assert!(filter_rows <= a[n].rows);
                assert!(filter_columns <= a[n].columns);
                for k in 0..filter_rows {
                    for l in 0..filter_columns {
                        let a_index = (j + l) * a[n].rows + (i + k);
                        b.value[inc] = a[n].value[a_index];
                        inc += 1;
                    }
                }
            }
        }
    }

    b
}

pub fn row2im(
    a: &matrix::Matrix,
    window_rows: usize,
    window_columns: usize,
) -> Vec<matrix::Matrix> {
    assert!(a.columns == window_rows * window_columns);

    let mut b: Vec<matrix::Matrix> = Vec::with_capacity(a.rows);

    for i in 0..a.rows {
        b.push(matrix::new(window_rows, window_columns));
        for j in 0..a.columns {
            let a_index = j * a.rows + i;
            for k in 0..window_rows {
                for l in 0..window_columns {
                    let b_index = l * b[i].rows + k;
                    b[i].value[b_index] = a.value[a_index];
                }
            }
        }
    }

    b
}

pub fn feedforward(conv: Conv, input: &[matrix::Matrix]) -> Vec<matrix::Matrix> {
    let window_size = get_window_size(
        input[0].rows,
        input[0].columns,
        conv.filter_rows,
        conv.filter_columns,
        conv.stride,
    );

    let mut output_matrix = im2col(
        input,
        window_size.0,
        window_size.1,
        conv.filter_rows,
        conv.filter_columns,
        conv.num_channels,
    );
    output_matrix = matrix::multiply(&conv.filters, &output_matrix);

    let mut output = row2im(&output_matrix, window_size.0, window_size.1);

    for i in 0..conv.num_filters {
        output[i] = matrix::scalar(
            &output[i],
            (conv.filter_rows * conv.filter_columns * conv.num_channels) as f32,
        );
        output[i] = matrix::element_wise_add(&output[i], conv.bias[i]);
    }

    output
}
