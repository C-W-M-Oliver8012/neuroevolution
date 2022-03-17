use crate::matrix;

#[derive(Clone)]
pub struct Conv2D {
    pub num_channels: usize,
    pub num_filters: usize,
    pub filter_rows: usize,
    pub filter_columns: usize,
    pub filters: matrix::Matrix,
    pub bias: matrix::Matrix,
}

pub fn new(
    num_channels: usize,
    num_filters: usize,
    filter_rows: usize,
    filter_columns: usize,
) -> Conv2D {
    Conv2D {
        num_channels,
        num_filters,
        filter_rows,
        filter_columns,
        filters: matrix::new(num_filters, filter_rows * filter_columns * num_channels),
        bias: matrix::new(1, num_filters),
    }
}

pub fn new_gaussian_noise(
    num_channels: usize,
    num_filters: usize,
    filter_rows: usize,
    filter_columns: usize,
) -> Conv2D {
    Conv2D {
        num_channels,
        num_filters,
        filter_rows,
        filter_columns,
        filters: matrix::new_gaussian_noise(
            num_filters,
            filter_rows * filter_columns * num_channels,
        ),
        bias: matrix::new_gaussian_noise(1, num_filters),
    }
}

pub fn print(conv: &Conv2D) {
    println!("Conv2D Layers:");
    println!("Num Channels: {}", conv.num_channels);
    println!("Num Filters: {}", conv.num_filters);
    println!("Filter Rows: {}", conv.filter_rows);
    println!("Filter Columns: {}", conv.filter_columns);

    let filters = get_filters(
        &conv.filters,
        conv.filter_rows,
        conv.filter_columns,
        conv.num_channels,
    );

    println!("Filters:");
    println!("-------------------------------------------");
    for (i, filter) in filters.iter().enumerate() {
        matrix::print(filter);
        if (i + 1) as f32 % conv.num_channels as f32 == 0.0 {
            println!("-------------------------------------------");
        }
    }

    println!("Bias:");
    matrix::print(&conv.bias);
}

// Output height = (Input hiehgt + padding height top + padding height bottom - kernel height) / (stride height) + 1
// Output width = (Input width + padding width right + padding width left - kernel width) / (stide width) + 1
// Output depth = Number of kernels

pub fn get_window_size(
    input_size: (usize, usize),
    filter_size: (usize, usize),
    stride_size: (usize, usize),
    padding: (usize, usize, usize, usize),
) -> (usize, usize) {
    assert!(stride_size.0 != 0, "Row stride cannot be zero.");
    assert!(stride_size.1 != 0, "Column stride cannot be zero.");
    let window_rows =
        (input_size.0 + padding.0 + padding.1 - filter_size.0) as f64 / stride_size.0 as f64 + 1.0;
    let window_columns =
        (input_size.1 + padding.2 + padding.3 - filter_size.1) as f64 / stride_size.1 as f64 + 1.0;

    assert!(
        window_rows % 1.0 == 0.0,
        "Non-integer output size from input size, filter size, and stride."
    );
    assert!(
        window_columns % 1.0 == 0.0,
        "Non-integer output size from input size, filter size, and stride."
    );

    (window_rows as usize, window_columns as usize)
}

pub fn im2col(
    a: &[matrix::Matrix],
    window_size: (usize, usize),
    filter_size: (usize, usize),
    num_channels: usize,
    stride_size: (usize, usize),
    padding: (usize, usize),
) -> matrix::Matrix {
    let mut b = matrix::new(
        filter_size.0 * filter_size.1 * num_channels,
        window_size.0 * window_size.1,
    );

    // offset by top_padding
    let neg_wr: isize = (-(padding.0 as isize)) * stride_size.0 as isize;
    let pos_wr: isize = (window_size.0 as isize - padding.0 as isize) * stride_size.0 as isize;

    // offset by left_padding
    let neg_wc: isize = (-(padding.1 as isize)) * stride_size.1 as isize;
    let pos_wc: isize = (window_size.1 as isize - padding.1 as isize) * stride_size.1 as isize;

    let mut inc: usize = 0;
    for wr in (neg_wr..pos_wr).step_by(stride_size.0) {
        for wc in (neg_wc..pos_wc).step_by(stride_size.1) {
            for nc in 0..num_channels {
                assert!(
                    a[0].rows == a[nc].rows,
                    "Input matrices must have same size."
                );
                assert!(
                    a[0].columns == a[nc].columns,
                    "Input matrices must have same size."
                );

                assert!(
                    filter_size.0 <= a[nc].rows,
                    "Filter size must be smaller than input size."
                );
                assert!(
                    filter_size.1 <= a[nc].columns,
                    "Filter size must be smaller than input size"
                );
                for fr in 0..filter_size.0 {
                    for fc in 0..filter_size.1 {
                        // if rows or columns are outside range of matrix, then set to 0.0
                        if wr < 0
                            || wc < 0
                            || wr as usize + fr >= a[nc].rows
                            || wc as usize + fc >= a[nc].columns
                        {
                            b.value[inc] = 0.0;
                        } else {
                            let a_index = (wc as usize + fc) * a[nc].rows + (wr as usize + fr);
                            b.value[inc] = a[nc].value[a_index];
                        }
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
    assert!(
        a.columns == window_rows * window_columns,
        "Colomn must contain size of output rows * output columns"
    );

    let mut b: Vec<matrix::Matrix> = Vec::with_capacity(a.rows);

    let mut ic = 0;
    for ir in 0..a.rows {
        b.push(matrix::new(window_rows, window_columns));
        for wr in 0..window_rows {
            for wc in 0..window_columns {
                let a_index = ic * a.rows + ir;
                let b_index = wc * b[ir].rows + wr;
                b[ir].value[b_index] = a.value[a_index];
                ic += 1;
            }
        }
        ic = 0;
    }

    b
}

pub fn get_filters(
    a: &matrix::Matrix,
    filter_rows: usize,
    filter_columns: usize,
    num_channels: usize,
) -> Vec<matrix::Matrix> {
    assert!(a.columns == filter_rows * filter_columns * num_channels);

    let mut b: Vec<matrix::Matrix> = Vec::with_capacity(a.rows * num_channels);

    let mut inc = 0;
    let mut ic = 0;
    for ir in 0..a.rows {
        for _nc in 0..num_channels {
            b.push(matrix::new(filter_rows, filter_columns));
            for fr in 0..filter_rows {
                for fc in 0..filter_columns {
                    let a_index = ic * a.rows + ir;
                    let b_index = fc * b[ir].rows + fr;
                    b[inc].value[b_index] = a.value[a_index];
                    ic += 1;
                }
            }
            inc += 1;
        }
        ic = 0;
    }

    b
}

pub fn feedforward(
    conv: &Conv2D,
    input: &[matrix::Matrix],
    stride: (usize, usize),
    padding: (usize, usize, usize, usize),
) -> Vec<matrix::Matrix> {
    assert!(
        input.len() == conv.num_channels,
        "Input depth and number of channels must match."
    );

    let window_size = get_window_size(
        (input[0].rows, input[0].columns),
        (conv.filter_rows, conv.filter_columns),
        (stride.0, stride.1),
        (padding.0, padding.1, padding.2, padding.3),
    );

    let mut output_matrix = im2col(
        input,
        (window_size.0, window_size.1),
        (conv.filter_rows, conv.filter_columns),
        conv.num_channels,
        (stride.0, stride.1),
        (padding.0, padding.2),
    );
    output_matrix = matrix::multiply(&conv.filters, &output_matrix);

    let mut output = row2im(&output_matrix, window_size.0, window_size.1);

    for (i, output_im) in output.iter_mut().enumerate() {
        *output_im = matrix::element_wise_add(output_im, conv.bias.value[i]);
    }

    output
}

pub fn add(a: &Conv2D, b: &Conv2D) -> Conv2D {
    let mut c = a.clone();

    c.filters = matrix::add(&c.filters, &b.filters);
    c.bias = matrix::add(&c.bias, &b.bias);

    c
}

pub fn scalar(a: &Conv2D, s: f32) -> Conv2D {
    let mut b = a.clone();

    b.filters = matrix::scalar(&b.filters, s);
    b.bias = matrix::scalar(&b.bias, s);

    b
}
