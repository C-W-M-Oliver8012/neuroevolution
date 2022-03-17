use crate::matrix;

#[derive(Clone)]
pub struct Conv2D {
    pub num_channels: usize,
    pub num_filters: usize,
    pub filter_rows: usize,
    pub filter_columns: usize,
    pub row_stride: usize,
    pub column_stride: usize,
    pub filters: matrix::Matrix,
    pub bias: matrix::Matrix,
}

pub fn new(
    num_channels: usize,
    num_filters: usize,
    filter_rows: usize,
    filter_columns: usize,
    row_stride: usize,
    column_stride: usize,
) -> Conv2D {
    Conv2D {
        num_channels,
        num_filters,
        filter_rows,
        filter_columns,
        row_stride,
        column_stride,
        filters: matrix::new(num_filters, filter_rows * filter_columns * num_channels),
        bias: matrix::new(1, num_filters),
    }
}

pub fn new_gaussian_noise(
    num_channels: usize,
    num_filters: usize,
    filter_rows: usize,
    filter_columns: usize,
    row_stride: usize,
    column_stride: usize,
) -> Conv2D {
    Conv2D {
        num_channels,
        num_filters,
        filter_rows,
        filter_columns,
        row_stride,
        column_stride,
        filters: matrix::new_gaussian_noise(num_filters, filter_rows * filter_columns * num_channels),
        bias: matrix::new_gaussian_noise(1, num_filters),
    }
}

pub fn print(conv: &Conv2D) {
    println!("Conv2D Layers:");
    println!("Num Channels: {}", conv.num_channels);
    println!("Num Filters: {}", conv.num_filters);
    println!("Filter Rows: {}", conv.filter_rows);
    println!("Filter Columns: {}", conv.filter_columns);
    println!("Row Stride: {}", conv.row_stride);
    println!("Column Stride: {}", conv.column_stride);

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
    input_rows: usize,
    input_columns: usize,
    filter_rows: usize,
    filter_columns: usize,
    row_stride: usize,
    column_stride: usize,
    top_padding: usize,
    bottom_padding: usize,
    left_padding: usize,
    right_padding: usize,
) -> (usize, usize) {
    assert!(row_stride != 0, "Row stride cannot be zero.");
    assert!(column_stride != 0, "Column stride cannot be zero.");
    let window_rows = (input_rows + top_padding + bottom_padding - filter_rows) as f64 / row_stride as f64 + 1.0;
    let window_columns = (input_columns + left_padding + right_padding - filter_columns) as f64 / column_stride as f64 + 1.0;

    assert!(window_rows % 1.0 == 0.0, "Non-integer output size from input size, filter size, and stride.");
    assert!(window_columns % 1.0 == 0.0, "Non-integer output size from input size, filter size, and stride.");

    (window_rows as usize, window_columns as usize)
}

/*
pub fn im2col(
    a: &[matrix::Matrix],
    window_rows: usize,
    window_columns: usize,
    filter_rows: usize,
    filter_columns: usize,
    num_channels: usize,
) -> matrix::Matrix {
    let mut b = matrix::new(
        filter_rows * filter_columns * num_channels,
        window_rows * window_columns,
    );

    let mut inc: usize = 0;
    for i in 0..window_rows {
        for j in 0..window_columns {
            for n in 0..num_channels {
                assert!(a[0].rows == a[n].rows, "Input matrices must have same size.");
                assert!(a[0].columns == a[n].columns, "Input matrices must have same size.");

                assert!(filter_rows <= a[n].rows, "Filter size must be smaller than input size.");
                assert!(filter_columns <= a[n].columns, "Filter size must be smaller than input size");
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
*/

// Output height = (Input hiehgt + padding height top + padding height bottom - kernel height) / (stride height) + 1
// Output width = (Input width + padding width right + padding width left - kernel width) / (stide width) + 1
// Output depth = Number of kernels

pub fn im2col(
    a: &[matrix::Matrix],
    window_rows: usize,
    window_columns: usize,
    filter_rows: usize,
    filter_columns: usize,
    num_channels: usize,
    row_stride: usize,
    column_stride: usize,
    top_padding: usize,
    left_padding: usize,
) -> matrix::Matrix {
    let mut b = matrix::new(
        filter_rows * filter_columns * num_channels,
        window_rows * window_columns,
    );

    // offset by top_padding
    let neg_wr: isize = (-1 * top_padding as isize) * row_stride as isize;
    let pos_wr: isize = (window_rows as isize - top_padding as isize) * row_stride as isize;

    // offset by left_padding
    let neg_wc: isize = (-1 * left_padding as isize) * column_stride as isize;
    let pos_wc: isize = (window_columns as isize - left_padding as isize) * column_stride as isize;

    let mut inc: usize = 0;
    for wr in (neg_wr..pos_wr).step_by(row_stride) {
        for wc in (neg_wc..pos_wc).step_by(column_stride) {
            for nc in 0..num_channels {
                assert!(a[0].rows == a[nc].rows, "Input matrices must have same size.");
                assert!(a[0].columns == a[nc].columns, "Input matrices must have same size.");

                assert!(filter_rows <= a[nc].rows, "Filter size must be smaller than input size.");
                assert!(filter_columns <= a[nc].columns, "Filter size must be smaller than input size");
                for fr in 0..filter_rows {
                    for fc in 0..filter_columns {
                        // if rows or columns are outside range of matrix, then set to 0.0
                        if wr < 0 || wc < 0 || wr as usize >= a[nc].rows || wc as usize >= a[nc].columns {
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
    assert!(a.columns == window_rows * window_columns, "Colomn must contain size of output rows * output columns");

    let mut b: Vec<matrix::Matrix> = Vec::with_capacity(a.rows);

    /*
    let mut j = 0;
    for i in 0..a.rows {
        b.push(matrix::new(window_rows, window_columns));
        for k in 0..window_rows {
            for l in 0..window_columns {
                let a_index = j * a.rows + i;
                let b_index = l * b[i].rows + k;
                b[i].value[b_index] = a.value[a_index];
                j += 1;
            }
        }
        j = 0;
    }
    */

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

    /*
    let mut inc = 0;
    let mut j = 0;
    for i in 0..a.rows {
        for _ in 0..num_channels {
            b.push(matrix::new(filter_rows, filter_columns));
            for k in 0..filter_rows {
                for l in 0..filter_columns {
                    let a_index = j * a.rows + i;
                    let b_index = l * b[i].rows + k;
                    b[inc].value[b_index] = a.value[a_index];
                    j += 1;
                }
            }
            inc += 1;
        }
        j = 0;
    }
    */

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
    top_padding: usize,
    bottom_padding: usize,
    left_padding: usize,
    right_padding: usize,
) -> Vec<matrix::Matrix> {
    assert!(input.len() == conv.num_channels, "Input depth and number of channels must match.");

    let window_size = get_window_size(
        input[0].rows,
        input[0].columns,
        conv.filter_rows,
        conv.filter_columns,
        conv.row_stride,
        conv.column_stride,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
    );

    let mut output_matrix = im2col(
        input,
        window_size.0,
        window_size.1,
        conv.filter_rows,
        conv.filter_columns,
        conv.num_channels,
        conv.row_stride,
        conv.column_stride,
        top_padding,
        left_padding,
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
