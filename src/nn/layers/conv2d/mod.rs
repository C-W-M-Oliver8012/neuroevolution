use crate::matrix;

#[derive(Clone)]
pub struct Conv2D {
    pub num_channels: usize,
    pub num_filters: usize,
    pub stride: usize,
    pub bias: matrix::Matrix,
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
) -> Conv2D {
    Conv2D {
        num_channels,
        num_filters,
        stride,
        bias: matrix::new(1, num_filters),
        filters: matrix::new(num_filters, filter_rows * filter_columns * num_channels),
        filter_rows,
        filter_columns,
    }
}

pub fn new_gaussian_noise(
    num_channels: usize,
    num_filters: usize,
    stride: usize,
    filter_rows: usize,
    filter_columns: usize,
) -> Conv2D {
    Conv2D {
        num_channels,
        num_filters,
        stride,
        bias: matrix::new_gaussian_noise(1, num_filters),
        filters: matrix::new_gaussian_noise(
            num_filters,
            filter_rows * filter_columns * num_channels,
        ),
        filter_rows,
        filter_columns,
    }
}

pub fn print(conv: &Conv2D) {
    println!("Conv2D Layers:");
    println!("Num Channels: {}", conv.num_channels);
    println!("Num Filters: {}", conv.num_filters);
    println!("Stride: {}", conv.stride);
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
    input_rows: usize,
    input_columns: usize,
    filter_rows: usize,
    filter_columns: usize,
    stride: usize,
) -> (usize, usize) {
    assert!(stride != 0);
    let window_rows = (input_rows as f32 - filter_rows as f32) / stride as f32 + 1.0;
    let window_columns = (input_columns as f32 - filter_columns as f32) / stride as f32 + 1.0;

    assert!(window_rows % 1.0 == 0.0);
    assert!(window_columns % 1.0 == 0.0);

    (window_rows as usize, window_columns as usize)
}

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

    b
}

pub fn feedforward(conv: &Conv2D, input: &[matrix::Matrix]) -> Vec<matrix::Matrix> {
    assert!(input.len() == conv.num_channels);

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
