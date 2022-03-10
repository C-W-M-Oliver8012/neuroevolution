use crate::matrix;

#[derive(Clone)]
pub struct Conv {
    pub num_channels: usize,
    pub num_filters: usize,
    pub stride: usize,
    pub filters: Vec<Vec<matrix::Matrix>>,
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
        filters: Vec::with_capacity(num_filters),
        filter_rows,
        filter_columns,
    };

    for i in 0..num_filters {
        conv.filters.push(Vec::with_capacity(num_channels));
        for _ in 0..num_channels {
            conv.filters[i].push(matrix::new(1, filter_rows * filter_columns));
        }
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
        filters: Vec::with_capacity(num_filters),
        filter_rows,
        filter_columns,
    };

    for i in 0..num_filters {
        conv.filters.push(Vec::with_capacity(num_channels));
        for _ in 0..num_channels {
            conv.filters[i].push(matrix::new_gaussian_noise(1, filter_rows * filter_columns));
        }
    }

    conv
}

pub fn reshape(
    a: &matrix::Matrix,
    filter_rows: usize,
    filter_columns: usize,
    stride: usize,
) -> (matrix::Matrix, usize, usize) {
    let mut output_rows: usize = 0;
    let mut output_columns: usize = 0;
    let mut b_value: Vec<f32> = Vec::new();
    let mut num_windows: usize = 0;

    assert!(filter_rows <= a.rows);
    assert!(filter_columns <= a.columns);

    for i in (0..a.rows - filter_rows + 1).step_by(stride) {
        output_columns = 0;
        output_rows += 1;
        for j in (0..a.columns - filter_columns + 1).step_by(stride) {
            output_columns += 1;
            for k in 0..filter_rows {
                for l in 0..filter_columns {
                    b_value.push(a.value[(j + l) * a.rows + (i + k)]);
                }
            }
            num_windows += 1;
        }
    }

    let mut b: matrix::Matrix = matrix::new(filter_rows * filter_columns, num_windows);

    assert!(b_value.len() == b.value.len());
    assert!(num_windows == output_rows * output_columns);
    
    b.value = b_value;

    (b, output_rows, output_columns)
}

/*
pub fn feedforward() -> Vec<Vec<matrix::Matrix>> {

}
*/
