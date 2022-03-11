use crate::matrix;

#[derive(Clone)]
pub struct Conv {
    pub num_channels: usize,
    pub num_filters: usize,
    pub stride: usize,
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
    Conv {
        num_channels,
        num_filters,
        stride,
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
) -> Conv {
    Conv {
        num_channels,
        num_filters,
        stride,
        filters: matrix::new_gaussian_noise(
            num_filters,
            filter_rows * filter_columns * num_channels,
        ),
        filter_rows,
        filter_columns,
    }
}

// Output height = (Input hiehgt + padding height top + padding height bottom - kernel height) / (stride height) + 1
// Output width = (Input width + padding width right + padding width left - kernel width) / (stide width) + 1
// Output depth = Number of kernels

pub fn im2col(
    a: &[matrix::Matrix],
    filter_rows: usize,
    filter_columns: usize,
    num_channels: usize,
    stride: usize,
) -> matrix::Matrix {
    assert!(num_channels == a.len());

    let window_rows = (a[0].rows - filter_rows) / stride + 1;
    let window_columns = (a[0].columns - filter_columns) / stride + 1;
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

pub fn row2im(a: &matrix::Matrix, im_rows: usize, im_columns: usize) -> Vec<matrix::Matrix> {
    assert!(a.columns == im_rows * im_columns);

    let mut b: Vec<matrix::Matrix> = Vec::with_capacity(a.rows);

    for i in 0..a.rows {
        b.push(matrix::new(im_rows, im_columns));
        for j in 0..a.columns {
            let a_index = j * a.rows + i;
            for k in 0..im_rows {
                for l in 0..im_columns {
                    let b_index = l * b[i].rows + k;
                    b[i].value[b_index] = a.value[a_index];
                }
            }
        }
    }

    b
}

/*
pub fn feedforward() -> Vec<Vec<matrix::Matrix>> {

}
*/
