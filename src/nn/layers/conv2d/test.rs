#[cfg(test)]
use crate::matrix;
#[cfg(test)]
use crate::nn::activations::param_relu;
#[cfg(test)]
use crate::nn::activations::Activate;
#[cfg(test)]
use crate::nn::layers::conv2d;
#[cfg(test)]
use std::fs;

#[test]
fn new_test() {
    let conv = conv2d::new(2, 2, (2, 3), param_relu::new(1.0, 0.001));

    assert!(conv.num_channels == 2);
    assert!(conv.num_filters == 2);
    assert!(conv.filter_size.0 == 2);
    assert!(conv.filter_size.1 == 3);

    assert!(conv.filters.rows == conv.num_filters);
    assert!(conv.filters.columns == conv.filter_size.0 * conv.filter_size.1 * conv.num_channels);
    assert!(conv.bias.rows == 1);
    assert!(conv.bias.columns == conv.num_filters);

    assert!(
        conv.filters.value
            == [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
    );
}

#[test]
fn new_gaussian_noise_test() {
    let conv = conv2d::new_gaussian_noise(2, 2, (2, 3), param_relu::new(1.0, 0.001));

    assert!(conv.num_channels == 2);
    assert!(conv.num_filters == 2);
    assert!(conv.filter_size.0 == 2);
    assert!(conv.filter_size.1 == 3);

    assert!(conv.filters.rows == conv.num_filters);
    assert!(conv.filters.columns == conv.filter_size.0 * conv.filter_size.1 * conv.num_channels);
    assert!(conv.bias.rows == 1);
    assert!(conv.bias.columns == conv.num_filters);

    assert!(
        conv.filters.value
            != [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
    );
}

#[test]
fn print_test() {
    let conv = conv2d::new_gaussian_noise(2, 2, (2, 2), param_relu::new(1.0, 0.001));

    conv2d::print(&conv);
}

#[test]
fn get_window_size_test() {
    let window_size = conv2d::get_window_size((100, 100), (4, 4), (2, 2), (1, 1, 1, 1));
    assert!(window_size.0 == 50);
    assert!(window_size.1 == 50);
}

#[test]
#[should_panic]
fn get_window_size_row_stride_panic_test() {
    let _window_size = conv2d::get_window_size((100, 100), (4, 4), (0, 2), (1, 1, 1, 1));
}

#[test]
#[should_panic]
fn get_window_size_column_stride_panic_test() {
    let _window_size = conv2d::get_window_size((100, 100), (4, 4), (2, 0), (1, 1, 1, 1));
}

#[test]
#[should_panic]
fn get_window_size_non_integer_row_test() {
    let _window_size = conv2d::get_window_size((101, 100), (4, 4), (2, 2), (0, 0, 0, 0));
}

#[test]
#[should_panic]
fn get_window_size_non_integer_column_test() {
    let _window_size = conv2d::get_window_size((100, 101), (4, 4), (2, 2), (0, 0, 0, 0));
}

#[test]
fn im2col_test() {
    let mut input = vec![matrix::new(3, 3), matrix::new(3, 3)];
    input[0].value = vec![3.0, 2.0, 1.0, 9.0, 8.0, 4.0, 0.0, 1.0, 8.0];
    input[1].value = vec![3.0, 2.0, 1.0, 9.0, 8.0, 4.0, 0.0, 1.0, 8.0];

    // 2x2 filter with a 1x1 stride and 0x0x0x0 padding
    let mut filter_size: (usize, usize) = (2, 2);
    let mut stride_size: (usize, usize) = (1, 1);
    let mut window_size = conv2d::get_window_size((3, 3), filter_size, stride_size, (0, 0, 0, 0));
    let mut a = conv2d::im2col(&input, window_size, filter_size, 2, stride_size, (0, 0));

    assert!(a.rows == 8);
    assert!(a.columns == 4);
    assert!(
        a.value
            == [
                3.0, 9.0, 2.0, 8.0, 3.0, 9.0, 2.0, 8.0, 9.0, 0.0, 8.0, 1.0, 9.0, 0.0, 8.0, 1.0,
                2.0, 8.0, 1.0, 4.0, 2.0, 8.0, 1.0, 4.0, 8.0, 1.0, 4.0, 8.0, 8.0, 1.0, 4.0, 8.0
            ]
    );

    // 3x3 filter with a 3x3 stride and 3x3x3x3 padding
    filter_size = (3, 3);
    stride_size = (3, 3);
    window_size = conv2d::get_window_size((3, 3), filter_size, stride_size, (3, 3, 3, 3));
    a = conv2d::im2col(&input, window_size, filter_size, 2, stride_size, (3, 3));

    assert!(a.rows == 18);
    assert!(a.columns == 9);
    assert!(
        a.value
            == [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 9.0, 0.0, 2.0, 8.0, 1.0, 1.0, 4.0,
                8.0, 3.0, 9.0, 0.0, 2.0, 8.0, 1.0, 1.0, 4.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
            ]
    );

    // 3x3 filter with 2x2 stride and 3x3x3x3 padding
    filter_size = (3, 3);
    stride_size = (2, 2);
    window_size = conv2d::get_window_size((3, 3), filter_size, stride_size, (3, 3, 3, 3));
    a = conv2d::im2col(&input, window_size, filter_size, 2, stride_size, (3, 3));

    assert!(a.rows == 18);
    assert!(a.columns == 16);
    assert!(
        a.value
            == [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 9.0,
                0.0, 2.0, 8.0, 0.0, 0.0, 0.0, 0.0, 3.0, 9.0, 0.0, 2.0, 8.0, 0.0, 0.0, 0.0, 9.0,
                0.0, 0.0, 8.0, 1.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 8.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 2.0, 8.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 2.0, 8.0, 0.0, 1.0,
                4.0, 0.0, 0.0, 0.0, 8.0, 1.0, 0.0, 4.0, 8.0, 0.0, 0.0, 0.0, 0.0, 8.0, 1.0, 0.0,
                4.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
    );
}

#[test]
fn row2im_test() {
    let mut m = matrix::new(4, 6);
    m.value = vec![
        1.0, 7.0, 13.0, 19.0, 2.0, 8.0, 14.0, 20.0, 3.0, 9.0, 15.0, 21.0, 4.0, 10.0, 16.0, 22.0,
        5.0, 11.0, 17.0, 23.0, 6.0, 12.0, 18.0, 24.0,
    ];

    let im = conv2d::row2im(&m, (2, 3));

    assert!(im.len() == 4);

    for ind_im in im.iter() {
        assert!(ind_im.rows == 2);
        assert!(ind_im.columns == 3);
    }

    assert!(im[0].value == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert!(im[1].value == [7.0, 10.0, 8.0, 11.0, 9.0, 12.0]);
    assert!(im[2].value == [13.0, 16.0, 14.0, 17.0, 15.0, 18.0]);
    assert!(im[3].value == [19.0, 22.0, 20.0, 23.0, 21.0, 24.0]);
}

#[test]
#[should_panic]
fn row2im_column_window_size_test() {
    let m = matrix::new(4, 6);

    let _im = conv2d::row2im(&m, (2, 4));
}

#[test]
fn get_filters_test() {
    let mut conv = conv2d::new(2, 3, (2, 2), param_relu::new(1.0, 0.001));

    conv.filters.value = vec![
        1.0, 9.0, 17.0, 2.0, 10.0, 18.0, 3.0, 11.0, 19.0, 4.0, 12.0, 20.0, 5.0, 13.0, 21.0, 6.0,
        14.0, 22.0, 7.0, 15.0, 23.0, 8.0, 16.0, 24.0,
    ];

    let filters = conv2d::get_filters(&conv.filters, conv.filter_size, conv.num_channels);

    assert!(filters.len() == 6);

    for filter in filters.iter() {
        assert!(filter.rows == 2);
        assert!(filter.columns == 2);
    }

    assert!(filters[0].value == [1.0, 3.0, 2.0, 4.0]);
    assert!(filters[1].value == [5.0, 7.0, 6.0, 8.0]);
    assert!(filters[2].value == [9.0, 11.0, 10.0, 12.0]);
    assert!(filters[3].value == [13.0, 15.0, 14.0, 16.0]);
    assert!(filters[4].value == [17.0, 19.0, 18.0, 20.0]);
    assert!(filters[5].value == [21.0, 23.0, 22.0, 24.0]);
}

#[test]
#[should_panic]
fn get_filters_column_size_test() {
    let conv = conv2d::new(2, 3, (2, 2), param_relu::new(1.0, 0.001));

    let _filters = conv2d::get_filters(&conv.filters, conv.filter_size, conv.num_channels + 1);
}

#[test]
fn feedforward_test() {
    let mut input: Vec<matrix::Matrix> = Vec::new();
    let mut conv = conv2d::new(3, 3, (2, 2), param_relu::new(1.0, 1.0));

    for _ in 0..3 {
        input.push(matrix::new(3, 3));
    }

    input[0].value = vec![2.0, 1.0, 8.0, 4.0, 9.0, 4.0, 7.0, 4.0, 8.0];
    input[1].value = vec![1.0, 3.0, 8.0, 5.0, 7.0, 1.0, 9.0, 7.0, 1.0];
    input[2].value = vec![8.0, 9.0, 8.0, 1.0, 2.0, 1.0, 5.0, 4.0, 7.0];

    conv.filters.value = vec![
        1.0, 2.0, 8.0, -1.0, 4.0, 5.0, 2.0, 7.0, -1.0, -2.0, -3.0, -4.0, 3.0, -8.0, 1.0, -3.0, 1.0,
        1.0, 2.0, 5.0, -1.0, 3.0, -8.0, -1.0, 7.0, -1.0, -2.0, 1.0, 1.0, -2.0, -3.0, 8.0, 4.0, 5.0,
        2.0, 6.0,
    ];

    conv.bias.value = vec![1.0, 2.0, 3.0];

    let output = conv2d::feedforward(&conv, &input, (1, 1), (0, 0, 0, 0));

    assert!(output.len() == 3);

    for output_matrix in output.iter() {
        assert!(output_matrix.rows == 2);
        assert!(output_matrix.columns == 2);
    }

    assert!(
        output[0].value
            == [
                37.0 / 4.0 + 1.0,
                53.0 / 4.0 + 1.0,
                56.0 / 4.0 + 1.0,
                52.0 / 4.0 + 1.0
            ]
    );
    assert!(
        output[1].value
            == [
                25.0 / 4.0 + 2.0,
                156.0 / 4.0 + 2.0,
                63.0 / 4.0 + 2.0,
                10.0 / 4.0 + 2.0
            ]
    );
    assert!(
        output[2].value
            == [
                25.0 / 4.0 + 3.0,
                46.0 / 4.0 + 3.0,
                62.0 / 4.0 + 3.0,
                102.0 / 4.0 + 3.0
            ]
    );
}

#[test]
fn feedforward_activation_test() {
    let pr = param_relu::new(0.5, 0.001);

    let mut input: Vec<matrix::Matrix> = Vec::new();
    let mut conv = conv2d::new(3, 3, (2, 2), param_relu::new(0.5, 0.001));

    for _ in 0..3 {
        input.push(matrix::new(3, 3));
    }

    input[0].value = vec![2.0, 1.0, 8.0, 4.0, 9.0, 4.0, 7.0, 4.0, 8.0];
    input[1].value = vec![1.0, 3.0, 8.0, 5.0, 7.0, 1.0, 9.0, 7.0, 1.0];
    input[2].value = vec![8.0, 9.0, 8.0, 1.0, 2.0, 1.0, 5.0, 4.0, 7.0];

    conv.filters.value = vec![
        1.0, 2.0, 8.0, -1.0, 4.0, 5.0, 2.0, 7.0, -1.0, -2.0, -3.0, -4.0, 3.0, -8.0, 1.0, -3.0, 1.0,
        1.0, 2.0, 5.0, -1.0, 3.0, -8.0, -1.0, 7.0, -1.0, -2.0, 1.0, 1.0, -2.0, -3.0, 8.0, 4.0, 5.0,
        2.0, 6.0,
    ];

    conv.bias.value = vec![1.0, 2.0, 3.0];

    let output = conv2d::feedforward(&conv, &input, (1, 1), (0, 0, 0, 0));

    assert!(output.len() == 3);

    for output_matrix in output.iter() {
        assert!(output_matrix.rows == 2);
        assert!(output_matrix.columns == 2);
    }

    let mut expected_output: Vec<matrix::Matrix> = Vec::new();
    expected_output.push(matrix::new(2, 2));
    expected_output.push(matrix::new(2, 2));
    expected_output.push(matrix::new(2, 2));
    expected_output[0].value = vec![
        37.0 / 4.0 + 1.0,
        53.0 / 4.0 + 1.0,
        56.0 / 4.0 + 1.0,
        52.0 / 4.0 + 1.0,
    ];
    expected_output[1].value = vec![
        25.0 / 4.0 + 2.0,
        156.0 / 4.0 + 2.0,
        63.0 / 4.0 + 2.0,
        10.0 / 4.0 + 2.0,
    ];
    expected_output[2].value = vec![
        25.0 / 4.0 + 3.0,
        46.0 / 4.0 + 3.0,
        62.0 / 4.0 + 3.0,
        102.0 / 4.0 + 3.0,
    ];

    for output_matrix in expected_output.iter_mut() {
        *output_matrix = pr.activate(output_matrix);
    }

    for (i, output_matrix) in output.iter().enumerate() {
        assert!(output_matrix.value == expected_output[i].value);
    }
}

#[test]
fn add_test() {
    let a = conv2d::new_gaussian_noise(2, 3, (2, 2), param_relu::new(1.0, 0.001));
    let b = conv2d::new_gaussian_noise(2, 3, (2, 2), param_relu::new(1.0, 0.001));

    let c = conv2d::add(&a, &b);

    for (i, val) in c.filters.value.iter().enumerate() {
        assert!(*val == a.filters.value[i] + b.filters.value[i]);
    }

    for (i, val) in c.bias.value.iter().enumerate() {
        assert!(*val == a.bias.value[i] + b.bias.value[i]);
    }
}

#[test]
fn scalar_test() {
    let a = conv2d::new_gaussian_noise(2, 3, (2, 2), param_relu::new(1.0, 0.001));

    let b = conv2d::scalar(&a, 0.5);

    for (i, val) in b.filters.value.iter().enumerate() {
        assert!(*val == a.filters.value[i] * 0.5);
    }

    for (i, val) in b.bias.value.iter().enumerate() {
        assert!(*val == a.bias.value[i] * 0.5);
    }
}

#[test]
fn save_load_test() {
    let a = conv2d::new_gaussian_noise(2, 3, (2, 2), param_relu::new(1.0, 0.001));

    conv2d::save(&a, "conv2d");
    let b = conv2d::load(&a, "conv2d");
    fs::remove_dir_all("conv2d").unwrap();

    assert!(a.num_channels == b.num_channels);
    assert!(a.num_filters == b.num_filters);
    assert!(a.filter_size == b.filter_size);

    assert!(a.filters.rows == b.filters.rows);
    assert!(a.filters.columns == b.filters.columns);

    assert!(a.bias.rows == b.bias.rows);
    assert!(a.bias.columns == b.bias.columns);

    assert!(a.filters.value == b.filters.value);
    assert!(a.bias.value == b.bias.value);
}
