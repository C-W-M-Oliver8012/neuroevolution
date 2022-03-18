#[cfg(test)]
use crate::matrix;
#[cfg(test)]
use crate::nn::layers::conv2d;

#[test]
fn new_test() {
    let conv = conv2d::new(2, 2, (2, 3));

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
    let conv = conv2d::new_gaussian_noise(2, 2, (2, 3));

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
    let conv = conv2d::new_gaussian_noise(2, 2, (2, 2));

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
