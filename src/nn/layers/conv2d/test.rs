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
    let mut conv = conv2d::new(2, 3, (2, 2));

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
    let conv = conv2d::new(2, 3, (2, 2));

    let _filters = conv2d::get_filters(&conv.filters, conv.filter_size, conv.num_channels + 1);
}