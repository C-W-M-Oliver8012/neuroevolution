//use crate::matrix;
//use crate::nn::activations;
//use crate::nn::layers::conv2d;

pub struct Res2D {
    //pub layers: Vec<conv2d::Conv2D>,
    pub strides: Vec<(usize, usize)>,
    pub padding: Vec<(usize, usize, usize, usize)>,
}

/*
// only the new function is needed as the vec could contain any kind of conv2d
pub fn new(
    layers: &[conv2d::Conv2D],
    strides: &[(usize, usize)],
    padding: &[(usize, usize, usize, usize)],
) -> Res2D {
    assert!(
        layers.len() == strides.len() && strides.len() == padding.len(),
        "Layers, Strides, and padding must have the same length."
    );
    Res2D {
        layers: layers.to_owned(),
        strides: strides.to_owned(),
        padding: padding.to_owned(),
    }
}

pub fn print(a: &Res2D) {
    for conv2d in a.layers.iter() {
        conv2d::print(conv2d);
    }
}

pub fn feedforward(a: &Res2D, input: &[matrix::Matrix]) -> Vec<matrix::Matrix> {
    let mut output: Vec<matrix::Matrix> = Vec::with_capacity(input.len());

    for i in 0..a.layers.len() {
        if i == 0 {
            output = conv2d::feedforward(&a.layers[i], input, a.strides[i], a.padding[i]);
            for m in output.iter_mut() {
                *m = activations::parameterized_relu(m, 1.0, 0.001);
            }
        } else if i + 1 != a.layers.len() {
            output = conv2d::feedforward(&a.layers[i], &output, a.strides[i], a.padding[i]);
            for m in output.iter_mut() {
                *m = activations::parameterized_relu(m, 1.0, 0.001);
            }
        } else {
            output = conv2d::feedforward(&a.layers[i], &output, a.strides[i], a.padding[i]);
            // add input to output
            assert!(input.len() == output.len());
            for (i, m) in output.iter_mut().enumerate() {
                *m = matrix::add(m, &input[i]);
            }
            for m in output.iter_mut() {
                *m = activations::parameterized_relu(m, 1.0, 0.001);
            }
        }
    }

    output
}
*/
