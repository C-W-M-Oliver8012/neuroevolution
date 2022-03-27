use crate::matrix;
use crate::nn::activations::no_activation;
use crate::nn::activations::Activate;
use crate::nn::layers::conv2d;

#[derive(Clone)]
pub struct Res2D<T: Activate> {
    pub layers: Vec<conv2d::Conv2D<T>>,
    pub last_layer: conv2d::Conv2D<no_activation::NoActivation>,
    pub strides: Vec<(usize, usize)>,
    pub padding: Vec<(usize, usize, usize, usize)>,
    pub activation: T,
}

// only the new function is needed as the vec could contain any kind of conv2d
pub fn new<T: Activate + Clone>(
    layers: &[conv2d::Conv2D<T>],
    last_layer: &conv2d::Conv2D<no_activation::NoActivation>,
    strides: &[(usize, usize)],
    padding: &[(usize, usize, usize, usize)],
    activation: T,
) -> Res2D<T> {
    assert!(
        layers.len() + 1 == strides.len() && strides.len() == padding.len(),
        "Layers, Strides, and padding must have the same length."
    );
    Res2D {
        layers: layers.to_owned(),
        last_layer: last_layer.clone(),
        strides: strides.to_owned(),
        padding: padding.to_owned(),
        activation,
    }
}

pub fn print<T: Activate>(a: &Res2D<T>) {
    for conv2d in a.layers.iter() {
        conv2d::print(conv2d);
    }
}

pub fn feedforward<T: Activate>(a: &Res2D<T>, input: &[matrix::Matrix]) -> Vec<matrix::Matrix> {
    let mut output: Vec<matrix::Matrix> = Vec::with_capacity(input.len());

    for i in 0..a.strides.len() {
        if i == 0 {
            output = conv2d::feedforward(&a.layers[i], input, a.strides[i], a.padding[i]);
        } else if i + 1 != a.strides.len() {
            output = conv2d::feedforward(&a.layers[i], &output, a.strides[i], a.padding[i]);
        } else {
            output = conv2d::feedforward(&a.last_layer, &output, a.strides[i], a.padding[i]);
            // add input to output
            assert!(input.len() == output.len());
            for (i, m) in output.iter_mut().enumerate() {
                *m = matrix::add(m, &input[i]);
            }
            for m in output.iter_mut() {
                *m = a.activation.activate(m);
            }
        }
    }

    output
}

pub fn add<T: Activate + Clone>(a: &Res2D<T>, b: &Res2D<T>) -> Res2D<T> {
    let mut c = a.clone();

    for (i, layer) in c.layers.iter_mut().enumerate() {
        *layer = conv2d::add(layer, &b.layers[i]);
    }
    c.last_layer = conv2d::add(&c.last_layer, &b.last_layer);

    c
}

pub fn scalar<T: Activate + Clone>(a: &Res2D<T>, s: f32) -> Res2D<T> {
    let mut b = a.clone();

    for layer in b.layers.iter_mut() {
        *layer = conv2d::scalar(layer, s);
    }
    b.last_layer = conv2d::scalar(&b.last_layer, s);

    b
}
