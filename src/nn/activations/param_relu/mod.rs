pub mod test;

use crate::matrix;
use crate::nn::activations::Activate;

#[derive(Clone)]
pub struct ParamRelu {
    pub positive_slope: f32,
    pub negative_slope: f32,
}

pub fn new(positive_slope: f32, negative_slope: f32) -> ParamRelu {
    ParamRelu {
        positive_slope,
        negative_slope,
    }
}

impl Activate for ParamRelu {
    fn activate(&self, a: &matrix::Matrix) -> matrix::Matrix {
        let mut b = a.clone();
        for i in 0..b.rows * b.columns {
            if b.value[i] < 0.0 {
                b.value[i] *= self.negative_slope;
            } else {
                b.value[i] *= self.positive_slope;
            }
        }

        b
    }

    fn print(&self) {
        println!("Parameterized Relu");
    }
}
