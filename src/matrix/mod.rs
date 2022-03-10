pub mod test;

#[cfg(feature = "enable-blas")]
extern crate blas_src;
#[cfg(feature = "enable-blas")]
use blas;

use rand::prelude::*;
use rand_distr::StandardNormal;
use std::fs;
use std::fs::File;
use std::io::Write;

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub value: Vec<f32>,
}

pub fn new(rows: usize, columns: usize) -> Matrix {
    let mut a = Matrix {
        rows,
        columns,
        value: Vec::with_capacity(rows * columns),
    };

    for _ in 0..a.rows * a.columns {
        a.value.push(0.0);
    }
    
    a
}

pub fn new_gaussian_noise(rows: usize, columns: usize) -> Matrix {
    let mut a = Matrix {
        rows,
        columns,
        value: Vec::with_capacity(rows * columns),
    };

    for _ in 0..a.rows * a.columns {
        a.value.push(thread_rng().sample(StandardNormal));
    }

    a
}

pub fn print(a: &Matrix) {
    print!("[");
    for i in 0..a.rows {
        print!("[");
        for j in 0..a.columns {
            let index = j * a.rows + i;
            print!("{}, ", a.value[index]);
        }
        if i == a.rows - 1 {
            print!("]");
        } else {
            println!("],");
        }
    }
    println!("]");
    println!();
}

#[cfg(feature = "enable-blas")]
pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.columns == b.rows, "Matrix sizes are incorrect.");

    let mut c = new(a.rows, b.columns);
    let (m, n, k) = (a.rows, b.columns, a.columns);

    unsafe {
        blas::sgemm(
            b'N',
            b'N',
            m as i32,
            n as i32,
            k as i32,
            1.0,
            &a.value,
            m as i32,
            &b.value,
            k as i32,
            0.0,
            &mut c.value,
            m as i32,
        );
    }

    c
}

#[cfg(feature = "naive")]
pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.columns == b.rows, "Matrix sizes are incorrect.");

    let mut c = new(a.rows, b.columns);

    for i in 0..a.rows {
        for j in 0..b.columns {
            let mut sum: f32 = 0.0;
            for k in 0..a.columns {
                sum += a.value[k * a.rows + i] * b.value[j * b.rows + k];
            }
            c.value[j * c.rows + i] = sum;
        }
    }

    c
}

pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.rows == b.rows, "Matrix sizes are incorrect.");
    assert!(a.columns == b.columns, "Matrix sizes are incorrect.");

    let mut c = new(a.rows, a.columns);
    for i in 0..c.rows * c.columns {
        c.value[i] = a.value[i] + b.value[i];
    }
    
    c
}

#[cfg(feature = "enable-blas")]
pub fn scalar(a: &Matrix, s: f32) -> Matrix {
    let mut b = a.clone();
    unsafe {
        blas::sscal((b.rows * b.columns) as i32, s, &mut b.value, 1);
    }

    b
}

#[cfg(feature = "naive")]
pub fn scalar(a: &Matrix, s: f32) -> Matrix {
    let mut b = a.clone();
    for i in 0..b.value.len() {
        b.value[i] *= s;
    }

    b
}

pub fn mean(a: &Matrix) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 0..a.rows * a.columns {
        sum += a.value[i];
    }

    sum / (a.rows * a.columns) as f32
}

pub fn variance(a: &Matrix, mean: f32) -> f32 {
    let mut variance: f32 = 0.0;
    for i in 0..a.rows * a.columns {
        variance += (a.value[i] - mean).powi(2);
    }

    variance / (a.rows * a.columns) as f32
}

pub fn normalize(a: &Matrix, new_mean: f32, new_std: f32) -> Matrix {
    let current_mean = mean(a);
    let current_variance = variance(a, current_mean);
    let current_std = current_variance.sqrt();

    let mut b = a.clone();
    for i in 0..b.rows * b.columns {
        b.value[i] = b.value[i] - current_mean + new_mean;
        if current_std != 0.0 {
            b.value[i] /= current_std;
            b.value[i] *= new_std;
        }
    }

    b
}

pub fn save(a: &Matrix, filename: &str) {
    let mut f = File::create(filename).unwrap();

    let mut matrix_bytes: Vec<[u8; 4]> = Vec::new();

    for i in 0..a.rows * a.columns {
        matrix_bytes.push(a.value[i].to_le_bytes());
    }

    for byte in &matrix_bytes {
        f.write_all(byte).unwrap();
    }
}

pub fn load(a: &Matrix, filename: &str) -> Matrix {
    let mut b = a.clone();
    let matrix_bytes = fs::read(filename).unwrap();

    assert_eq!(matrix_bytes.len() as f32 % 4.0, 0.0);
    assert_eq!(matrix_bytes.len() as f32 / 4.0, (a.rows * a.columns) as f32);

    let mut num_values = 0;
    let mut num_bytes = 0;
    let mut all_fbytes: [u8; 4] = [0; 4];
    for byte in matrix_bytes {
        all_fbytes[num_bytes] = byte;
        num_bytes += 1;

        if num_bytes == 4 {
            b.value[num_values] = f32::from_le_bytes(all_fbytes);
            num_bytes = 0;
            num_values += 1;
        }
    }

    b
}
