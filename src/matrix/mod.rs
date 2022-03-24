pub mod test;

#[link(name = "blis")]
extern "C" {
    fn sgemm_(
        transa: *const std::os::raw::c_char,
        transb: *const std::os::raw::c_char,
        m: *const std::os::raw::c_int,
        n: *const std::os::raw::c_int,
        k: *const std::os::raw::c_int,
        alpha: *const std::os::raw::c_float,
        a: *const std::os::raw::c_float,
        lda: *const std::os::raw::c_int,
        b: *const std::os::raw::c_float,
        ldb: *const std::os::raw::c_int,
        beta: *const std::os::raw::c_float,
        c: *mut std::os::raw::c_float,
        ldc: *const std::os::raw::c_int,
    );

    fn sscal_(
        n: *const std::os::raw::c_int,
        alpha: *const std::os::raw::c_float,
        x: *mut std::os::raw::c_float,
        incx: *const std::os::raw::c_int,
    );
}

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
    assert!(rows != 0);
    assert!(columns != 0);
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
    assert!(rows != 0);
    assert!(columns != 0);
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

pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.columns == b.rows, "Matrix sizes are incorrect.");

    let mut c = new(a.rows, b.columns);
    let (m, n, k) = (a.rows, b.columns, a.columns);

    unsafe {
        sgemm_(
            &(b'N' as std::os::raw::c_char),
            &(b'N' as std::os::raw::c_char),
            &(m as std::os::raw::c_int),
            &(n as std::os::raw::c_int),
            &(k as std::os::raw::c_int),
            &(1.0_f32),
            a.value.as_ptr(),
            &(m as std::os::raw::c_int),
            b.value.as_ptr(),
            &(k as std::os::raw::c_int),
            &(0.0_f32),
            c.value.as_mut_ptr(),
            &(m as std::os::raw::c_int),
        );
    }

    c
}

pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.rows == b.rows, "Matrix sizes are incorrect.");
    assert!(a.columns == b.columns, "Matrix sizes are incorrect.");

    let mut c = new(a.rows, a.columns);

    for i in 0..c.value.len() {
        c.value[i] = a.value[i] + b.value[i];
    }

    c
}

pub fn scalar(a: &Matrix, s: f32) -> Matrix {
    let mut b = a.clone();

    unsafe {
        sscal_(
            &((b.rows * b.columns) as std::os::raw::c_int),
            &s,
            b.value.as_mut_ptr(),
            &(1_i32),
        );
    }

    b
}

pub fn element_wise_add(a: &Matrix, e: f32) -> Matrix {
    let mut b = a.clone();

    for i in 0..b.value.len() {
        b.value[i] += e;
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
    for i in 0..a.value.len() {
        variance += (a.value[i] - mean).powi(2);
    }

    variance / (a.rows * a.columns) as f32
}

pub fn normalize(a: &Matrix, new_mean: f32, new_std: f32) -> Matrix {
    let current_mean = mean(a);
    let current_variance = variance(a, current_mean);
    let current_std = current_variance.sqrt();

    let mut b = a.clone();
    for i in 0..b.value.len() {
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

    for i in 0..a.value.len() {
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
