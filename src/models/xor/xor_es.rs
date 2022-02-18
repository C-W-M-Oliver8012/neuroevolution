use crate::matrix;
use crate::models::xor;
use std::sync::mpsc;
use std::thread;

fn fitness(model: &xor::XorModel) -> (f32, bool) {
    let mut did_pass = false;
    let mut score: f32 = 0.0;

    // 0.0, 0.0 -> 0.0
    let mut input = matrix::new(1, 2);
    input.value = vec![0.0, 0.0];
    let mut output = xor::feedforward(model, &input);
    score += (output.value[0] - output.value[1]).min(1.0);

    // 1.0, 0.0 -> 1.0
    input.value = vec![1.0, 0.0];
    output = xor::feedforward(model, &input);
    score += (output.value[1] - output.value[0]).min(1.0);

    // 0.0, 1.0 -> 1.0
    input.value = vec![0.0, 1.0];
    output = xor::feedforward(model, &input);
    score += (output.value[1] - output.value[0]).min(1.0);

    // 1.0, 1.0 -> 1.0
    input.value = vec![1.0, 1.0];
    output = xor::feedforward(model, &input);
    score += (output.value[0] - output.value[1]).min(1.0);

    if score >= 4.0 {
        did_pass = true;
    }

    (score, did_pass)
}

pub fn print_outputs(model: &xor::XorModel) {
    let mut input = matrix::new(1, 2);

    input.value = vec![0.0, 0.0];
    let mut output = xor::feedforward(model, &input);
    println!("0.0, 0.0 => {}, {}", output.value[0], output.value[1]);

    input.value = vec![1.0, 0.0];
    output = xor::feedforward(model, &input);
    println!("1.0, 0.0 => {}, {}", output.value[0], output.value[1]);

    input.value = vec![0.0, 1.0];
    output = xor::feedforward(model, &input);
    println!("0.0, 1.0 => {}, {}", output.value[0], output.value[1]);

    input.value = vec![1.0, 1.0];
    output = xor::feedforward(model, &input);
    println!("1.0, 1.0 => {}, {}", output.value[0], output.value[1]);
}

pub fn train(lr: f32, std: f32, ps: usize, nt: usize) {
    let mut xor_model = xor::new_gaussian_noise();

    let mut g: usize = 0;
    while !fitness(&xor_model).1 {
        let s = fitness(&xor_model);
        println!("Generation {}: {} {}", g, s.0, s.1);
        g += 1;

        let mut xor_update = xor::new();

        let (tx, rx) = mpsc::channel();
        let mut handles = vec![];

        for _ in 0..nt {
            let txc = tx.clone();
            let xorc = xor_model.clone();
            let handle = thread::spawn(move || {
                let mut thread_update = xor::new();
                for _ in 0..ps {
                    let mut gn = xor::new_gaussian_noise();
                    gn = xor::scalar(&gn, std);
                    let nxor_model = xor::add(&xorc, &gn);
                    let score = fitness(&nxor_model).0;
                    gn = xor::scalar(&gn, score);
                    thread_update = xor::add(&thread_update, &gn);
                }
                txc.send(thread_update).unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        drop(tx);

        for r in rx {
            xor_update = xor::add(&xor_update, &r);
        }
        xor_update = xor::scalar(&xor_update, lr / ((ps * nt) as f32 * std));
        xor_model = xor::add(&xor_model, &xor_update);
    }
    let s = fitness(&xor_model);
    println!("Generation {}: {} {}", g, s.0, s.1);
    println!();
    xor::print(&xor_model);
    print_outputs(&xor_model);
    xor::save(&xor_model, "XorModel");
}
