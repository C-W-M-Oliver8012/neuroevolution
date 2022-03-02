use rand::prelude::*;
use crate::matrix;
use crate::models::xor;

#[derive(Clone)]
struct Individual {
    individual: xor::XorModel,
    score: (f32, bool)
}

fn init_population(pop_size: usize) -> Vec<Individual> {
    let mut population: Vec<Individual> = Vec::new();

    for _ in 0..pop_size {
        population.push(Individual {
            individual: xor::new_gaussian_noise(),
            score: (0.0, false)
        });
    }

    population
}

fn score_population(population: &Vec<Individual>) -> (f32, f32, Vec<Individual>, usize) {
    let mut best_fitness: f32 = -10000.0;
    let mut average_fitness: f32 = 0.0;
    let mut updated_pop: Vec<Individual> = Vec::new();
    let mut best_index: usize = 0;

    for i in 0..population.len() {
        let score = fitness(&population[i].individual);
        updated_pop.push(Individual {
            individual: population[i].individual.clone(),
            score,
        });

        average_fitness += score.0;
        if score.0 > best_fitness {
            best_fitness = score.0;
            best_index = i;
        }
    }
    average_fitness /= population.len() as f32;

    (best_fitness, average_fitness, updated_pop, best_index)
}

fn mutate(a: &xor::XorModel, std: f32) -> xor::XorModel {
    let mut b = a.clone();

    let mut mutation = xor::new_gaussian_noise();
    mutation = xor::scalar(&mutation, std);

    b = xor::add(&b, &mutation);
    b
}

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

pub fn train() {
    let population = init_population(1000);
    let mut info = score_population(&population);

    let mut g = 0;
    while info.0 < 4.0 {
        println!("Generation {}: bf: {} af: {}", g, info.0, info.1);
        g += 1;
        // perform a single generation
        let old_population = info.2.clone();
        for i in 0..info.2.len() {
            if i == 0 {
                info.2[i].individual = old_population[info.3].individual.clone();
            } else {
                let mut parent_one_index = thread_rng().gen::<usize>() % info.2.len();

                for _ in 0..5 {
                    let r = thread_rng().gen::<usize>() % info.2.len();
                    if old_population[r].score.0 >= old_population[parent_one_index].score.0 {
                        parent_one_index = r;
                    }
                }

                info.2[i].individual = mutate(&old_population[parent_one_index].individual, 0.05);
            }
        }

        info = score_population(&info.2);
    }

    println!("Generation {}: bf: {} af: {}", g, info.0, info.1);

    xor::print(&info.2[info.3].individual);
    print_outputs(&info.2[info.3].individual);
}
