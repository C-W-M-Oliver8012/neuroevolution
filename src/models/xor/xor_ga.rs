use rand::prelude::*;
use rand_distr::StandardNormal;
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

fn mutate(a: &xor::XorModel, mutation_rate: f32, std: f32) -> xor::XorModel {
    let mut b = a.clone();

    // fc1
    // weights
    for i in 0..a.fc1.weights.value.len() {
        let m = thread_rng().gen::<f32>() % 1.0;

        if m <= mutation_rate {
            let w: f32 = thread_rng().sample(StandardNormal);
            b.fc1.weights.value[i] += w * std;
        }
    }

    // bias
    for i in 0..a.fc1.bias.value.len() {
        let m = thread_rng().gen::<f32>() % 1.0;

        if m <= mutation_rate {
            let w: f32 = thread_rng().sample(StandardNormal);
            b.fc1.bias.value[i] += w * std;
        }
    }

    // fc2
    //weights
    for i in 0..a.fc2.weights.value.len() {
        let m = thread_rng().gen::<f32>() % 1.0;

        if m <= mutation_rate {
            let w: f32 = thread_rng().sample(StandardNormal);
            b.fc2.weights.value[i] += w * std;
        }
    }

    // bias
    for i in 0..a.fc2.bias.value.len() {
        let m = thread_rng().gen::<f32>() % 1.0;

        if m <= mutation_rate {
            let w: f32 = thread_rng().sample(StandardNormal);
            b.fc2.bias.value[i] += w * std;
        }
    }
    b
}

fn crossover(a: &xor::XorModel, b: &xor::XorModel, mutation_rate: f32, std: f32) -> xor::XorModel {
    let mut c = a.clone();

    // fc1
    // weights
    for i in 0..a.fc1.weights.value.len() {
        let r = thread_rng().gen::<f32>() % 1.0;
        let m = thread_rng().gen::<f32>() % 1.0;
        if r < 0.5 {
            c.fc1.weights.value[i] = a.fc1.weights.value[i];
        } else {
            c.fc1.weights.value[i] = b.fc1.weights.value[i];
        }

        if m <= mutation_rate {
            let w: f32 = thread_rng().sample(StandardNormal);
            c.fc1.weights.value[i] += w * std;
        }
    }

    // bias
    for i in 0..a.fc1.bias.value.len() {
        let r = thread_rng().gen::<f32>() % 1.0;
        let m = thread_rng().gen::<f32>() % 1.0;
        if r < 0.5 {
            c.fc1.bias.value[i] = a.fc1.bias.value[i];
        } else {
            c.fc1.bias.value[i] = b.fc1.bias.value[i];
        }

        if m <= mutation_rate {
            let w: f32 = thread_rng().sample(StandardNormal);
            c.fc1.bias.value[i] += w * std;
        }
    }

    // fc2
    //weights
    for i in 0..a.fc2.weights.value.len() {
        let r = thread_rng().gen::<f32>() % 1.0;
        let m = thread_rng().gen::<f32>() % 1.0;
        if r < 0.5 {
            c.fc2.weights.value[i] = a.fc2.weights.value[i];
        } else {
            c.fc2.weights.value[i] = b.fc2.weights.value[i];
        }

        if m <= mutation_rate {
            let w: f32 = thread_rng().sample(StandardNormal);
            c.fc2.weights.value[i] += w * std;
        }
    }

    // bias
    for i in 0..a.fc2.bias.value.len() {
        let r = thread_rng().gen::<f32>() % 1.0;
        let m = thread_rng().gen::<f32>() % 1.0;
        if r < 0.5 {
            c.fc2.bias.value[i] = a.fc2.bias.value[i];
        } else {
            c.fc2.bias.value[i] = b.fc2.bias.value[i];
        }

        if m <= mutation_rate {
            let w: f32 = thread_rng().sample(StandardNormal);
            c.fc2.bias.value[i] += w * std;
        }
    }
    c
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
    let population = init_population(10000);
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
                /*
                let mut parent_one_index = thread_rng().gen::<usize>() % info.2.len();
                let mut parent_two_index = thread_rng().gen::<usize>() % info.2.len();

                // parent 1
                for _ in 0..5 {
                    let r = thread_rng().gen::<usize>() % info.2.len();
                    if old_population[r].score.0 > old_population[parent_one_index].score.0 {
                        parent_one_index = r;
                    }
                }

                // parent 2
                for _ in 0..5 {
                    let r = thread_rng().gen::<usize>() % info.2.len();
                    if old_population[r].score.0 > old_population[parent_two_index].score.0 {
                        parent_two_index = r;
                    }
                }

                info.2[i].individual = crossover(&old_population[parent_one_index].individual, &old_population[parent_two_index].individual, 1.0, 0.05);
                */
                let mut parent_one_index = thread_rng().gen::<usize>() % info.2.len();

                for _ in 0..5 {
                    let r = thread_rng().gen::<usize>() % info.2.len();
                    if old_population[r].score.0 >= old_population[parent_one_index].score.0 {
                        parent_one_index = r;
                    }
                }

                info.2[i].individual = mutate(&old_population[parent_one_index].individual, 1.0, 0.05);
            }
        }

        info = score_population(&info.2);
    }

    println!("Generation {}: bf: {} af: {}", g, info.0, info.1);

    xor::print(&info.2[info.3].individual);
    print_outputs(&info.2[info.3].individual);
}
