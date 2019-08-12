mod neural_network;
mod read_idx;

const STEP_SIZE: usize = 50;

use neural_network::AverageDCost;

use rand::prelude::*;
use std::fs::File;
use std::io::prelude::*;

fn randomize_layer(tab: &mut [f64]) {
    for b in tab.iter_mut() {
        *b = thread_rng().gen_range(0., 1.);
    }
}
fn bet(output: &[f64]) -> u8 {
    let mut bet = (0, 0.);

    for i in 0..output.len() {
        if bet.1 < output[i] {
            bet = (i, output[i]);
        }
    }
    bet.0 as u8
}

fn calculate_loss(
    test_images: &[[f64; 784]],
    test_labels: &[u8],
    neural_network: &neural_network::NeuralNet,
) -> f64 {
    let mut tested = 0;
    let mut failed = 0;

    for k in 0..test_images.len() / 2 {
        let output = neural_network.compute(&test_images[k]).unwrap();
        let output = output.last().unwrap();
        tested += 1;
        if bet(output) != test_labels[k] {
            failed += 1;
        }
    }
    failed as f64 / tested as f64
}

fn save_neural_network(path: &str, neural_network: &neural_network::NeuralNet) {
    println!("Save to : {}", path);
    let encoded: Vec<u8> = bincode::serialize(&neural_network).unwrap();
    let mut buffer = File::create(path).unwrap();

    buffer.write_all(&encoded).unwrap();
    println!("Done !");
}

fn load_neural_network(path: &str) -> neural_network::NeuralNet {
    let mut buffer = File::open(path).unwrap();
    let mut neural_network = Vec::new();

    buffer.read_to_end(&mut neural_network).unwrap();
    bincode::deserialize(&neural_network).unwrap()
}

fn main() {
    let images = read_idx::read_images("./data/train-images-idx3-ubyte").unwrap();
    let labels = read_idx::read_labels("./data/train-labels-idx1-ubyte").unwrap();

    let test_images = read_idx::read_images("./data/t10k-images-idx3-ubyte").unwrap();
    let test_labels = read_idx::read_labels("./data/t10k-labels-idx1-ubyte").unwrap();

    let mut rng = thread_rng();
    let mut nn = neural_network::NeuralNet::new(&[784, 50, 28, 50, 784]);
    let mut nn = load_neural_network("model-03500");
    /*let mut random_layer = [0.0; 784];
    for n in 0..10000 {
        for i in 0..28 {
            for j in 0..28 {
                print!(
                    "{} ",
                    if images[n][i * 28 + j] > 0.66 {
                        "X"
                    } else if images[n][i * 28 + j] > 0.33 {
                        "-"
                    } else {
                        " "
                    }
                );
            }
            print!("\n");
        }
        let output = nn.compute(&images[n]).unwrap();
        let output = output.last().unwrap();

        let bet = bet(output);
        println!("bet: {}, {:#?}", bet, output);
    }*/
    for a in 0..images.len() / STEP_SIZE {
        println!("step {}", a);
        if a % 1 == 0 {
            let test_image = &test_images[rng.gen_range(0, test_images.len())];
            let output = nn.compute(test_image).unwrap();
            let output = output.last().unwrap();

            for i in 0..28 {
                for j in 0..28 {
                    print!(
                        "{} ",
                        if output[i * 28 + j] > 0.50 {
                            "X"
                        } else if output[i * 28 + j] > 0.40 {
                            "-"
                        } else {
                            " "
                        }
                    );
                }
                print!("\n");
            }
            println!("\n");
            for i in 0..28 {
                for j in 0..28 {
                    print!(
                        "{} ",
                        if test_image[i * 28 + j] > 0.66 {
                            "X"
                        } else if test_image[i * 28 + j] > 0.33 {
                            "-"
                        } else {
                            " "
                        }
                    );
                }
                print!("\n");
            }
        }

        if a % 100 == 0 {
            save_neural_network(&format!("./train/model-{:05}", a), &nn);
        }

        let mut d_cost_list = Vec::new();

        for k in 0..STEP_SIZE {
            let mut goal: [f64; 10] = [0.0; 10];
            goal[labels[a * STEP_SIZE + k] as usize] = 1.0;
            d_cost_list.push(
                nn.backprop(&images[a * STEP_SIZE + k], &images[a * STEP_SIZE + k])
                    .unwrap(),
            );
        }
        nn.apply_d_cost_net(d_cost_list.average().unwrap()).unwrap();
    }
    save_neural_network(
        &format!("./train/model-{:05}", images.len() / STEP_SIZE),
        &nn,
    );
}
