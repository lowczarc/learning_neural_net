mod read_idx;

use rand::prelude::*;

fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    let tmp = (-x).exp();
    tmp / (1. + tmp).powi(2)
}

fn randomize(tab: &mut [f64]) {
    for b in tab.iter_mut() {
        *b = thread_rng().gen_range(-1., 1.);
    }
}

fn randomize_layer(tab: &mut [f64]) {
    for b in tab.iter_mut() {
        *b = thread_rng().gen_range(0., 1.);
    }
}

fn cost_function(neurone_value: f64, goal: f64) -> f64 {
    (neurone_value - goal).powi(2)
}

fn cost_function_derivative(neurone_value: f64, goal: f64) -> f64 {
    2. * (neurone_value - goal)
}

fn z(layer1: &[f64], neurone_nb: usize, biais: &[f64], weight: &[f64]) -> f64 {
    let mut res = biais[neurone_nb];
    for i in 0..layer1.len() {
        res += weight[neurone_nb * layer1.len() + i] * layer1[i];
    }
    res
}

fn layer_compute(layer1: &[f64], layer2: &mut [f64], biais: &[f64], weight: &[f64]) {
    for i in 0..layer2.len() {
        layer2[i] = sigmoid(z(&layer1, i, &biais, &weight));
    }
}

fn backprop_biais(
    d_co_neu: f64,
    neurone_nb: usize,
    d_cost_biais: &mut [f64],
    layer_prec: &[f64],
    biais: &[f64],
    weight: &[f64],
) {
    d_cost_biais[neurone_nb] =
        sigmoid_derivative(z(&layer_prec, neurone_nb, &biais, &weight)) * d_co_neu;
}

fn backprop_weight(
    d_co_neu: f64,
    neu_nb: usize,
    d_cost_weight: &mut [f64],
    layer_prec: &[f64],
    biais: &[f64],
    weight: &[f64],
) {
    for j in 0..layer_prec.len() {
        d_cost_weight[neu_nb * layer_prec.len() + j] =
            layer_prec[j] * sigmoid_derivative(z(&layer_prec, neu_nb, &biais, &weight)) * d_co_neu;
    }
}

fn backprop_prec_layer(
    d_co_neu: f64,
    neu_nb: usize,
    d_cost_prec_layer: &mut [f64],
    layer_prec: &[f64],
    biais: &[f64],
    weight: &[f64],
) {
    for j in 0..layer_prec.len() {
        d_cost_prec_layer[j] += weight[neu_nb * layer_prec.len() + j]
            * sigmoid_derivative(z(&layer_prec, neu_nb, &biais, &weight))
            * d_co_neu;
    }
}

fn main() {
    let mut layer_value1: [f64; 784] = [0.0; 784];
    let mut layer_value2: [f64; 16] = [0.0; 16];
    let mut layer_value3: [f64; 16] = [0.0; 16];
    let mut layer_value4: [f64; 10] = [0.0; 10];

    let mut biais2: [f64; 16] = [0.0; 16];
    let mut biais3: [f64; 16] = [0.0; 16];
    let mut biais4: [f64; 10] = [0.0; 10];

    let mut weight1_2: [f64; 12544] = [0.0; 12544];
    let mut weight2_3: [f64; 256] = [0.0; 256];
    let mut weight3_4: [f64; 160] = [0.0; 160];

    randomize(&mut biais2);
    randomize(&mut biais3);
    randomize(&mut biais4);

    randomize(&mut weight1_2);
    randomize(&mut weight2_3);
    randomize(&mut weight3_4);

    randomize_layer(&mut layer_value1);

    let images = read_idx::read_images("./data/train-images-idx3-ubyte").unwrap();
    let labels = read_idx::read_labels("./data/train-labels-idx1-ubyte").unwrap();

    for n in 0..1 {
        for i in 0..28 {
            for j in 0..28 {
                print!(
                    "{} ",
                    if images[n][i * 28 + j] > 0.66 {
                        "X"
                    } else if images[n][i * 28 + j] > 0.33 {
                        "x"
                    } else {
                        " "
                    }
                );
            }
            print!("\n");
        }
        println!("{}", labels[n]);
    }

    
    layer_compute(&images[0], &mut layer_value2, &biais2, &weight1_2);
    layer_compute(&layer_value2, &mut layer_value3, &biais3, &weight2_3);
    layer_compute(&layer_value3, &mut layer_value4, &biais4, &weight3_4);

    let mut goal: [f64; 10] = [0.0; 10];
    goal[labels[0] as usize] = 1.0;

    for i in 0..layer_value4.len() {
        println!(
            "{}: {:.3}, goal: {:.3}, cost: {:.3}",
            i,
            layer_value4[i],
            goal[i],
            cost_function(layer_value4[i], goal[i])
        );
    }

    let mut d_cost_biais4 = [0.0; 10];
    let mut d_cost_weight3_4 = [0.0; 160];
    let mut d_cost_layer3 = [0.0; 16];
    for i in 0..layer_value4.len() {
        backprop_biais(
            cost_function_derivative(layer_value4[i], goal[i]),
            i,
            &mut d_cost_biais4,
            &layer_value3,
            &biais4,
            &weight3_4,
        );
        backprop_weight(
            cost_function_derivative(layer_value4[i], goal[i]),
            i,
            &mut d_cost_weight3_4,
            &layer_value3,
            &biais4,
            &weight3_4,
        );
        backprop_prec_layer(
            cost_function_derivative(layer_value4[i], goal[i]),
            i,
            &mut d_cost_layer3,
            &layer_value3,
            &biais4,
            &weight3_4,
        );
    }

    let mut d_cost_biais3 = [0.0; 16];
    let mut d_cost_weight2_3 = [0.0; 256];
    let mut d_cost_layer2 = [0.0; 16];
    for i in 0..layer_value3.len() {
        backprop_biais(
            d_cost_layer3[i],
            i,
            &mut d_cost_biais3,
            &layer_value2,
            &biais3,
            &weight2_3,
        );
        backprop_weight(
            d_cost_layer3[i],
            i,
            &mut d_cost_weight2_3,
            &layer_value2,
            &biais3,
            &weight2_3,
        );
        backprop_prec_layer(
            d_cost_layer3[i],
            i,
            &mut d_cost_layer2,
            &layer_value2,
            &biais3,
            &weight2_3,
        );
    }

    let mut d_cost_biais2 = [0.0; 16];
    let mut d_cost_weight1_2 = [0.0; 12544];
    for i in 0..layer_value2.len() {
        backprop_biais(
            d_cost_layer2[i],
            i,
            &mut d_cost_biais2,
            &layer_value3,
            &biais2,
            &weight1_2,
        );
        backprop_weight(
            d_cost_layer2[i],
            i,
            &mut d_cost_weight1_2,
            &layer_value1,
            &biais2,
            &weight1_2,
        );
    }

    for i in 0..biais4.len() {
        biais4[i] -= d_cost_biais4[i];
    }

    for i in 0..biais3.len() {
        biais3[i] -= d_cost_biais3[i];
    }

    for i in 0..biais2.len() {
        biais2[i] -= d_cost_biais2[i];
    }
    println!("biais modify\n");

    layer_compute(&layer_value3, &mut layer_value4, &biais4, &weight3_4);

    for i in 0..layer_value4.len() {
        println!(
            "{}: {:.3}, goal: {:.3}, cost: {:.3}",
            i,
            layer_value4[i],
            goal[i],
            cost_function(layer_value4[i], goal[i])
        );
    }

    for i in 0..weight3_4.len() {
        weight3_4[i] -= d_cost_weight3_4[i];
    }

    for i in 0..weight2_3.len() {
        weight2_3[i] -= d_cost_weight2_3[i];
    }

    for i in 0..weight1_2.len() {
        weight1_2[i] -= d_cost_weight1_2[i];
    }

    println!("weight modify\n");
    layer_compute(&layer_value3, &mut layer_value4, &biais4, &weight3_4);

    for i in 0..layer_value4.len() {
        println!(
            "{}: {:.3}, goal: {:.3}, cost: {:.3}",
            i,
            layer_value4[i],
            goal[i],
            cost_function(layer_value4[i], goal[i])
        );
    }
    println!("step ++\n");
}
