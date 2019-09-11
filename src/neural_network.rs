use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::{Error, ErrorKind};

const PROPORTION_CHANGE_D_COST: f64 = 1.;

fn randomize(tab: &mut [f64]) {
    for b in tab.iter_mut() {
        *b = thread_rng().gen_range(-1., 1.);
    }
}

fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    let tmp = (-x).exp();
    tmp / (1. + tmp).powi(2)
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

#[derive(Serialize, Deserialize, Debug)]
pub struct NeuralNet {
    layer_sizes: Vec<usize>,
    biais: Vec<Vec<f64>>,
    weights: Vec<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct DCostNeuralNet {
    pub layer_sizes: Vec<usize>,
    pub biais: Vec<Vec<f64>>,
    pub weights: Vec<Vec<f64>>,
    pub first_layer_dcost: Vec<f64>,
}

impl DCostNeuralNet {
    pub fn new(layers: &[usize]) -> Self {
        let mut biais = Vec::new();
        let mut weights = Vec::new();
        let mut first_layer_dcost = Vec::new();

        first_layer_dcost.resize(layers[0], 0.);

        for i in 0..layers.len() - 1 {
            let mut layer_biais = Vec::new();
            let mut layer_weights = Vec::new();

            layer_weights.resize(layers[i] * layers[i + 1], 0.);
            weights.push(layer_weights);

            layer_biais.resize(layers[i + 1], 0.);
            biais.push(layer_biais);
        }

        Self {
            layer_sizes: Vec::from(layers),
            biais,
            weights,
            first_layer_dcost,
        }
    }
}

pub trait AverageDCost {
    fn average(&self) -> Result<DCostNeuralNet, Error>;
}

impl AverageDCost for Vec<DCostNeuralNet> {
    fn average(&self) -> Result<DCostNeuralNet, Error> {
        let mut average = if let Some(first) = self.first() {
            DCostNeuralNet::new(&first.layer_sizes)
        } else {
            return Err(Error::new(
                ErrorKind::Other,
                "Can't make an average of an empty DCostNeuralNet set",
            ));
        };

        for d_cost_neural_net in self {
            if d_cost_neural_net.biais.len() != average.biais.len()
                || d_cost_neural_net.weights.len() != average.weights.len()
            {
                return Err(Error::new(ErrorKind::Other, "Wrong network size"));
            }
            for i in 0..average.biais.len() {
                if average.biais[i].len() != d_cost_neural_net.biais[i].len() {
                    return Err(Error::new(ErrorKind::Other, "Wrong network size"));
                }
                for j in 0..average.biais[i].len() {
                    average.biais[i][j] += d_cost_neural_net.biais[i][j] / self.len() as f64;
                }
            }
            for i in 0..average.weights.len() {
                if average.weights[i].len() != d_cost_neural_net.weights[i].len() {
                    return Err(Error::new(ErrorKind::Other, "Wrong network size"));
                }
                for j in 0..average.weights[i].len() {
                    average.weights[i][j] += d_cost_neural_net.weights[i][j] / self.len() as f64;
                }
            }
            for i in 0..average.first_layer_dcost.len() {
                average.first_layer_dcost[i] += d_cost_neural_net.first_layer_dcost[i];
            }
        }
        Ok(average)
    }
}

impl NeuralNet {
    pub fn new(layers: &[usize]) -> Self {
        let mut biais = Vec::new();
        let mut weights = Vec::new();

        for i in 0..layers.len() - 1 {
            let mut layer_biais = Vec::new();
            let mut layer_weights = Vec::new();

            layer_weights.resize(layers[i] * layers[i + 1], 0.);
            randomize(&mut layer_weights);
            weights.push(layer_weights);

            layer_biais.resize(layers[i + 1], 0.);
            randomize(&mut layer_biais);
            biais.push(layer_biais);
        }

        Self {
            layer_sizes: Vec::from(layers),
            biais,
            weights,
        }
    }

    pub fn compute(&self, input: &[f64]) -> Result<Vec<Vec<f64>>, Error> {
        if input.len() != self.layer_sizes[0] {
            return Err(Error::new(
                ErrorKind::Other,
                format!(
                    "Wrong Input Size: {}, expected: {}",
                    input.len(),
                    self.layer_sizes[0]
                ),
            ));
        }

        let mut layer_values: Vec<Vec<f64>> = vec![Vec::from(input)];

        for i in 0..self.layer_sizes.len() - 1 {
            let mut next_layer = Vec::new();
            for j in 0..self.layer_sizes[i + 1] {
                next_layer.push(sigmoid(z(
                    &layer_values[i],
                    j,
                    &self.biais[i],
                    &self.weights[i],
                )));
            }
            layer_values.push(next_layer);
        }

        Ok(layer_values)
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
            d_cost_weight[neu_nb * layer_prec.len() + j] = layer_prec[j]
                * sigmoid_derivative(z(&layer_prec, neu_nb, &biais, &weight))
                * d_co_neu;
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

    pub fn backprop(&self, input: &[f64], output_goal: &[f64]) -> Result<DCostNeuralNet, Error> {
        let layer_values = self.compute(input)?;

        let d_co_prec: Vec<f64> = layer_values
            .last()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, x)| cost_function_derivative(*x, output_goal[i]))
            .collect();

        self.backprop_from_dcost(&layer_values, &d_co_prec)
    }

    pub fn backprop_from_dcost(
        &self,
        layer_values: &Vec<Vec<f64>>,
        d_co_prec: &Vec<f64>,
    ) -> Result<DCostNeuralNet, Error> {
        let mut d_co_prec: Vec<f64> = d_co_prec.clone();

        let mut d_cost_neural_net = DCostNeuralNet::new(&self.layer_sizes);

        for k in 1..self.layer_sizes.len() {
            let layer_nb = self.layer_sizes.len() - 1 - k;

            let mut new_d_co = Vec::new();
            new_d_co.resize(self.layer_sizes[layer_nb], 0.);

            for i in 0..d_co_prec.len() {
                Self::backprop_biais(
                    d_co_prec[i],
                    i,
                    &mut d_cost_neural_net.biais[layer_nb],
                    &layer_values[layer_nb],
                    &self.biais[layer_nb],
                    &self.weights[layer_nb],
                );
                Self::backprop_weight(
                    d_co_prec[i],
                    i,
                    &mut d_cost_neural_net.weights[layer_nb],
                    &layer_values[layer_nb],
                    &self.biais[layer_nb],
                    &self.weights[layer_nb],
                );
                Self::backprop_prec_layer(
                    d_co_prec[i],
                    i,
                    &mut new_d_co,
                    &layer_values[layer_nb],
                    &self.biais[layer_nb],
                    &self.weights[layer_nb],
                );
            }
            d_co_prec = new_d_co;
        }
        d_cost_neural_net.first_layer_dcost = d_co_prec;
        Ok(d_cost_neural_net)
    }

    pub fn apply_d_cost_net(&mut self, d_cost_neural_net: DCostNeuralNet) -> Result<(), Error> {
        if self.biais.len() != d_cost_neural_net.biais.len()
            || self.weights.len() != d_cost_neural_net.weights.len()
        {
            return Err(Error::new(ErrorKind::Other, "Wrong network size"));
        }
        for i in 0..self.biais.len() {
            if self.biais[i].len() != d_cost_neural_net.biais[i].len() {
                return Err(Error::new(ErrorKind::Other, "Wrong network size"));
            }
            for j in 0..self.biais[i].len() {
                self.biais[i][j] -= d_cost_neural_net.biais[i][j] * PROPORTION_CHANGE_D_COST;
            }
        }
        for i in 0..self.weights.len() {
            if self.weights[i].len() != d_cost_neural_net.weights[i].len() {
                return Err(Error::new(ErrorKind::Other, "Wrong network size"));
            }
            for j in 0..self.weights[i].len() {
                self.weights[i][j] -= d_cost_neural_net.weights[i][j] * PROPORTION_CHANGE_D_COST;
            }
        }
        Ok(())
    }
}
