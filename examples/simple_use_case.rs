/*!
This is a simple example designed to show how this library could be used.
It is built to be well documented and relativly easy to read. This example
is not written with the goal of showing off every function in the library,
for that you will have to refer to that function's own documentation.

### PLEASE NOTE:
Their are NO checks for deadends or regressions, this means that the program
may not get anywhere.

The goal of this program is to teach a `NeuralNetwork` to output somthing
within `MARGIN` of `TARGET` it will proccess a maximum of `GENERATIONS`
generations before for exiting the program and returning the closest result.
If any of the networks gets within `MARGIN` of `TARGET` then it will output
the `generation`, the `value` the `network` outputted, and the `network` itself.
*/

use smarty_pants::neural_network::*;
use std::time::Instant;

fn main() {
    /// This is our target, we will consider `NeuralNetworks` that get closer to this to be learning.
    const TARGET:f64 = 10.0;

    /// The margin of error for the networks.
    const MARGIN:f64 = 0.1;

    /// This is the `INPUT` for the networks, the goal being for them to "learn" how to take this number as
    /// an input and output `TARGET`. When they output a value within `MARGIN` of `TARGET` that `NeuralNetwork`
    /// will be considered to have reach the goal.
    const INPUT:f64 = 1.0;

    /// The maximum number of `GENERATIONS` that will be ran, less may be ran if a network gets within `margin` of
    /// the target before `GENERATIONS` number of generations are ran.
    const GENERATIONS:usize = 10_000;

    // Create a `Vector` containing all `NeuralNetwork`s in the current generation using `batch_mutate()` and `NeuralNetwork::new()`
    let mut networks:Vec<NeuralNetwork> = batch_mutate(5,0.25,&mut NeuralNetwork::new(1.0,1,3,1), true);

    // This stores the closest value found by the network, it defaults to negative infinity.
    let mut closest_value:f64 = f64::NEG_INFINITY;

    // Get the current instant so that it can later be used to time how long it took to finish/fail
    let time:Instant = Instant::now();

    // For `generation` in `GENERATIONS` perform `batch_run()` and check if the networks are getting closer.
    for generation in 0..GENERATIONS {
        // Run the networks using `INPUT` as an input and store the output in `output`
        let outputs:Vec<Vec<f64>> = batch_run(&mut networks, &vec![INPUT]);

        // The `closest_network` used for creating the next generation.
        let mut closest_network:usize = 0;

        // Loop through every value in `outputs` checking to see if any of the outputs are within `MARGIN` of `TARGET`
        // And use a range so that we can track the index of the output easily.
        for output in 0..outputs.len() {
            // Since the networks are only outputting a single value we can simply grab the first value of the `Vector`
            // and check if thats within `MARGIN` of `TARGET` using a range.
            if (TARGET-MARGIN..=TARGET+MARGIN).contains(&outputs[output][0]) {
                // If true then print the value found by the network, the network itself, the current generation, and exit from the program.
                println!("Finished in {:?}!\nGeneration: {:?}\nValue: {:?}\nNetwork: {{\nHiddenLayers: {:?}\nOutputLayer: {:?}\n}}", time.elapsed(),generation, outputs[output][0], networks[output].get_wheights(), networks[output].get_output_wheights());
                // Exit code 0 on Linux means no problem, on Windows however this should be 256 but that is outside the scope of this example.
                std::process::exit(0);
            } else {
                // If the `output` was not within `MARGIN` of `TARGET` then check if this value is closer to `TARGET` than the last `closest_value`.
                // and set `closest_value` to `output` if it is closer.
                if outputs[output][0] < TARGET && outputs[output][0] > closest_value {
                    closest_value = outputs[output][0];
                    closest_network = output;
                } else if outputs[output][0] > TARGET && outputs[output][0] < closest_value {
                    closest_value = outputs[output][0];
                    closest_network = output;
                }
            }
        }

        // Set all `networks` to various mutations of the `closest_network`.
        networks = batch_mutate(5, 0.25, &networks[closest_network], true);
    }

    // If we managed to get through `GENERATIONS` number of generations without getting within `MARGIN` of `TARGET` then output the `closest_value` we found.
    println!("Failed to get within `MARGIN` within {:?} number of generations, this is the `closest_value` obtained: {:?}. In {:?}", GENERATIONS, closest_value, time.elapsed());
}