# Unfair Casino HMM Simulation

This project implements a Hidden Markov Model (HMM) simulation of an unfair casino where a fair die can be secretly switched with a loaded one.

## Overview

The simulation models a casino game with two states:
- Fair die (F) - equal probability (1/6) for each number
- Loaded die (L) - biased probability favoring 6 (1/2 for 6, 1/10 for others)

The switching between dice happens with probabilities:
- Fair to Loaded: 0.05
- Loaded to Fair: 0.1 

## Features

### Core Functionality
- Sequence generation with true states
- Viterbi algorithm for finding most likely state sequence
- Forward-Backward algorithm for state probability estimation
- Baum-Welch parameter estimation from known states

### Visualization and Analysis
- State sequence visualization
- Dice roll and state prediction display
- Accuracy metrics comparison between algorithms
- Parameter estimation comparison

## Usage

```python
# Create casino simulation
casino = UnfairCasino(fair_probabilities, cheat_probabilities, 
                     fair_to_cheat=0.05, cheat_to_fair=0.1)

# Generate sequence
rolls, true_states = casino.generate_sequence(n_rolls)

# Predict states using different algorithms
viterbi_states = casino.viterbi(rolls)
fb_states = casino.forward_backward(rolls)

# Estimate parameters
estimated_transition, estimated_emission = casino.baum_welch(rolls, true_states)
