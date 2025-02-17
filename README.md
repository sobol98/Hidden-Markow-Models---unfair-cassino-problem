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

change number of sequences in main function, and then:

```bash
python3 unfrair_casino_sim.py
