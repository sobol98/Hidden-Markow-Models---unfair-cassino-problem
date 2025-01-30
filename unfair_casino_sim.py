import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict


    
class UnfairCasino:
    def __init__(self, fair_probabilities: Dict[int, float], cheat_probabilities: Dict[int, float], fair_to_cheat: float, cheat_to_fair: float):     
        '''        
            Args:
                fair_probabilities: Probabilities of rolling each number on a fair six-sided die 
                cheat_probabilities: Probabilities of rolling each number on a cheating six-sided die
                fair_to_fair: Probability of staying on a fair die
                loaded_to_loaded: Probability of staying on a cheating die  
                fair_to_loaded: Probability of transitioning from a fair die to a cheating die
                loaded_to_fair: Probability of transitioning from a cheating die to a fair die
        '''

        
        # States Fair: 0, Cheat: 1
        self.states = ['F', 'L']
        
        # Game die option 1-6
        self.observations = list(range(1, 7))

            
        self.transition_matrix = np. array([[
            1 - fair_to_cheat, fair_to_cheat],   #Fair to Fair, Fair to Cheat
            [cheat_to_fair, 1- cheat_to_fair]    #Cheat to Fair, Cheat to Cheat
        ])
    
        self.emission_matrix = np.array([
            list(fair_probabilities.values()),
            list(cheat_probabilities.values())
        ])
        
        
    
    def generate_sequence(self, lenght: int) -> Tuple[List[int], List[int]]:
        ''' 
            Generate a sequence of rolls and states
            Args:
                lenght: The number of rolls to generate
            Returns:
                A tuple of lists, the first list contains the rolls, the second list contains the states
        '''
        current_state = 0 #fair: 0 , cheat: 1
        observations = []
        true_states = []
        
        
        for _ in range(lenght):
            true_states.append(self.states[current_state])
            
            # Generate observation based on current state
            roll = np.random.choice(self.observations, p=self.emission_matrix[current_state])
            observations.append(roll)
            
            # Transition to next state
            current_state = np.random.choice([0, 1], p=self.transition_matrix[current_state])
        
        return observations, true_states

    
    
    
    def viterbi(self, observations: List[int]) -> List[int]:    
        """
        Use Viterbi algorithm to find most likely sequence of states
        
        Args:
            observations: List of dice rolls
            
        Returns:
            Most likely sequence of states
        """
            
        n_states = len(self.states)
        T = len(observations)
        
        # V [state, time]
        # holds the probability of the most probable path ending in state at time
        
        V = np.zeros((n_states, T))
        
        # B [state, time]
        #  holds the state at time-1 that gave the maximum probability of reaching state at time
        B = np.zeros((n_states, T), dtype=int)
        
        # using logs instead of multiplying probabilities to avoid underflow
        # 0.5 is the initial probability of starting in either state
        for state in range(n_states):
            V[state, 0] = np.log(0.5) + np.log(self.emission_matrix[state, observations[0]-1]) 
        
        for probe in range(1, T):
            for state in range(n_states):
                emissions = np.log(self.emission_matrix[state, observations[probe]-1])
                transitions = np.log(self.transition_matrix[:, state]) + V[:, probe-1]
                B[state, probe] = np.argmax(transitions)
                V[state, probe] = emissions + np.max(transitions)
        
        # Backtrack
        path = []
        current_state = np.argmax(V[:, -1])
    
        for probe in range(T-1, -1, -1):
            path.append(self.states[current_state])
            current_state = B[current_state, probe]
            
        return path[::-1]
                
            
    def forward_backward(self, observations: List[int]) -> List[Dict[str, float]]:
        """
        Use forward-backward algorithm to find probability of being in each state
        
        Args:
            observations: List of dice rolls
            
        Returns:
            List of dictionaries containing probabilities for each state at each time step
        """
        T = len(observations)
        n_states = len(self.states)
        
        forward = np.zeros((n_states, T))
        backward = np.zeros((n_states, T))

        
        # Forward
        for state in range(n_states):
            forward[state, 0] = np.log(0.5) + np.log(self.emission_matrix[state, observations[0]-1])
            
        for probe in range(1, T):
            for state in range(n_states):
                sum = -np.inf 
                for another_state in range(n_states):
                    sum = np.logaddexp(sum, forward[another_state, probe-1] + np.log(self.transition_matrix[another_state, state]))
                
                forward[state, probe] = np.log(self.emission_matrix[state, observations[probe]-1]) + sum
    
    

        # Backward
        backward[:, -1] = np.log(1.0)
        
        for probe in range(T-2, -1, -1):
            for state in range(n_states):
                transitions_log = []
                
                for another_state in range(n_states):
                    prob_log = backward[another_state, probe+1]
                    prob_log += np.log(self.transition_matrix[state, another_state])
                    prob_log += np.log(self.emission_matrix[another_state, observations[probe+1]-1])
                    transitions_log.append(prob_log)
                
                backward[state, probe] = np.logaddexp.reduce(transitions_log)
        

        path = []
            
        for probe in range(T):
            log_probs = forward[:, probe] + backward[:, probe]
            probs = np.exp(log_probs - np.logaddexp.reduce(log_probs))
            
            state = self.states[0] if probs[0] > probs[1] else self.states[1]
            path.append(state)
        
        return path  

        
    # ADDIDIONAL
    def baum_welch(self, observations: List[int], true_states: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Estimate HMM parameters based on known states and observations
        
        Args:
            observations: List of dice rolls
            true_states: List of true states ('F'/'L')
            
        Returns:
            Tuple of (transition_matrix, emission_matrix)
        '''

        n_symbols = len(self.observations)
        n_states = len(self.states)

        transition_counts = np.zeros((n_states, n_states))
        emission_counts = np.zeros((n_states, n_symbols))
            
        # transitions
        for i in range(len(true_states) - 1):
            current_idx = self.states.index(true_states[i])
            next_idx = self.states.index(true_states[i + 1])
            transition_counts[current_idx][next_idx] += 1
        
        # emissions
        for i in range(len(observations)):
            state_idx = self.states.index(true_states[i])
            observation = observations[i] - 1 
            emission_counts[state_idx][observation] += 1
        
        # to avoid zeros in the probability calculations
        epsilon = 1e-10
        
        
        transition_matrix = np.zeros((n_states , n_states))
        for i in range(n_states):
            row_sum = np.sum(transition_counts[i]) + epsilon * n_states
            for j in range(n_states):
                transition_matrix[i][j] = (transition_counts[i][j] + epsilon) / row_sum
        
        emission_matrix = np.zeros((n_states, n_symbols))
        for i in range(n_states):
            row_sum = np.sum(emission_counts[i]) + epsilon * n_symbols
            for j in range(n_symbols):
                emission_matrix[i][j] = (emission_counts[i][j] + epsilon) / row_sum
        
        
        return transition_matrix, emission_matrix
        
    
        
        
def calculate_accuracy(true_states: List[str], predicted_states: List[str]) -> dict:
    """
    Calculate various accuracy metrics for predictions
    """
    
    total = len(true_states)
    correct = 0
    fair_correct = 0
    fair_total = 0
    loaded_correct = 0
    loaded_total = 0
    
    for true, pred in zip(true_states, predicted_states):
        if true == pred:
            correct += 1
            if true == 'F':
                fair_correct += 1
            else:
                loaded_correct += 1
                
        if true == 'F':
            fair_total += 1
        else:
            loaded_total += 1
    
    return {
        'total_accuracy': correct / total,
        'fair_accuracy': fair_correct / fair_total if fair_total > 0 else 0,
        'loaded_accuracy': loaded_correct / loaded_total if loaded_total > 0 else 0
    }
        
        

        
def print_sequence(rolls: List[int], states: List[str], viterbi_states: List[str], forward_backward_states: List[str]):
    """
    Print the sequence in specified format
    format_type: continuous string
    """

    print("\nRoll sequence:")
    print("Roll    " + "".join(str(roll) for roll in rolls))
    print("Die     " + "".join(state for state in states))
    print("Viterbi " + "".join(v_state for v_state in viterbi_states))
    print("F-B     " + "".join(fb_state for fb_state in forward_backward_states))
    
            

            
def run_simulation(n_rolls: int, ):
    
    fair_probabilities = {1: 1/6, 
                          2: 1/6, 
                          3: 1/6,
                          4: 1/6, 
                          5: 1/6, 
                          6: 1/6}
    
    cheat_probabilities = {1: 1/10, 
                           2: 1/10, 
                           3: 1/10, 
                           4: 1/10, 
                           5: 1/10, 
                           6: 1/2}
    
    fair_to_cheat = 0.05
    cheat_to_fair = 0.1
    
    
    
    casino = UnfairCasino(fair_probabilities, cheat_probabilities, fair_to_cheat, cheat_to_fair)
    
    rolls, true_states = casino.generate_sequence(n_rolls)   
    viterbi_states = casino.viterbi(rolls)
    forward_backward_states = casino.forward_backward(rolls)
    
    viterbi_metrics = calculate_accuracy(true_states, viterbi_states)
    fb_metrics = calculate_accuracy(true_states, forward_backward_states)
    
    
    # ADDIDIONAL
    estimated_transition, estimated_emission = casino.baum_welch(rolls, true_states)


    print_sequence(rolls, true_states, viterbi_states, forward_backward_states)
    

    # ----------------- Accuracy Metrics -----------------
    print("\nAccuracy metrics:")
    print("Viterbi Algorithm:")
    print(f"Total accuracy:     {viterbi_metrics['total_accuracy']:.2%}")
    print(f"Fair die accuracy:  {viterbi_metrics['fair_accuracy']:.2%}")
    print(f"Cheat die accuracy: {viterbi_metrics['loaded_accuracy']:.2%}")
    
    print("\nForward-Backward Algorithm:")
    print(f"Total accuracy:     {fb_metrics['total_accuracy']:.2%}")
    print(f"Fair die accuracy:  {fb_metrics['fair_accuracy']:.2%}")
    print(f"Cheat die accuracy: {fb_metrics['loaded_accuracy']:.2%}")
    # ----------------- Accuracy Metrics -----------------

    
    
    
    # ADDIDIONAL
    # ----------------- Baum Welch | Parameter Estimation -----------------
    print("\n------------------------------------------------------------------")
    print("\nEstimated Parameters:")
    print("\nTransition Matrix (Original -> Estimated):")
    print("Original:")
    print(casino.transition_matrix)
    print("\nEstimated:")
    print(estimated_transition)

    print("\nEmission Matrix (Original -> Estimated):")
    print("Original:")
    print(casino.emission_matrix)
    print("\nEstimated:")
    print(estimated_emission)
    # ----------------- Baum Welch | Parameter Estimation -----------------

    
    
    
    # ----------------- Plotting -----------------
    plt.figure(figsize=(12, 6))
    
    for i in range(len(true_states)):
        if true_states[i] == 'L':
            plt.axvspan(i, i+1, color='gray', alpha=0.3)
    

    # Plot true states
    true_values = [1 if state == 'F' else 0 for state in true_states]
    plt.plot(range(len(true_states)), true_values, 'r--', label='True state', linewidth=2)
    
    # Plot Viterbi predictions as binary (1 for Fair, 0 for Cheat)
    y_values = [1 if state == 'F' else 0 for state in viterbi_states]
    plt.plot(range(len(viterbi_states)), y_values, 'b-', label='Viterbi prediction')
    
    
    # Plot Forward-Backward predictions as binary (1 for Fair, 0 for Cheat)
    fb_values = [1 if state == 'F' else 0 for state in forward_backward_states]
    plt.plot(range(len(forward_backward_states)), fb_values, 'g-.', label='Forward-Backward')

    
    plt.xlabel('Roll number')
    plt.ylabel('P(Fair)')
    plt.grid(True)
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.show()
    
    

if __name__ == "__main__":
    
    sequence_length = 1000
    run_simulation(sequence_length)