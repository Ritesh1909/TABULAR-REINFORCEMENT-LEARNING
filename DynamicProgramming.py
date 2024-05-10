
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import matplotlib.pyplot as plt
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: Add own code
        
        # We need argmax of Q(s,a) in a variable.
        a_best = np.argmax(self.Q_sa[s]) 
        return a_best
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        
        # Initialize expected avg reward variable
        expected_avg_reward = 0
        # We need to iterate over each state present in n_states which hold height * width space state
        for s_new in range(self.n_states):
            
           # Equation 1 implementation  
           expected_avg_reward = expected_avg_reward + p_sas[s_new] * (r_sas[s_new] + self.gamma * np.max(self.Q_sa[s_new]))
           # Maximum absolute error after each step is in Q_value_iteration()
           
        # Update Q-value
        self.Q_sa[s, a] = expected_avg_reward
        
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
 
    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    
    #Tracking iteration with variable i 
    i = 0 
    # Threshold is given. We stop iterating when we reach convergence (error < Threshold)
    while True:
        
        MaximumChange = 0

        for s in range(env.n_states):
            for a in range(env.n_actions):
               
                # Get transition probabilities and rewards for state s and action a from Envirnoment.py
                p_sas, r_sas = env.model(s, a)
                # Storing current estimate
                x = QIagent.Q_sa[s,a]
                # Update Q-value using the update function
                QIagent.update(s, a, p_sas, r_sas)
                # Updating Maximum
                MaximumChange = max(MaximumChange, abs(x - QIagent.Q_sa[s, a] ))

        i+=1
        # To see render at the beginning
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=False,step_pause=0.2)
        # print the maximum absolute error after each full sweep 
        print(" Maximum absolute error for full sweep {} is {}".format(i,MaximumChange))
        # Check for convergence to end loop.
        if MaximumChange < threshold:
            break 
    
    env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=False,step_pause=0.9)
    print("Q-value iteration, iteration {}, max error {}".format(i,MaximumChange))
 
    # return QIagent
    return QIagent 

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    # Unpacking the tuple returned from Q_value_iteration
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # view optimal policy
    done = False
    s = env.reset()
    i = 0
    total_reward = 0
    
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.6)
        s = s_next
        i += 1
        total_reward += r 
        
    # TO DO: Compute mean reward per timestep under the optimal policy
    
    # i tracked the no. of iteration in Q_value_iteration
    mean_reward_per_timestep = total_reward / i
    print(total_reward)
    print(i)
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    
if __name__ == '__main__':
    experiment()


