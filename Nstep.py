#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        
        T_ep = len(rewards)

        for iteration in range(T_ep):
    
            G_t = 0  
            m = min(n, T_ep - iteration)
            
            # To prevent going out of bound.
            if iteration + m < T_ep:
                # To check if terminal, rewards array is used.
                if rewards[iteration + m] > -1 : 
                    # n-step target without bootstrap 
                    for i in range(m): 
                        G_t += (self.gamma**i) * rewards[iteration + i]  
                else:
                    # n-step return with bootstrap for non-terminal state, n-step target
                    for i in range(m): 
                        G_t += (self.gamma**i) * rewards[iteration + i]
                    G_t = G_t + (self.gamma**m) * np.max(self.Q_sa[states[iteration + m], :]) #( : becouse we want max from array)  
            # Update Q Table
            self.Q_sa[states[iteration], actions[iteration]] += self.learning_rate * (G_t - self.Q_sa[states[iteration], actions[iteration]])
        pass
         
def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your n-step Q-learning algorithm here!
    iteration = 0

    while iteration < n_timesteps: 
        # Reset to start state  
        s = env.reset()
        # Initialising arrays to send to update() 
        states = [s]
        actions = []
        rewards = []
        # To count the steps in inner-loop
        counter_timestep = 0
        for iteration2 in range(max_episode_length):
            # Select action using the specified policy
            a = pi.select_action(s, policy, epsilon, temp)
            actions.append(a)
            # Take action and observe the next state and reward
            s, r, done = env.step(a)
            states.append(s)
            rewards.append(r)
            
            counter_timestep = counter_timestep + 1 
            
            eval_interation = iteration + iteration2 + 1
            if eval_interation % eval_interval == 0:
                mean_return = pi.evaluate(eval_env)
                eval_returns.append(mean_return)
                eval_timesteps.append(eval_interation)
                print(f"Iteration {eval_interation}: Mean Return {mean_return}")
            
            if done: 
                break   
            
        pi.update(states, actions, rewards, done, n)  
        # To prevent overtraining and actually doing n_timesteps.
        iteration = iteration + counter_timestep

        # if plot:
        #     env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
    
    # To see returned values and check workings of the code.    
    print(np.array(eval_returns))
    print(np.array(eval_timesteps)) 
    
    return np.array(eval_returns), np.array(eval_timesteps) 
    
def test():
    n_timesteps = 100000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
           
if __name__ == '__main__':
    test()
