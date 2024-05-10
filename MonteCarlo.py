
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

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        T_ep = len(rewards)
        G_t = 0   
        # Loop backwards 
        for iteration in reversed(range(T_ep)): 
            G_t = rewards[iteration] + self.gamma * G_t
            self.Q_sa[states[iteration], actions[iteration]] += self.learning_rate * (G_t - self.Q_sa[states[iteration], actions[iteration]]) 
            
def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your Monte Carlo RL algorithm here!
    iteration = 0
   
    while iteration < n_timesteps: 
        # Reset to start state     
        s = env.reset()
        # initialising arrays to send to update() 
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
            if (eval_interation) % eval_interval == 0:
                mean_return = pi.evaluate(eval_env)
                eval_returns.append(mean_return)
                eval_timesteps.append(eval_interation)
                print(f"Iteration {eval_interation}: Mean Return {mean_return}")
               
            if done:
                break    
            
        pi.update(states, actions, rewards)      
        
        # To prevent overtraining and actually doing n_timesteps.
        iteration = iteration + counter_timestep  
        
    
    # To see returned values and check workings of the code.
    print(np.array(eval_returns))
    print(np.array(eval_timesteps)) 
     
    return np.array(eval_returns), np.array(eval_timesteps) 

    
def test():
    n_timesteps = 100000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    
            
if __name__ == '__main__':
    test()
