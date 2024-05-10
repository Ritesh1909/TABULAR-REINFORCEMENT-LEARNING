
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

class SarsaAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,a_next,done):
        # TO DO: Add own code
        # Implementating of equation 8 and 9 
        Gt = r + self.gamma * self.Q_sa[s_next, a_next]
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (Gt - self.Q_sa[s, a])
        pass

        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your SARSA algorithm here!
    s = env.reset()
    a = pi.select_action(s, policy, epsilon, temp)
    
    for iteration in range(n_timesteps):

        # Take action and observe the next state and reward
        s_next, r, done = env.step(a)
        # Select action using the specified policy
        a_next = pi.select_action(s_next, policy, epsilon, temp)
        # Update the Q-values by using update() 
        pi.update(s, a, r, s_next, a_next, done)
    
        s = s_next
        a = a_next
        
        if done:
            print("reached goal state in %d iterration"%iteration)
            s = env.reset()
            a = pi.select_action(s, policy, epsilon, temp)  
            
        # Evaluate the policy at regular intervals of 100 in our case    
        eval_interation = iteration + 1 
        if  (eval_interation)  % eval_interval == 0:
            mean_return = pi.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(eval_interation)  
            
        # if plot:    
        #     env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.01) # Plot the Q-value estimates during SARSA execution

    return np.array(eval_returns), np.array(eval_timesteps) 


def test():
    n_timesteps = 30000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    eval_returns, eval_timesteps = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    # To see returned values and check workings of the code.
    print(eval_returns,eval_timesteps)
            
    
if __name__ == '__main__':
    test()
