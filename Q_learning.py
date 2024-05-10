
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

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        # TO DO: Add own code
        # Implementating of equation 4 and 5 of back-up estimate/target and tabular learning update 
        Gt = r + self.gamma * np.max(self.Q_sa[s_next,:])  # Q-learning target (: becouse we want max from array)
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (Gt - self.Q_sa[s, a]) 
        pass

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    # Start state
    s = env.reset()
    
    # TO DO: Write your Q-learning algorithm here!
    for iteration in range(n_timesteps):
        
        # Select action using the specified policy
        a = agent.select_action(s, policy, epsilon, temp)
        # Take action and observe the next state and reward
        s_new, r, done = env.step(a)
        # Update the Q-values by using update() 
        agent.update(s, a, r, s_new, done) 
        
        s = s_new 
           
        # Check for terminate 
        if done:   
            # reset the environment to start state
            print("reached goal state in %d iterration"%iteration)
            s = env.reset() 
                 
        # Evaluate the policy at regular intervals of 100 in our case
        eval_interation = iteration + 1 
        if  (eval_interation)  % eval_interval == 0:
            mean_return = agent.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(eval_interation)
        
        # if plot: 
        #     env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.001)    

    return np.array(eval_returns), np.array(eval_timesteps)   

def test():
    
    n_timesteps = 30000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    
    # To see returned values and check workings of the code.
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()
