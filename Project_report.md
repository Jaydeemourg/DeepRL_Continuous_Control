## 1. Project Goal

In this project, I developed a reinforcement learning (RL) agent that controls a robotic arm using Unity's [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. The goal is to get a robotic arm to maintain contact with the green spheres.

A reward of +0.1 is provided for each timestep that the agent's hand is in the goal location. The goal of the agent is to maintain its position at the target location for as many time steps as possible.

In order to solve the environment, our agent must achieve a score of +30 averaged across all 20 agents for 100 consecutive episodes.

## 2. Solution Approach
Here are the steps taken in developing an agent that solves the environment.

* Identify the state and action space.
* Select ddpg algorithm and implementing it.
* Train the agent

#### Identify the state and action space
The state space space has 33 dimensions corresponding to the position, rotation, velocity, and angular velocities of the robotic arm. There are two sections of the arm analogous to those connecting the shoulder and elbow (i.e., the humerus), and the elbow to the wrist (i.e., the forearm) on a human body.

Each action is a vector with four numbers, corresponding to the torque applied to the two joints (shoulder and elbow). Every element in the action vector must be a number between -1 and 1, making the action space continuous.

#### Select ddpg algorithm and implementing it
the action space of the task is continuous i.e. there's an unlimited range of possible action values to control the robotic arm. DDPG is suitable for this type of action space

**DDPG algorithm (Deep Deterministic Policy Gradient):**
this algorithm consists of two networks;
- a Critic that measures how good the action taken is using value-based method. The value function maps each state action pair to a value which quantifies how it is. The value function calculates what is the maximum expected future reward given a state and an action.

- an Actor that controls how the agent behaves using policy-based method.  The policy is optimized without using a value function. This is useful when the action space is continuous or stochastic.

an update is made at each step using TD Learning, this is done without waiting until the end of the episode. The Critic observes the agent's action and provides feedback in order to update the policy and be better at playing that game.

**Batch Normalization**
This is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.
 
Similar to the exploding gradient issue mentioned above, running computations on large input values and model parameters can inhibit learning. Batch normalization addresses this problem by scaling the features to be within the same range throughout the model and across different environments and units. In additional to normalizing each dimension to have unit mean and variance, the range of values is often much smaller, typically between 0 and 1. We implementeed on first layer of fully connected layers of both actor and critic models
    
    
**Experience Replay**
Experience replay allows the RL agent to learn from past experience.

As with DQN in the previous project, DDPG also utilizes a replay buffer to gather experiences from each agent. Each experience is stored in a replay buffer as the agent interacts with the environment. 

The replay buffer contains a collection of experience tuples with the state, action, reward, and next state (s, a, r, s'). Each agent samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive algorithm could otherwise become biased by correlations between sequential experience tuples.

**Hyperparameters** 

 Parameters | Value | Description
----------- | ----- | -----------
BUFFER_SIZE | int(1e6) | replay buffer size
BATCH_SIZE | 128 | minibatch size
GAMMA | 0.99 | discount factor
TAU | 1e-3 | for soft update of target parameters
LR_ACTOR | 1e-3 | learning rate of the actor
LR_CRITIC | 1e-3 | learning rate of the critic
WEIGHT_DECAY | 0 | L2 weight decay
NUM_AGENTS | 1 | Number of agents
fc1_units | 400 | Number of nodes in first hidden layer for actor
fc2_units | 300 | Number of nodes in second hidden layer for actor
fc1_units |400 | Number of nodes in first hidden layer for critic
fc2_units | 300 | Number of nodes in second hidden layer for critic

#### Train the agent 
The ddpg agent is then traimed for 200 episodes until the performance threshold is realized.  

## 3. Performance for DDPG Agent.
Environment was solved in 106 episodes. Average score: 30.019
![alt text](https://github.com/Jaydeemourg/DeepRL_Continuous_Control/blob/main/score_per_episode_plot.png)

## 4. Future Improvements 
- **Add prioritized experience replay**  Rather than selecting experience tuples randomly, prioritized replay selects experiences based on a priority value that is correlated with the magnitude of error. This can improve learning by increasing the probability that rare and important experience vectors are sampled.


**Experiment with other algorithms like**
- PPO [paper](https://arxiv.org/abs/1707.06347) 
- (A2C) [paper](https://arxiv.org/abs/1602.01783v2)
