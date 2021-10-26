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
    
    
**Experience Replay**
Experience replay allows the RL agent to learn from past experience.

As with DQN in the previous project, DDPG also utilizes a replay buffer to gather experiences from each agent. Each experience is stored in a replay buffer as the agent interacts with the environment. 

The replay buffer contains a collection of experience tuples with the state, action, reward, and next state (s, a, r, s'). Each agent samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive algorithm could otherwise become biased by correlations between sequential experience tuples.

#### Network architecture

Two deep neural networks cpmprising of Actor Critic models are used;

The Actor Network receives as input 33 variables representing the state size, with two hidden layers each with 256 and 128 nodes. i used ReLU activation functions on the hidden layers and tanh on the output layers and generate output of 4 numbers representing the predicted best action for that observed state. That means, the Actor is used to approximate the optimal policy π deterministically.

The Critic Network receives as input 33 variables representing the observation space , also with two hidden layers each with 256 and 128 nodes.
The output of this network is the prediction of the target value based on the given state and the estimated best action.
That means the Critic calculates the optimal action-value function Q(s, a) by using the Actor's best-believed action.

**Hyperparameters** 

 Parameters | Value | Description
----------- | ----- | -----------
BUFFER_SIZE | int(1e5) | replay buffer size
BATCH_SIZE | 128 | minibatch size
GAMMA | 0.99 | discount factor
TAU | 1e-3 | for soft update of target parameters
LR_ACTOR | 1e-4 | learning rate of the actor
LR_CRITIC | 1e-4 | learning rate of the critic
WEIGHT_DECAY | 0 | L2 weight decay
NUM_AGENTS | 20 | Number of agents
fc1_units | 256 | Number of nodes in first hidden layer for actor
fc2_units | 128 | Number of nodes in second hidden layer for actor
fc1_units |256 | Number of nodes in first hidden layer for critic
fc2_units | 128 | Number of nodes in second hidden layer for critic

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
