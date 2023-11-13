# Readme

## Introduction

Multi-agent pathfinding (MAPF) is an abstract model for the navigation of multiple robots in warehouse automation, where multiple robots plan collision-free paths from the start to goal positions. Reinforcement learning (RL) has been employed to develop partially observable distributed MAPF policies that can be scaled to any number of agents. 
However, RL-based MAPF policies often get agents stuck in deadlock due to warehouse automation's dense and structured obstacles.
This paper proposes a novel hybrid MAPF policy, RDE, based on switching among the RL-based MAPF policy, the Distance heat map (DHM)-based policy and the Escape policy. 
The RL-based policy is used for coordination among agents.
In contrast, when no other agents are in the agent's field of view, it can get the next action by querying the DHM. 
The escape policy that randomly selects valid actions can help agents escape the deadlock.

## RDE

The code of RDE is improved on the code of DHC and DCC.

### DHC

Distributed reinforcement learning with communication for decentralized multi-agent pathfinding.

https://github.com/ZiyuanMa/DHC

### DCC

Original code and trained model for paper: 

Learning Selective Communication for Multi-Agent Path Finding

https://arxiv.org/abs/2109.05413

