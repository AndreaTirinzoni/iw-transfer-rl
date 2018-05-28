# Importance Weighted Transfer of Samples in Reinforcement Learning

This repository contains an implementation of the Importance Weighted Fitted Q-Iteration (IWFQI) algorithm proposed by [1], together with instructions on how to reproduce the experiments proposed in the paper.Here is some text.

### Abstract

We consider the transfer of experience samples (i.e., tuples < s, a, s', r >) in reinforcement learning (RL), collected from a set of source tasks to improve the learning process in a given target task. Most of the related approaches focus on selecting the most relevant source samples for solving the target task, but then all the transferred samples are used without considering anymore the discrepancies between the task models. In this paper, we propose a model-based technique that automatically estimates the relevance (importance weight) of each source sample for solving the target task. In the proposed approach, all the samples are transferred and used by a batch RL algorithm to solve the target task, but their contribution to the learning process is proportional to their importance weight. By extending the results for importance weighting provided in supervised learning literature, we develop a finite-sample analysis of the proposed batch RL algorithm. Furthermore, we empirically compare the proposed algorithm to state-of-the-art approaches, showing that it achieves better learning performance and is very robust to negative transfer, even when some source tasks are significantly different from the target task.

### Repository Structure

The repository is organized in the following folders:

 - examples: some examples on how to run the code. Currently, only an example of the plot functions is provided;
 - experiments: the set of scripts for running the experiments proposed in the paper;
 - tests: some tests of our code (to be ignored);
 - trlib: a general library for the transfer of samples in reinforcement learning (described in the following section).

### TRLIB: A Library for the Transfer of Samples in RL

TRLIB is a small library for the transfer of samples in RL that we implemented to support our experiments. Currently, focus is given on transferring into Fitted Q-Iteration (FQI) [2]. The following algorithms are implemented:

 - Fitted Q-Iteration (FQI) [2];
 - Importance Weighted Fitted Q-Iteration (IWFQI) [1];
 - The relevance-based transfer (RBT) algorithm of [3];
 - The shared-dynamics transfer (SDT) algorithm of [4].

Besides the algorithms, TRLIB provides different environments (currently only the ones proposed in the paper), different policies, and several other useful utilities.

### Requirements

```
Python 3
numpy
scikit-learn
joblib
matplotlib
gym
```

### How to Reproduce our Experiments

The scripts for all experiments are under experiments/ICML2018. In particular, there are four folders, each corresponding to one of the four domains proposed in the paper: puddle world with shared-dynamics, puddle world with puddle-based dynamics, acrobot, and the control of a water reservoir (dam). Specifically, each folder contains:

 - The source data used in the experiment as .pkl files. Each file contains the samples from a specific source task together with the corresponding predictions from the source/target Gaussian Processes (see Section 5 of the paper for more info);
 - The scripts to run the algorithms (run_*.py). In particular, run_algorithms.py runs FQI, RBT, and SDT (the latter only if applicable), while run_wfqi runs IWFQI. For the puddle world experiments, run_wfqi_ideal runs IWFQI with ideal importance weights;
 - Other files to generate new source data (generate_data.py, learn_source_policy.py, and others).

In order to reproduce our experiments, simply run the desired script file. If you wish to generate new source data, each folder contains a script (generate_data.py) which runs a given policy to collect samples, fits the GPs, and saves everything in .pkl format. The policy to be used can be learned using learn_source_policy.py or can be manually coded (e.g., in the acrobot and dam environments).

### References

[1] Andrea Tirinzoni, Andrea Sessa, Matteo Pirotta, and Marcello Restelli. Importance Weighted Transfer of Samples in Reinforcement Learning. International Conference on Machine Learning. 2018.  
[2] Ernst, Damien, Pierre Geurts, and Louis Wehenkel. "Tree-based batch mode reinforcement learning." Journal of Machine Learning Research 6.Apr (2005): 503-556.  
[3] Lazaric, Alessandro, Marcello Restelli, and Andrea Bonarini. "Transfer of samples in batch reinforcement learning." Proceedings of the 25th international conference on Machine learning. ACM, 2008.  
[4] Laroche, Romain, and Merwan Barlier. "Transfer Reinforcement Learning with Shared Dynamics." AAAI. 2017.  
