# contextualbayesoptim

A MATLAB implementation of **Contextual Bayesian Optimization (CBO)** using Gaussian Process Regression (GPR).

CBO is a variant of Bayesian Optimization that explicitly accounts for the influence of external variables, known as *context*, on the objective function. The library provides a modular and user-friendly interface for solving black-box data-driven optimization problems while incorporating contextual information to improve decision-making.

This implementation is particularly suited for:

- **Sequential decision-making under contextual influence**, as described in [Krause and Ong (2011)](#reference-1)
- **Optimal mapping reconstruction**, as discussed in [Ginsbourger et al. (2014)](#reference-2)

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
	- [Sequential decision-making problems](#sequential-decision-making-problems)
    - [Optimal mapping reconstruction problems](#optimal-mapping-reconstruction-problems)
3. [Examples](#examples)
4. [References](#references)
5. [License](#license)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/mmusazzi/contextualbayesoptim.git
```
### 2. Add the library to your MATLAB path

Open MATLAB and run the following commands:

```matlab
addpath(genpath('path/to/contextualbayesoptim'))
savepath
```
Replace `'path/to/contextualbayesoptim'` with the actual path to the cloned repository on your system.

### 3. Check dependencies

This library uses MATLAB’s built-in functionality for Gaussian Process Regression.

Make sure you have the following:

- MATLAB **R2017b** or newer (recommended)
- **Statistics and Machine Learning Toolbox**

You can verify the toolbox is installed by running `ver` and checking for `Statistics and Machine Learning Toolbox` in the list.

## Usage

The usage of the `ContextualBayesianOptimizer` class will be demonstrated through two minimal examples of typical problems.

### Sequential decision-making problems

This type of problem can be summarized as follows: at each iteration, we must choose an action to perform based on the value of an uncontrollable but measurable context. The goal is to achieve a high result value, which will be measured after executing the chosen action. A deeper insight into this topic can be found in [Krause and Ong (2011)](#reference-1).

#### 1. Create a `ContextualBayesianOptimizer` object

```matlab
% Define dimensions of the action and context spaces
config.ActionSpaceDim = 2;
config.ContextSpaceDim = 1;

% Define lower and upper bounds for the action and context spaces
config.ActionSpaceLB = [0, 0];
config.ActionSpaceUB = [5, 5];
config.ContextSpaceLB = [0];
config.ContextSpaceUB = [1];

% Define GPR kernel and contextual acquisition function
config.KernelName = 'ardmatern52';
config.AcqFuncName = 'cgp-ucb';
config.AcqFuncConfig.BetaEvolutionFuncHandle = @(iter) 2 * log(iter + 1);

% Define parameters for the auxiliary optimizer (auxGlobalMaxSearch), used to maximize the acquisition function
config.NumCandidates = 1e4;
config.NumLocalSearches = 10;
config.MaxIterLocalSearch = 10;
config.RelTolLocalSearch = 1e-3;

% Create the optimizer object
cbo = contextualbayesoptim.ContextualBayesianOptimizer(config);
```

#### 2. Add initially available observations

```matlab
% Example observations
actions = [1, 2; 2, 3];  % Actions (N x ActionSpaceDim matrix)
contexts = [0.1; 0.2];   % Contexts (N x ContextSpaceDim matrix)
results = [0.5; 0.8];    % Results (N x 1 vector)

% Add observations to the optimizer
cbo.addObservations(actions, contexts, results);
```

#### 3. Update the Gaussian Process model based on current observations

```matlab
cbo.updateGP();
```

#### 4. Update the acquisition function state

```matlab
cbo.updateAcqFuncState();
```

#### 5. Measure current context

```matlab
context = measureContext(); % Placeholder for the actual context measurement function
```

#### 6. Compute action to perform and measure result

```matlab
action = cbo.computeNextActionGivenContext(context);
result = performAction(action); % Placeholder for the actual result measurement function
```

#### 7. Add the new observation to the optimizer

```matlab
cbo.addObservations(action, context, result);
```

#### 8. Go to 3.

### Optimal mapping reconstruction problems

In this case, the objective is to estimate the optimal mapping between context and action. That is, for each possible value of the context variables, we aim to find the corresponding action values that maximize the result. Unlike the sequential decision-making problem, we can control the context and choose the sequence of action-context pairs to sample to improve our estimation of the optimal mapping. This problem is discussed in [Ginsbourger et al. (2014)](#reference-2).

#### 1. TO-DO

TO-DO

## Examples

- `sequential_decision_cgpucb.m`: Example of using the `ContextualBayesianOptimizer` class for **sequential decision-making** in a 1D action × 1D context space, utilizing the **CGP-UCB** acquisition function.

## References

<a id="reference-1"></a>
[1] Krause, A., & Ong, C. S. (2011). *Contextual Gaussian Process Bandit Optimization*. Advances in Neural Information Processing Systems (NeurIPS). [PDF](https://proceedings.neurips.cc/paper/2011/file/f3f1b7fc5a8779a9e618e1f23a7b7860-Paper.pdf)

<a id="reference-2"></a>
[2] Ginsbourger, D., Baccou, J., Chevalier, C., Perales, F., Garland, N., & Monerie, Y. (2014). *Bayesian Adaptive Reconstruction of Profile Optima and Optimizers*. SIAM/ASA Journal on Uncertainty Quantification, 2(1), 490–510. [DOI:10.1137/130949555](https://doi.org/10.1137/130949555)

## License

This project is licensed under the **GNU General Public License v3.0**.  
You are free to use, modify, and distribute this software under the terms of the GPL.

See the [LICENSE](LICENSE) file for the full license text.