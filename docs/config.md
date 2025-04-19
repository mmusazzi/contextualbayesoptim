# Configuration Parameters for `ContextualBayesianOptimizer`

This file documents all configuration parameters available for initializing a `ContextualBayesianOptimizer` object.

## Core Parameters

| Parameter              | Type     | Description                                                                 | Example           |
|------------------------|----------|-----------------------------------------------------------------------------|-------------------|
| `ActionSpaceDim`       | Integer  | Dimensionality of the action space                                          | `2`               |
| `ContextSpaceDim`      | Integer  | Dimensionality of the context space                                         | `1`               |
| `ActionSpaceLB`        | Vector   | Lower bounds for each action dimension                                      | `[0, 0]`          |
| `ActionSpaceUB`        | Vector   | Upper bounds for each action dimension                                      | `[5, 5]`          |
| `ContextSpaceLB`       | Vector   | Lower bounds for each context dimension                                     | `[0]`             |
| `ContextSpaceUB`       | Vector   | Upper bounds for each context dimension                                     | `[1]`             |

## Gaussian Process Configuration

| Parameter      | Type   | Description                                                         | Example           |
|----------------|--------|---------------------------------------------------------------------|-------------------|
| `KernelName`   | String | Specifies the kernel function used in Gaussian Process Regression   | `'ardmatern52'`   |

> **Note**: Possible values correspond to kernel functions supported by MATLAB’s `fitrgp` with the `'KernelFunction'` option. See [MATLAB GPR documentation](#reference-3)

## Acquisition Function Configuration

| Parameter                     | Type              | Description                                                                 | Example                      |
|------------------------------|-------------------|-----------------------------------------------------------------------------|------------------------------|
| `AcqFuncName`                | String            | Name of the acquisition function                                            | `'cgp-ucb'`                  |

> #### Possible values for `AcqFuncName`:
> - `'cgp-ucb'` – Contextual GP Upper Confidence Bound (See [Krause and Ong (2011)](#reference-2))

| Parameter                     | Type              | Description                                                                 | Example                      |
|------------------------------|-------------------|-----------------------------------------------------------------------------|------------------------------|
| `AcqFuncConfig`              | Struct            | Additional configuration for the acquisition function                      | `struct()`                     |

>#### `AcqFuncConfig` fields for `AcqFuncName = 'cgp-ucb'`:
>
>| Field                     | Type              | Description                                                                 | Example                      |
>|------------------------------|-------------------|-----------------------------------------------------------------------------|------------------------------|
>| `AcqFuncConfig.BetaEvolutionFuncHandle` | function_handle | Evolution over iterations of the 'BetaUCB' parameter of CGP-UCB  | `@(iter) 2 * log(iter .^ 2)`  |

## Acquisition function maximization (Auxiliary Optimizer parameters)

| Parameter             | Type     | Description                                                                  | Example      |
|----------------------|----------|------------------------------------------------------------------------------|--------------|
| `NumCandidates`       | Integer  | Number of random samples to generate for the global search stage             | `1e4`        |
| `NumLocalSearches`    | Integer  | Number of local search runs to refine promising candidates                   | `10`         |
| `MaxIterLocalSearch`  | Integer  | Maximum iterations allowed per local search                                  | `10`         |
| `RelTolLocalSearch`   | Double   | Relative tolerance used to determine convergence of the local search         | `1e-3`       |

## Notes

- All vectors (e.g., bounds) should be 1×D arrays where D matches the respective space dimensionality.

## See Also

- [README.md](../README.md) for usage examples
<a id="reference-2"></a>
- [Krause and Ong (2011)](https://proceedings.neurips.cc/paper/2011/file/f3f1b7fc5a8779a9e618e1f23a7b7860-Paper.pdf) for CGP-UCB
<a id="reference-3"></a>
- [MATLAB GPR documentation](https://it.mathworks.com/help/stats/regressiongp.html) for supported kernel functions

