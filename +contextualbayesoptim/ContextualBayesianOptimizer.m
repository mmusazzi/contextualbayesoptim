
classdef ContextualBayesianOptimizer < handle
    % Implements a class to perform Contextual Bayesian Optimization

    properties

    % Configuration parameters

        % CBO parameters
        ActionSpaceDim {mustBeInteger, mustBePositive}  % Dimension of action space
        ContextSpaceDim {mustBeInteger, mustBePositive} % Dimension of context space
        ActionSpaceLB {mustBeNumeric}                   % Action space lower bound (1 x ActionSpaceDim row vector)
        ActionSpaceUB {mustBeNumeric}                   % Action space upper bound (1 x ActionSpaceDim row vector)
        ContextSpaceLB {mustBeNumeric}                  % Context space lower bound (1 x ContextSpaceDim row vector)
        ContextSpaceUB {mustBeNumeric}                  % Context space upper bound (1 x ContextSpaceDim row vector)
        KernelName {mustBeTextScalar} = ''              % Name of the kernel function to be used
        AcqFuncName {mustBeTextScalar} = ''             % Name of the contextual acquisition function to be used
        AcqFuncConfig struct                            % Struct with contextual acquisition function configuration parameters

        % Auxiliary optimizer (auxGlobalMaxSearch) parameters
        NumCandidates {mustBeInteger, mustBePositive}       % Number of initial random candidates
        NumLocalSearches {mustBeInteger, mustBePositive}    % Number of local optimizer runs
        MaxIterLocalSearch {mustBeInteger, mustBePositive}  % Maximum iterations per local optimizer run
        RelTolLocalSearch {mustBePositive}                  % Local optimizer stopping tolerance

    % State

        Observations   % Struct containing past observations
        GP             % RegressionGP object modeling the objective function and observation process
        AcqFunc        % Contextual acquisition function object

    end

    methods

        function obj = ContextualBayesianOptimizer(config)
            % Constructor

            obj.ActionSpaceDim = config.ActionSpaceDim;
            obj.ContextSpaceDim = config.ContextSpaceDim;
            obj.ActionSpaceLB = config.ActionSpaceLB;
            obj.ActionSpaceUB = config.ActionSpaceUB;
            obj.ContextSpaceLB = config.ContextSpaceLB;
            obj.ContextSpaceUB = config.ContextSpaceUB;
            obj.KernelName = config.KernelName;
            obj.AcqFuncName = config.AcqFuncName;
            obj.AcqFuncConfig = config.AcqFuncConfig;
            obj.NumCandidates = config.NumCandidates;
            obj.NumLocalSearches = config.NumLocalSearches;
            obj.MaxIterLocalSearch = config.MaxIterLocalSearch;
            obj.RelTolLocalSearch = config.RelTolLocalSearch;

            obj.Observations = struct('X', [], 'y', []);
            obj.GP = [];

            switch lower(obj.AcqFuncName)
                case 'cgp-ucb'
                    obj.AcqFunc = contextualbayesoptim.CGPUCB(obj.AcqFuncConfig);
                otherwise
                    error(['Unknown contextual acquisition function name: %s.\nValid names: ' ...
                        '\n\t%s'], obj.AcqFuncName, 'cgp-ucb');
            end

        end

        function obj = addObservations(obj, actions, contexts, results)
            % Adds new observations (as action-context-result sets) to
            % the stored ones
            %
            %   dS  :   dimension of action space
            %   dZ  :   dimension of context space
            %   N   :   number of new observations
            %
            % Input arguments:
            %   actions:  (N x dS) matrix with action values as rows
            %   contexts: (N x dZ) matrix with context values as rows
            %   results:  (N x 1)  vector with observed results

            N = size(actions, 1);
            dS = size(actions, 2);
            dZ = size(contexts, 2);

            if (size(contexts, 1) ~= N || size(results, 1) ~= N)
                error('Inconsistent dimensions for actions-contexts-results')
            end
            if (dS ~= obj.ActionSpaceDim)
                error('Size of actions is not compatible with ActionSpaceDim')
            end
            if (dZ ~= obj.ContextSpaceDim)
                error('Size of contexts is not compatible with ContextSpaceDim')
            end
            if (size(results, 2) ~= 1)
                error('Incompatible dimensions for results: it must be a column vector')
            end

            X = [actions, contexts];
            obj.Observations.X = [obj.Observations.X; X];
            obj.Observations.y = [obj.Observations.y; results];

        end

        function obj = updateGP(obj)
            % Updates the GP model for the objective function and
            % observation process by estimating the hyperparameters (kernel
            % parameters, prior mean, observation noise standard deviation)
            % from observations
            
            obj.GP = fitrgp(obj.Observations.X, obj.Observations.y, ...
                            'KernelFunction', obj.KernelName, ...
                            'FitMethod', 'exact', ...        % Exact GP regression with MLE of hyperparams based on all available observations
                            'BasisFunction', 'constant', ... % Constant prior mean function
                            'PredictMethod', 'exact');       % Exact GP inference based on all available observations
        end

        function obj = updateAcqFuncState(obj)
            % Performs the operations needed to update the relevant state variables
            % before being able to evaluate the contextual acquisition function
            % at a new iteration

            obj.AcqFunc.updateState(obj.GP);

        end

        function s_next = computeNextActionGivenContext(obj, context)
            % Computes the next action by maximizing the contextual
            % acquisition function restricted to the given context
            %
            % Input arguments
            %   context: (1 x ContextSpaceDim) vector with context at which
            %            the acquisition function will be evaluated
            %
            % Return values
            %   s_next: (1 x ActionSpaceDim) vector with action to perform

            if (any(size(context) ~= [1, obj.ContextSpaceDim]))
                error('Size of context is not consistent: it must be a (1 x ContextSpaceDim) vector')
            end
            
            % Define acquisition function contextualized in the current context.
            % Repeat copies of context to allow for vectorialization w.r.t.
            % the actions s
            contextualAF = @(s) obj.AcqFunc.compute(obj.GP, s, repmat(context, size(s, 1), 1));
            
            s_next = contextualbayesoptim.auxGlobalMaxSearch(contextualAF, ...
                                obj.ActionSpaceLB, obj.ActionSpaceUB, ...
                                obj.NumCandidates, obj.NumLocalSearches, ...
                                obj.MaxIterLocalSearch, obj.RelTolLocalSearch);

        end

        function [postMean, postStd] = computePost(obj, actions, contexts)
            % Returns the posterior mean and standard deviation of the
            % objective function GP evaluated at the specified locations
            %
            %   dS  :   dimension of action space
            %   dZ  :   dimension of context space
            %   N   :   number of points at which to compute the posterior
            %
            % Input arguments:
            %   actions:  (N x dS) matrix with action values as rows
            %   contexts: (N x dZ) matrix with context values as rows
            %
            % Return values:
            %   postMean: (N x 1) vector with values of the posterior mean
            %   postStd:  (N x 1) vector with values of the posterior
            %             standard deviation of the objective function GP

            if (size(actions, 1) ~= size(contexts, 1))
                error('Inconsistent dimensions for actions-contexts')
            end
            if (size(actions, 2) ~= obj.ActionSpaceDim)
                error('Size of actions is not compatible with ActionSpaceDim')
            end
            if (size(contexts, 2) ~= obj.ContextSpaceDim)
                error('Size of contexts is not compatible with ContextSpaceDim')
            end

            X = [actions, contexts];

            % Compute posterior mean and standard deviation of the
            % observation GP
            [postMean, postStd_y] = obj.GP.predict(X);

            % Compute standard deviation of the objective function model
            % Bound it at zero to account for floating point math artifacts
            postStd = sqrt(max(0, postStd_y.^2 - obj.GP.Sigma^2));

        end

        function afValues = computeAcqFuncValues(obj, actions, contexts)
            % Returns a vector of contextual acquisition function values at
            % the specified locations at the current state of the CBO
            %
            %   dS  :   dimension of action space
            %   dZ  :   dimension of context space
            %   N   :   number of points at which to compute the
            %           contextual acquisition function
            %
            % Input arguments:
            %   actions:  (N x dS) matrix with action values as rows
            %   contexts: (N x dZ) matrix with context values as rows
            %
            % Return values:
            %   afValues: (N x 1) vector with contextual acquisition
            %             function values

            if (size(actions, 1) ~= size(contexts, 1))
                error('Inconsistent dimensions for actions-contexts')
            end
            if (size(actions, 2) ~= obj.ActionSpaceDim)
                error('Size of actions is not compatible with ActionSpaceDim')
            end
            if (size(contexts, 2) ~= obj.ContextSpaceDim)
                error('Size of contexts is not compatible with ContextSpaceDim')
            end

            afValues = obj.AcqFunc.compute(obj.GP, actions, contexts);

        end

    end

end
