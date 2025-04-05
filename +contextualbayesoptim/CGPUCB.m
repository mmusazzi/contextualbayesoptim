
classdef CGPUCB < contextualbayesoptim.ContextualAcquisitionFunction
    % Implements the CGP-UCB contextual acquisition function, see:
    % Krause Andreas and Cheng Soon Ong. “Contextual Gaussian Process Bandit Optimization” (2011)

    properties

        % Configuration parameters
        BetaEvolutionFuncHandle function_handle = @(iter) 4   % Handle to a function implementing the evolution
                                                              %   over iterations of the "Beta" parameter of CGP-UCB
        % State
        BetaUCB {mustBePositive}                              % "Beta" parameter of CGP-UCB
        Iteration {mustBeInteger, mustBeNonnegative}          % Iteration value used to compute BetaUCB

    end

    methods

        function obj = CGPUCB(acqFuncConfig)
            % Constructor

            if nargin == 0 || ~isfield(acqFuncConfig, 'BetaEvolutionFuncHandle') || isempty(acqFuncConfig.BetaEvolutionFuncHandle)
                warning('No BetaEvolutionFuncHandle given, using default value of constant BetaUCB = 4')
            else
                obj.BetaEvolutionFuncHandle = acqFuncConfig.BetaEvolutionFuncHandle;
            end

            obj.BetaUCB = [];
            obj.Iteration = 0;

        end

        function obj = updateState(obj, ~)
            % Updates BetaUCB using BetaEvolutionFuncHandle

            obj.Iteration = obj.Iteration + 1;
            obj.BetaUCB = obj.BetaEvolutionFuncHandle(obj.Iteration);

        end

       function afValues = compute(obj, gp, actions, contexts)
            % Returns a vector of contextual acquisition function values
            % computed at the specified locations
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

            X = [actions, contexts];

            % Compute posterior mean and standard deviation of the
            % observation GP
            [postMean, postStd_y] = gp.predict(X);

            % Compute standard deviation of the objective function model
            % Bound it at zero to account for floating point math artifacts
            postStd = sqrt(max(0, postStd_y.^2 - gp.Sigma^2));
            
            % Compute CGP-UCB values
            afValues = postMean + sqrt(obj.BetaUCB) * postStd;

        end

    end

end