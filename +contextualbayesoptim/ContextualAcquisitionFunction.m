
classdef (Abstract) ContextualAcquisitionFunction < handle
    % Implements an interface for contextual acquisition functions in the
    % setting of Contextual Bayesian Optimization

    methods (Abstract)

        updateState(obj, gp)
            % Performs the operations needed to update the relevant state variables
            % before being able to evaluate the contextual acquisition function
            % at a new iteration
            %
            % Input arguments:
            %   gp:  RegressionGP object containing observations and fitted
            %        GP model for the objective function and observation
            %        model

        afValues = compute(obj, gp, actions, contexts)
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

    end

end