
function x_max = auxGlobalMaxSearch(f, lb, ub, nCandidates, nLocalSearches, maxIterLocalSearch, relTolLocalSearch)
    % Looks for the global maximizer of a function inside given bounds.
    % Initially, the function is evaluated in a number of candidate points,
    % selected randomly from a uniform distribution between the bounds. The
    % specified number of best candidates among these is used as starting point
    % for local refinements. The best overall point is returned.
    %
    % Input arguments:
    %   f : function handle pointing to the function to maximize
    %   lb : (1 x d) row vector with the lower bound of the search space
    %   ub : (1 x d) row vector with the upper bound of the search space
    %   nCandidates : number of initial random candidates
    %   nLocalSearches : number of local optimizer runs
    %   maxIterLocalSearch : maximum iterations per local optimizer run
    %   relTolLocalSearch : local optimizer stopping tolerance
    %
    % Return values:
    %   x_max : global optimizer

    % Generate random points from a uniform distribution
    x0_candidates = repmat(lb, nCandidates, 1) + rand(nCandidates, size(lb, 2)) .* repmat(ub - lb, nCandidates, 1);

    % Use the best nLocalSearches points as starting points for local refinement
    [~, idx] = sort(f(x0_candidates), 'descend');
    x0 = x0_candidates(idx(1:nLocalSearches), :);
    
    % Perform local unconstrained searches on an artificially bounded
    % function
    [xs, fsNeg] = bayesoptim.fminsearch(@boundedNegFunc, x0, ...
                  optimset('MaxIter', maxIterLocalSearch, 'TolX', Inf, ...
                           'TolFun', relTolLocalSearch));
    
    % Choose the best maximizer overall
    [~, idx] = min(fsNeg);
    x_max = xs(idx, :);
    
    function y = boundedNegFunc(x)

        % Points outside the bounds are assigned infinite value to
        % artifically constrain the optimization
        y = inf(size(x, 1), 1);

        % Indices of the points which comply with the bounds
        rows = all(x > lb & x < ub, 2);

        % For the points which comply with the bounds, compute the
        % corresponding value of the function to be maximized, with opposite
        % sign since the local search algorithm is a minimization one
        if any(rows)
            y(rows) = -f(x(rows, :));
        end

    end

end