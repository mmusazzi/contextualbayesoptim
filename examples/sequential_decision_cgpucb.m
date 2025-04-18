%% Example: Sequential Decision-Making with CGP-UCB
%
% This script demonstrates the use of contextualbayesoptim for a sequential
% decision-making problem in a 1D action × 1D context space, using the 
% CGP-UCB acquisition function and a synthetic environment.

clear
close all

%% Setup and initialization
% Define problem settings, domain bounds, and configure the CBO optimizer

% Number of randomly selected observations before starting CBO
numInitialSamples = 5;

% Number of subsequent observations and iterations of CBO
numIterationsCBO = 40;

% Domain
config.ActionSpaceDim = 1;
config.ContextSpaceDim = 1;
config.ActionSpaceLB = 0;
config.ActionSpaceUB = 15;
config.ContextSpaceLB = -5;
config.ContextSpaceUB = 10;

% CBO hyperparameters
config.KernelName = 'ardmatern52';  % GP kernel
config.AcqFuncName = 'cgp-ucb';     % Acquisition function
config.AcqFuncConfig.BetaEvolutionFuncHandle = @(iter) 2 * log(iter .^ 2); % Exploration parameter schedule

% Optimization settings for acquisition function maximization
config.NumCandidates = 1e4;
config.NumLocalSearches = 10;
config.MaxIterLocalSearch = 10;
config.RelTolLocalSearch = 1e-3;

% Create optimizer object
cbo = contextualbayesoptim.ContextualBayesianOptimizer(config);

% Aliases for readability
dim_act = cbo.ActionSpaceDim;
dim_ctx = cbo.ContextSpaceDim;
lb_act = cbo.ActionSpaceLB;
ub_act = cbo.ActionSpaceUB;
lb_ctx = cbo.ContextSpaceLB;
ub_ctx = cbo.ContextSpaceUB;

% --- Define simulated environment ---

% Define context simulation function
measureContext = @() lb_ctx + rand(1, dim_ctx) .* (ub_ctx - lb_ctx);

% Define noisy result from environment
obsNoiseStd = 0.1;  % Observation noise std dev
trueResult = @(action, context) cos(action / 4) .* cos(context / 4) + ...
                                3 * sin(action / 4) .* sin(context / 4) + ...
                                0.5 * cos(action .* context / 16);
measureResult = @(action, context) trueResult(action, context) + obsNoiseStd * randn(1);

%% Random initial sampling
% Collect an initial set of observations using random actions

for iter = 1:numInitialSamples

    % Measure current context
    context = measureContext();

    % Select a random action within bounds
    action = lb_act + rand(1, dim_act) .* (ub_act - lb_act);

    % Execute action and observe noisy result
    result = measureResult(action, context);

    % Add observation to the model
    cbo.addObservations(action, context, result);

    % Print sample to the console
    printInitialSample(iter, action, context, result);
end

% Train GP on initial data
cbo.updateGP();

% Initialize contextual acquisition function state
cbo.updateAcqFuncState();

%% Contextual Bayesian Optimization iterations

for iter = 1:numIterationsCBO

    % Measure current context
    context = measureContext();

    % Select action using acquisition function (CGP-UCB)
    action = cbo.computeNextActionGivenContext(context);

    % Execute action and observe noisy result
    result = measureResult(action, context);

    % Update the CBO model with new data
    cbo.addObservations(action, context, result);
    cbo.updateGP();
    cbo.updateAcqFuncState();

    % Print sample to the console
    printIterationCBO(iter, action, context, result);
end

% Plot the results of CBO
finalPlots(cbo, measureResult);

%% Local functions

% Display initial random samples in a formatted table
function printInitialSample(iter, action, context, result)

    if iter == 1
        fprintf('\nInitial randomly sampled observations:\n');
        fprintf('\n%10s | %-15s | %-15s | %-15s\n', 'Sample', 'Action', 'Context', 'Result');
        fprintf(repmat('-', 1, 65));
        fprintf('\n');
    end

    fprintf('%10d | %-15.4e | %-15.4e | %-15.4e\n', iter, action, context, result);
end

% Display CBO iterations in a formatted table
function printIterationCBO(iter, action, context, result)

    if iter == 1
        fprintf('\nCBO iterations:\n');
        fprintf('\n%10s | %-15s | %-15s | %-15s\n', 'Iteration', 'Action', 'Context', 'Result');
        fprintf(repmat('-', 1, 65));
        fprintf('\n');
    end

    fprintf('%10d | %-15.4e | %-15.4e | %-15.4e\n', iter, action, context, result);
end

% Final visualization of the result function, GP mean, and uncertainty
function finalPlots(cbo, trueResultFuncHandle)

    plotResolution = 50;

    lb_act = cbo.ActionSpaceLB;
    ub_act = cbo.ActionSpaceUB;
    lb_ctx = cbo.ContextSpaceLB;
    ub_ctx = cbo.ContextSpaceUB;

    act_points = linspace(lb_act, ub_act, plotResolution);
    ctx_points = linspace(lb_ctx, ub_ctx, plotResolution);
    [A, C] = meshgrid(act_points, ctx_points);

    R = trueResultFuncHandle(A, C);     % Ground truth values

    % GP posterior mean and std dev
    [posteriorMean, posteriorStd] = cbo.computePost(A(:), C(:));
    posteriorMean = reshape(posteriorMean, size(A));
    posteriorStd = reshape(posteriorStd, size(A));

    % Determine plot color limits
    cmin = min(min(R(:)), min(posteriorMean(:)));
    cmax = max(max(R(:)), max(posteriorMean(:)));

    % Plot true result function
    fig_posterior = figure();
    fig_posterior.Position = [400, 200, 800, 250];
    legendLabels = {};
    legendEntries = [];

    ax1 = axes('Position', [0.075, 0.15, 0.28, 0.625]);
    contourf(ax1, C, A, R, 10)
    colormap(gca, 'hot')
    clim([cmin, cmax])
    colorbar
    hold on
    h1 = plot(ax1, cbo.GP.X(:, 2), cbo.GP.X(:, 1), ...
        'o', 'MarkerSize', 5, 'MarkerEdgeColor', [0 1 0], ...
        'MarkerFaceColor', [1 1 1], 'LineWidth', 1.5);
    legendLabels{end + 1} = 'Observations';
    legendEntries(end + 1) = h1;
    hold off
    grid on
    xlabel('Context')
    ylabel('Action')
    title('True result function')

    % Plot GP posterior mean
    ax2 = axes('Position', [0.39, 0.15, 0.28, 0.625]);
    contourf(ax2, C, A, posteriorMean, 10)
    colormap(gca, 'hot')
    clim([cmin, cmax])
    colorbar
    hold on
    plot(ax2, cbo.GP.X(:, 2), cbo.GP.X(:, 1), 'o', ...
        'MarkerSize', 5, 'MarkerEdgeColor', [0 1 0], ...
        'MarkerFaceColor', [1 1 1], 'LineWidth', 1.5);
    hold off
    grid on
    xlabel('Context')
    title('GP posterior mean')

    % Plot GP posterior standard deviation
    ax3 = axes('Position', [0.7, 0.15, 0.28, 0.625]);
    contourf(ax3, C, A, posteriorStd, 10)
    colormap(gca, 'parula')
    colorbar
    hold on
    plot(ax3, cbo.GP.X(:, 2), cbo.GP.X(:, 1), ...
        'o', 'MarkerSize', 5, 'MarkerEdgeColor', [0 1 0], ...
        'MarkerFaceColor', [1 1 1], 'LineWidth', 1.5);
    hold off
    grid on
    xlabel('Context')
    title('GP posterior standard deviation')

    % Add legend
    h = legend(legendEntries, legendLabels);
    set(h, 'position', [0.405 0.875 0.175 0.1]);
end