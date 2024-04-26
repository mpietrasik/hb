% Script to run the Gibbs sampler on the Synthetic Binary Tree dataset
% described in the paper.

% Declare hyperparameters
gamma = 0.75; 
lambda = 0.1;
eta = 0.9; 
K = 4;
mu = 0.9999;
sigma = 0.0001;

% Load data
syntheticData = load('../../data/people_places/second_order.mat');

% Gibbs sampling parameters
GS_PARAM.NUM_SAMPLES    = 10;
GS_PARAM.BURN_IN        = 200;
GS_PARAM.LAG            = 2;
    
% Random seeds that can be set for reproducibility
SEED_gs = uint32(1);
    
% Number of threads
NUM_THREADS = 1;
    
% Record parameters
data.mu = mu;
data.sigma = sigma;
data.gamma = gamma;
data.lambda = lambda;
data.eta = eta;
data.K_gs = K_gs;
data.GS_PARAM = GS_PARAM;
data.SEED_gs = SEED_gs;
data.NUM_THREADS_gs = NUM_THREADS_gs;

% Run gibbs sampling algorithm
fprintf('Running Gibbs sampler...\n');
gs_start        = tic;


E = logical(ones(1142));

data.samples    = hmmsb_gs(E, K_gs, mu, sigma, gamma, lambda, eta, GS_PARAM, SEED_gs, NUM_THREADS_gs);

data.gs_time    = toc(gs_start);
fprintf('Finished Gibbs sampling in %f seconds\n',data.gs_time);
