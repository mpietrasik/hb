% Script to run the Gibbs sampler on the Synthetic Binary Tree dataset
% described in the paper.

% Declare hyperparameters
gamma = 0.4; 
lambda = 1;
eta = 1; 
L = 4;
mu = 0.5;
sigma = 2;

% Data parameter
N = 400;
R = 2;

% Gibbs sampling parameters
GS_PARAM.NUM_SAMPLES    = 10;
GS_PARAM.BURN_IN        = 6000;
GS_PARAM.LAG            = 3;
    
% Random seeds that can be set for reproducibility
SEED_gs = uint32(1);
    
% Number of threads
NUM_THREADS_gs = 1;
    
% Record parameters
data.mu = mu;
data.sigma = sigma;
data.gamma = gamma;
data.lambda = lambda;
data.eta = eta;
data.L = L;
data.GS_PARAM = GS_PARAM;
data.SEED_gs = SEED_gs;
data.NUM_THREADS_gs = NUM_THREADS_gs;

% Run gibbs sampling algorithm
fprintf('Running Gibbs sampler...\n');
gs_start        = tic;

data.samples    = sampler(N, R, L, mu, sigma, gamma, lambda, eta, GS_PARAM, SEED_gs, NUM_THREADS_gs);

data.gs_time    = toc(gs_start);
fprintf('Finished Gibbs sampling in %f seconds\n',data.gs_time);
