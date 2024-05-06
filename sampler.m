% Wrapper for the C++ Gibbs sampler.
%
% Outputs a cell array of structs containing path/level samples.

function samples = sampler(N, R, L, mu, sigma, gamma, lambda, eta, GS_PARAM, SEED, NUM_THREADS, useHpTuning)
    
    useHpTuning = [true true true]; % Use hyperparameter tuning for gamma, (mu,sigma) and (lambda,eta) respectively?
   
    % Declare return variables
    samples = cell(1,GS_PARAM.NUM_SAMPLES);
    
    % Take samples
    samples{1}.all_ll = [];
    for i = 1:GS_PARAM.NUM_SAMPLES
        if i == 1
            % First iteration initialization
            ITERS = GS_PARAM.BURN_IN; % Burn-in on first iteration
            if ~isfield(GS_PARAM,'RANDOM_INTEGER_INIT') || GS_PARAM.RANDOM_INTEGER_INIT == false

                % For now just declare empty inputs.
                paths = uint16(zeros(N,L));
                z_r = uint16(zeros(N,N));
                z_l = uint16(zeros(N,N));
                INITIALIZE  = true;
                disp('...Using generative process initialization');

            else

                % Initialize by drawing random integers in [1,L].
                rng = RandStream('mt19937ar','Seed',SEED);
                paths = uint16(rng.randi(L,N,L));
                z_r = uint16(rng.randi(L,N,N));
                z_l = uint16(rng.randi(L,N,N));
                INITIALIZE  = false;
                disp('...Using random-integer initialization');
            end
        else
            ITERS = GS_PARAM.LAG;
            INITIALIZE = false; 
        end

        fprintf('...Taking sample %d/%d (burn-in/lag for %d iterations)\n',i,GS_PARAM.NUM_SAMPLES,ITERS);
        [paths, z_r, z_l ,mu, sigma, gamma, lambda, eta, ll] = gateway(N, R, mu, sigma, gamma, lambda, eta, paths, z_r, z_l, ITERS+1, SEED+i-1, NUM_THREADS, INITIALIZE, useHpTuning);
        
        % Save the sample
        samples{i}.paths = paths;
        samples{i}.z_r = z_r;
        samples{i}.z_l = z_l;
        samples{i}.mu = mu; 
        samples{i}.sigma = sigma;
        samples{i}.gamma = gamma;
        samples{i}.lambda = lambda;
        samples{i}.eta = eta;
      
        % Save the complete log-likelihood
        samples{i}.ll = ll; 
        samples{1}.all_ll = [samples{1}.all_ll ll]; 
    end
    
end
