#include "hb.hpp"

#include <limits>

#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <array>
#include <unordered_set>
#include <set>
#include <boost/container_hash/hash.hpp>


using namespace std;

/*
 * MEX gateway function.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    /* Input argument list:
     * 0.  N (mwSize)
     * 1.  R (mwSize)
     * 2.  mu (scalar)
     * 3.  sigma (scalar)
     * 4.  gamma (scalar)
     * 5.  lambda (scalar)
     * 6.  eta (scalar)
     * 7.  paths (NxK, uint16)
     * 8.  z_r (NxN, uint16)   NB: (i,j)-th element is z_{i->j}
     * 9.  z_l (NxN, uint16)   NB: (j,i)-th element is z_{i<-j}
     * 10.  ITERS (scalar)
     * 11. SEED (scalar)
     * 12. NUM_THREADS (scalar)
     * 13. INITIALIZE (logical scalar)
     * 14. useHpTuning (1x3 logical)
     *
     * Output argument list:
     * 0.  paths (NxK, uint16)
     * 1.  z_r (NxN, uint16)
     * 2.  z_l (NxN, uint16)
     * 3.  mu (scalar)
     * 4.  sigma (scalar)
     * 5.  gamma (scalar)
     * 6.  lambda (scalar)
     * 7.  eta (scalar)
     * 8.  ll (1xITERS) (complete log-likelihood at every iteration)
     *
     */
    
    hb::uint16 *paths, *z_r, *z_l;
    double *ll = NULL;
    double mu, sigma, gamma, lambda, eta;
    double *mu_out, *sigma_out, *gamma_out, *lambda_out, *eta_out;
    mwSize model_N, model_R, model_L;
    unsigned int ITERS, SEED, NUM_THREADS;
    bool INITIALIZE;
    const bool* useHpTuning;
    
    // Check number of arguments
    if (nrhs < 15 || nrhs > 15) {
        mexErrMsgTxt("Requires 15 input arguments");
    } else if (nlhs < 9 || nlhs > 9) {
        mexErrMsgTxt("Requires 9 output arguments");
    }
    
    // Initialize some outputs
    plhs[0] = mxDuplicateArray(prhs[7]);    // paths
    plhs[1] = mxDuplicateArray(prhs[8]);    // z_r
    plhs[2] = mxDuplicateArray(prhs[9]);    // z_l
    plhs[3] = mxDuplicateArray(prhs[2]);    // m
    plhs[4] = mxDuplicateArray(prhs[3]);    // sigma
    plhs[5] = mxDuplicateArray(prhs[4]);    // gamma
    plhs[6] = mxDuplicateArray(prhs[5]);    // lambda
    plhs[7] = mxDuplicateArray(prhs[6]);    // eta
    
    // Get inputs/outputs/pointers
    paths = (hb::uint16*) mxGetPr(plhs[0]);
    z_r = (hb::uint16*) mxGetPr(plhs[1]);
    z_l = (hb::uint16*) mxGetPr(plhs[2]);
    
    model_N = mxGetM(plhs[0]);
    model_R = (mwSize) mxGetScalar(prhs[1]);
    mu = (double) mxGetScalar(prhs[2]);
    sigma = (double) mxGetScalar(prhs[3]);
    gamma  = (double) mxGetScalar(prhs[4]);
    lambda = (double) mxGetScalar(prhs[5]);
    eta = (double) mxGetScalar(prhs[6]);
    
    mu_out = (double*) mxGetPr(plhs[3]);
    sigma_out = (double*) mxGetPr(plhs[4]);
    gamma_out  = (double*) mxGetPr(plhs[5]);
    lambda_out = (double*) mxGetPr(plhs[6]);
    eta_out = (double*) mxGetPr(plhs[7]);
    
    ITERS = (unsigned int) mxGetScalar(prhs[10]);
    SEED = (unsigned int) mxGetScalar(prhs[11]);
    NUM_THREADS = (unsigned int) mxGetScalar(prhs[12]);
    INITIALIZE = (bool) mxGetScalar(prhs[13]);
    // model_N = mxGetM(plhs[0]);
    model_L = mxGetN(plhs[0]);
    useHpTuning = (const bool*) mxGetPr(prhs[14]);
    
    // Initialize remaining outputs
    plhs[8] = mxCreateNumericMatrix(1,ITERS,mxDOUBLE_CLASS,mxREAL);
    ll = (double*) mxGetPr(plhs[8]);

    // Read dataset
    std::unordered_set< int > coordinates;
    std::unordered_set< std::vector<int>, boost::hash<std::vector<int>> > coordinatesu;
    array<int, 3> a;

    vector<int> coordinate;
    int temp;
    std::string filename = "data/synthetic_binary_tree.txt"; 
    std::ifstream file(filename);

    if (file.is_open()) {
        mexPrintf("File opened successfully \n");
        std::string line;
        while (std::getline(file, line, '\n')) {
            stringstream ss(line.c_str());
            coordinate.clear();
            while(ss >> temp){
                coordinate.push_back(temp);
            }
            
            a = {coordinate[0],coordinate[1],coordinate[2]};
            coordinates.insert(coordinate[2]*model_N*model_N + coordinate[1]*model_N + coordinate[0]);
            coordinatesu.insert(coordinate);
        }
        file.close();
    }else{
        mexPrintf("File does not exist", filename);
    }

    // Establish weight of entities for stochastic sampling TODO: only perform this if using stoachstic sampling
    vector<int> entity_link_counts;
    for (mwIndex i=0; i < model_N; ++i){
        int tmp_int = 0;
        for (mwIndex j = 0; j < model_N; ++j) {
            for (mwIndex r = 0; r < model_R; ++r){
                if (coordinates.find(r*model_N*model_N + j*model_N + i)  != coordinates.end() ){
                    tmp_int = tmp_int + 1;
                }
                if (coordinates.find(r*model_N*model_N + i*model_N + j)  != coordinates.end() ){
                    tmp_int = tmp_int + 1;
                }
            }
        }
        entity_link_counts.push_back(tmp_int);
    }

    //mexPrintf("model_N %d Model_R %d model_L %d \n", model_N, model_R, model_L);

    // Initialize Gibbs sampler
    hb::sampler_class sampler(paths,z_r,z_l,mu,sigma,gamma,lambda,eta,model_N,model_R,model_L,SEED,NUM_THREADS, coordinates, entity_link_counts);
    
    // Initialize based on priors or existing parameters
    if (INITIALIZE) {  
        mexPrintf("......Initializing with generative process\n");
        mexEvalString("pause(.001);");
        sampler.generative_latent();
        sampler.initialize_gs_ss_observed();
    } else {   
        mexPrintf("......Initializing with latent variable inputs\n");
        mexEvalString("pause(.001);");
        sampler.initialize_gs_ss();
    }

    // Run sampler for ITERS iterations
    for (unsigned int t = 0; t < ITERS; ++t) {
        if ((t+1) % 100 == 0) {
            mexPrintf("......Iteration %d\n",t+1);
            mexEvalString("pause(.001);");
        }
        sampler.gs_all();
        ll[t] = sampler.log_complete_likelihood();
    }

    // Output hyperparameter estimates
    mu_out[0]        = sampler.get_mu();
    sigma_out[0]       = sampler.get_sigma();
    gamma_out[0]    = sampler.get_gamma();
    lambda_out[0]  = sampler.get_lambda();
    eta_out[0]  = sampler.get_eta();
}
