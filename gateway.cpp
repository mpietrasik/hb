#include "hmmsb.hpp"

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
     * 0.  E (NxN, logical)
     * 1.  mu (scalar)
     * 2.  sigma (scalar)
     * 3.  gamma (scalar)
     * 4.  lambda (scalar)
     * 5.  eta (scalar)
     * 6.  paths (NxK, uint16)
     * 7.  z_r (NxN, uint16)   NB: (i,j)-th element is z_{i->j}
     * 8.  z_l (NxN, uint16)   NB: (j,i)-th element is z_{i<-j}
     * 9.  ITERS (scalar)
     * 10. SEED (scalar)
     * 11. NUM_THREADS (scalar)
     * 12. INITIALIZE (logical scalar)
     * 13. useHpTuning (1x3 logical)
     *
     * Output argument list:
     * 0.  c (NxK, uint16)
     * 1.  z_r (NxN, uint16)
     * 2.  z_l (NxN, uint16)
     * 3.  m (scalar)
     * 4.  sigma (scalar)
     * 5.  gamma (scalar)
     * 6.  lambda (scalar)
     * 7.  eta (scalar)
     * 8.  ll (1xITERS) (complete log-likelihood at every iteration)
     *
     * The values contained in the arrays c,z_r,z_l start from one, just
     * like the MATLAB version of the sampler. This preserves compatibility
     * with our existing tools.
     *
     * If INITIALIZE is true, the sampler runs the latent variable generative
     * process to initialize c, z_r and z_l, and ignores their initial values.
     *
     * Note that the elements of c,z_r,z_l are addressed using zero-based,
     * column-major indexing!
     */
    
    bool            *E;
    hmmsb::uint16   *paths, *z_r, *z_l;
    double          *ll = NULL;
    double          m, sigma, gamma, lambda, eta;
    double          *m_out, *sigma_out, *gamma_out, *lambda_out, *eta_out;
    mwSize          model_N, model_K;
    unsigned int    ITERS, SEED, NUM_THREADS;
    bool            INITIALIZE;
    const bool*     useHpTuning;
    
    // Check number of arguments
    if (nrhs < 14 || nrhs > 14) {
        mexErrMsgTxt("Requires 14 input arguments");
    } else if (nlhs < 9 || nlhs > 9) {
        mexErrMsgTxt("Requires 9 output arguments");
    }
    
    // Initialize some outputs
    plhs[0] = mxDuplicateArray(prhs[6]);    // paths
    plhs[1] = mxDuplicateArray(prhs[7]);    // z_r
    plhs[2] = mxDuplicateArray(prhs[8]);    // z_l
    plhs[3] = mxDuplicateArray(prhs[1]);    // m
    plhs[4] = mxDuplicateArray(prhs[2]);    // sigma
    plhs[5] = mxDuplicateArray(prhs[3]);    // gamma
    plhs[6] = mxDuplicateArray(prhs[4]);    // lambda
    plhs[7] = mxDuplicateArray(prhs[5]);    // eta
    
    // Get inputs/outputs/pointers
    //E = (bool*) mxGetPr(prhs[0]);
    paths = (hmmsb::uint16*) mxGetPr(plhs[0]);
    z_r = (hmmsb::uint16*) mxGetPr(plhs[1]);
    z_l = (hmmsb::uint16*) mxGetPr(plhs[2]);
    
    mu = (double) mxGetScalar(prhs[1]);
    sigma = (double) mxGetScalar(prhs[2]);
    gamma  = (double) mxGetScalar(prhs[3]);
    lambda = (double) mxGetScalar(prhs[4]);
    eta = (double) mxGetScalar(prhs[5]);
    
    mu_out = (double*) mxGetPr(plhs[3]);
    sigma_out = (double*) mxGetPr(plhs[4]);
    gamma_out  = (double*) mxGetPr(plhs[5]);
    lambda_out = (double*) mxGetPr(plhs[6]);
    eta_out = (double*) mxGetPr(plhs[7]);
    
    ITERS = (unsigned int) mxGetScalar(prhs[9]);
    SEED = (unsigned int) mxGetScalar(prhs[10]);
    NUM_THREADS = (unsigned int) mxGetScalar(prhs[11]);
    INITIALIZE = (bool) mxGetScalar(prhs[12]);
    model_N = mxGetM(plhs[0]);  // Determine N,K from paths variable
    model_K = mxGetN(plhs[0]);
    useHpTuning = (const bool*) mxGetPr(prhs[13]);
    
    // Initialize remaining outputs
    plhs[8] = mxCreateNumericMatrix(1,ITERS,mxDOUBLE_CLASS,mxREAL);
    ll = (double*) mxGetPr(plhs[8]);

    model_N = 400; 
    const mwSize model_R = 2;

    std::unordered_set< int > coordinates;
    std::unordered_set< std::vector<int>, boost::hash<std::vector<int>> > coordinatesu;
    array<int, 3> a;

    vector<int> coordinate;
    int temp;
    std::ifstream file("synthetic_binary_tree_coordinates.txt");

    if (file.is_open()) {
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
    }

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

    mexPrintf("model_N %d model_K %d model_R %d \n", model_N, model_K, model_R);

    // Initialize Gibbs sampler
    hmmsb::sampler_class sampler(paths,z_r,z_l,mu,sigma,gamma,lambda,eta,model_N,model_R,model_K,SEED,NUM_THREADS, coordinates, entity_link_counts);
    
    if (INITIALIZE) {   // Run generative process, which also initializes the sufficient statistics
        mexPrintf("......Initializing with generative process\n");
        mexEvalString("pause(.001);");
        sampler.generative_latent();
        sampler.initialize_gs_ss_observed();
        //sampler.print_tree();
    } else {    // Don't run generative process, just initialize sufficient statistics with input
        mexPrintf("......Initializing with latent variable inputs\n");
        mexEvalString("pause(.001);");
        sampler.initialize_gs_ss();
        //sampler.print_tree();
    }

    // Run sampler for ITERS iterations
    for (unsigned int t = 0; t < ITERS; ++t) {
        if ((t+1) % 100 == 0) {
            mexPrintf("......Iteration %d\n",t+1);
            mexEvalString("pause(.001);");
        }
        sampler.gs_all();
        //sampler.hpTuning(useHpTuning);
        ll[t] = sampler.log_complete_likelihood();
    }

    for (int i = 0; i < model_N; i++){
        for (int j = 0; j < model_N; j++){
            if ((i % 7 == 0) && (j % 7 == 0)){
                mexPrintf("%d %d ->", i,j);
                for (int r=0; r < model_R; r++){
                    mexPrintf("%d = %.5f, ", r, sampler.get_community_relation(i,j,r));
                }
                mexPrintf("\n");
            }
        }
    }

    // Output hyperparameter estimates
    mu_out[0]        = sampler.get_mu();
    sigma_out[0]       = sampler.get_sigma();
    gamma_out[0]    = sampler.get_gamma();
    lambda_out[0]  = sampler.get_lambda();
    eta_out[0]  = sampler.get_eta();
}
