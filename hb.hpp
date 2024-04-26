#ifndef HB_HPP_
#define HB_HPP_

//#include "cp_thread_group.hpp"  // Rudimentary thread group to get around boost::thread segfaults with MATLAB on Linux

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <iterator>

#include <mex.h>

#include <set>
#include <unordered_set>
#include <array>
#include <boost/container_hash/hash.hpp>


namespace hb {
    
    using namespace std;
    
    typedef unsigned short uint16;  // Should work for all x86/x64 compilers
    
    enum prob_mode_t {PRIOR, POSTERIOR};    // Specifies a probability type to compute
    enum z_mode_t {Z_R, Z_L};               // Specifies the latent variables z_r or z_l
    
    /*
     * Random number generator class.
     */
    class rng_class {
        boost::mt19937                                                      generator;
        boost::uniform_real<>                                               zero_one_dist;
        boost::variate_generator<boost::mt19937&, boost::uniform_real<> >   zero_one_generator;
        
    public:
        rng_class(unsigned int seed)
            : generator(seed),
              zero_one_dist(0,1),
              zero_one_generator(generator, zero_one_dist)
        { }
        
        /*
         * Draws a random unsigned integer.
         */
        unsigned int rand_int() { return generator(); }
        
        /*
         * Draws a random real number in [0,1).
         */
        double rand() { return zero_one_generator(); }

        
        /*
         * Draws a random number from a Gamma(a) distribution.
         */
        double rand_gamma(double a) {
            boost::gamma_distribution<> gamma_dist(a);
            boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> >
                gamma_generator(generator, gamma_dist);
            return gamma_generator();
        }
        
        /*
         * Draws a random number from an Exponential(a) distribution.
         */
        double rand_exponential(double a) {
            boost::exponential_distribution<> exponential_dist(a);
            boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> >
                exponential_generator(generator, exponential_dist);
            return exponential_generator();
        }
        
        /*
         * Draws a random vector from a symmetric Dirichlet(a) distribution.
         * The dimension of the vector is determined from output.size().
         */
        void rand_sym_dirichlet(double a, std::vector<double>& output) {
            boost::gamma_distribution<> gamma_dist(a);
            boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> >
                gamma_generator(generator, gamma_dist);
            double total    = 0;
            for (unsigned int i = 0; i < output.size(); ++i) {
                output[i]   = gamma_generator();
                total       += output[i];
            }
            for (unsigned int i = 0; i < output.size(); ++i) {
                output[i]   /= total;
            }
        }
        
        /*
         * Samples from an unnormalized discrete distribution, in the range
         * [begin,end].
         */
        size_t rand_discrete(const vector<double>& distrib, size_t begin, size_t end) {
            double totprob  = 0;
            for (size_t i = begin; i <= end; ++i) totprob += distrib[i];
            double rrr        = totprob * zero_one_generator();
            double cur_max  = distrib[begin];
            size_t idx      = begin;
            while (rrr>cur_max) { cur_max += distrib[++idx]; }
            return idx;
        }
        
        /*
         * Converts the range [begin,end] of a vector of log-probabilities
         * into relative probabilities.
         */
        static void logprobs_to_relprobs(vector<double> &distrib, size_t begin, size_t end) {
            double max_log = *max_element( distrib.begin()+begin , distrib.begin()+end+1 ); // Find the maximum element in [begin,end]
            for (size_t i = begin; i <= end; ++i) distrib[i] = exp(distrib[i] - max_log);   // Avoid over/underflow by centering log-probabilities to their max
        }
        
        /*
         * Log-gamma function.
         */
        static double ln_gamma(double xx) {
            int j;
            double x,y,tmp1,ser;
            static const double cof[6]={76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5};
            y=xx;
            x=xx;
            tmp1=x+5.5;
            tmp1 -= (x+0.5)*log(tmp1);
            ser=1.000000000190015;
            for (j=0;j<6;j++) ser += cof[j]/++y;
            return -tmp1+log(2.5066282746310005*ser/x);
        }
    };
    
    /*
     * Class to perform generative process sampling, Gibbs sampling,
     * and log-likelihood Monte Carlo integration.
     */
    class sampler_class {
        
        typedef boost::tuple<unsigned int,unsigned int> B_t;    // B-element sufficient statistics type
        typedef boost::tuple<unsigned int[1000],unsigned int[1000]> B_t_r;
        
        /*
         * Tree vertex (restaurant) data structure, meant to serve as
         * sufficient stats for paths c and blockmatrices B.
         */
        struct vertex {
            struct child {  // Data structure to represent children
                vertex  *ptr;                   // Pointer to child
                int     pathcount;              // Number of paths associated with a child
                int     potential_pathcount;    // Number of potential paths associated with a child
                                                // (current paths plus paths from the next new table at each restaurant)
                std::map<int,B_t>   B_row;      // Sufficient stats for B. children[a].B_row[b] is the sufficient stat for the
                                                // blockmatrix B(a,b) in our model. This sufficient stat is defined as a pair whose
                                                // first element is the number of 1-edges E(i,j) such that c_i[z_coarse(i,j)] == a
                                                // and c_j[z_coarse(i,j)] == b. The second element is the number of 0-edges.
                std::map<int,B_t_r>   B_row_r;
                child() : ptr(NULL), pathcount(0), potential_pathcount(0) {}
            };
            map<int,child>  children;   // Map of children/tables.
        };
        
        /*
         * Data structure containing references to thread-specific objects
         * required when sampling levels z.
         */
        struct z_thread_data_t {
            rng_class       &rng;
            vector<double>  &level_greater_counts;
            vector<double>  &level_probs;
            vector<double>  &level_prod_vector;
            
            z_thread_data_t(rng_class& rngi, vector<double> &level_greater_countsi,
                            vector<double> &level_probsi, vector<double> &level_prod_vectori)
            : rng(rngi), level_greater_counts(level_greater_countsi),
              level_probs(level_probsi), level_prod_vector(level_prod_vectori)
            { }
        };
         
        /*
         * Thread worker class for the parallel generative process on z.
         *
         * As with the parallel Gibbs sampler, we sample z_r(i,j), z_l(i,j)
         * in parallel over i. Unlike the Gibbs sampler, the generative
         * process does not involve the P(E(i,j)|z_r(i,j),z_l(i,j)) term,
         * hence each thread may sample all j's sequentially without
         * waiting for other threads to finish.
         */
        class generative_z_worker {
            // The object will be copied by boost::thread, so we should
            // only store the minimal number of data members. All other
            // objects should be initialized in operator().
            sampler_class   &sampler;       // Reference to the main sampler object
            mwIndex         i_offset;
            unsigned int    seed;
            
        public:

            generative_z_worker(sampler_class &sampleri, mwIndex i_offseti, unsigned int seedi)
                : sampler(sampleri), i_offset(i_offseti), seed(seedi)
            { }
            
            void operator()() {
                // Instantiate thread-specific data structure for sampling z
                rng_class       rng(seed);
                vector<double>  level_greater_counts(sampler.model_K+2);
                vector<double>  level_probs(sampler.model_K+2);
                vector<double>  level_prod_vector(sampler.model_K+2);
                z_thread_data_t ztd(rng,level_greater_counts,level_probs,level_prod_vector);
                
                // Now sample z
                for (mwIndex i = i_offset; i < sampler.model_N; i += sampler.num_threads) {
                    for (mwIndex j = 0; j < sampler.model_N; ++j) {
                        if (i==j) continue;
                        sampler.generative_z(Z_R,i,j,ztd);
                        sampler.generative_z(Z_L,i,j,ztd);
                    }
                }
            }
        };
        
        // Constants
        
        // Pointers to MATLAB observed/latent variables (all column-major indexed)
        //bool            *E;             // From prhs (input) or plhs (output), depending on purpose.
        std::unordered_set< int > coordinates;
        std::vector< int > entity_link_counts; 
        uint16          *c, *z_r, *z_l; // From plhs (output). Values start from 1, not 0.
        
        // Model parameters
        double          m, pi, gamma, lambda1, lambda2;
        const mwSize    model_N, model_R, model_K;
        
        // Tree data structure (sufficient statistics for paths c and blockmatrices B)
        vertex          *head;
        int             pathcount;
        int             potential_pathcount;
        
        // Temporary variables for sampling paths c
        vector<double>  path_probs;     // Stores potential path relative probabilities
        
        // Sufficient statistics for levels z
        vector<unsigned int>    z_counts;               // Stores level counts. Has size model_N*model_K. Actor-major, level-minor indexing.
        
        // Main thread temporary variables for sampling levels z
        vector<double>  level_greater_counts;   // Stores counts >= a given level
        vector<double>  level_probs;            // Stores level relative probabilities
        vector<double>  level_prod_vector;      // Temp vector to assist in computing level_probs
        
        // RNG
        rng_class       rng;
        
        // Thread-related
        unsigned int    num_threads;
        z_thread_data_t z_thread_data;  // Contains references to thread-specific objects necessary for sampling z (for the main thread)
        
        // Miscellaneous/debugging
        int             iterCount;

        bool bool_lookup(mwIndex i, mwIndex j, mwIndex r) const{
            //vector<int> vect{ (int) i, (int) j, (int) r };
            return (coordinates.find(r*model_N*model_N + j*model_N + i)  != coordinates.end() );
        }
        
        
        /*
         * Returns a pointer to the element of B_row that corresponds to
         * E(i,j), based on the current values of c, z_r and z_l. If there
         * is no such element (see code for details), returns NULL.
         *
         * This function assumes that the paths denoted by c(i) and c(j)
         * are actually in the tree, so as to get the sufficient statistics
         * B_row. Hence, when sampling new potential paths, one should call
         * add_path() to ensure the appropriate B_row's have been allocated.
         */
        B_t* get_B_element_ptr(mwIndex i, mwIndex j) { return const_cast<B_t*>(get_B_element_ptr_helper(i,j)); }
        const B_t* get_B_element_ptr(mwIndex i, mwIndex j) const { return get_B_element_ptr_helper(i,j); }
        const B_t* get_B_element_ptr_helper(mwIndex i, mwIndex j) const {
            
            unsigned int z_coarse = min(z_r[j*model_N + i],z_l[i*model_N + j]); // Note that z_coarse is a 1-indexed offset

            // Find the deepest level ell <= z_coarse such that
            // c(i,1:ell-1) == c(j,1:ell-1), i.e. "deepest common node of paths i,j".
            // By convention we set ell = 1 if c(i,1) != c(j,1). We then return
            // the element of B_row that stores the sufficient statistics for
            // edges from community c(i,ell) to c(j,ell).
            mwIndex ell = 1;
            vertex *cur_vertex = head;  // cur_vertex represents the parent at depth d-1
            for (; ell < z_coarse; ++ell) {
                // Invariant: at this point, c(i,1:ell-1) == c(j,1:ell-1)
                if (c[(ell-1)*model_N + i] != c[(ell-1)*model_N + j]) { // c is 0-indexed, so we subtract 1 from levels ell
                    break;
                }
                cur_vertex = cur_vertex->children[(int) c[(ell-1)*model_N + i]].ptr;
            }
            return &cur_vertex->children[(int) c[(ell-1)*model_N + i]].B_row[(int) c[(ell-1)*model_N + j]];
        }

        B_t_r* get_B_element_ptr_r(mwIndex i, mwIndex j) { return const_cast<B_t_r*>(get_B_element_ptr_r_helper(i,j)); }
        const B_t_r* get_B_element_ptr_r(mwIndex i, mwIndex j) const { return get_B_element_ptr_r_helper(i,j); }
        const B_t_r* get_B_element_ptr_r_helper(mwIndex i, mwIndex j) const {
        
            
            unsigned int z_coarse = min(z_r[j*model_N + i],z_l[i*model_N + j]); // Note that z_coarse is a 1-indexed offset
            double rrr = ((double) rand() / (RAND_MAX));

            // Find the deepest level ell <= z_coarse such that
            // c(i,1:ell-1) == c(j,1:ell-1), i.e. "deepest common node of paths i,j".
            // By convention we set ell = 1 if c(i,1) != c(j,1). We then return
            // the element of B_row that stores the sufficient statistics for
            // edges from community c(i,ell) to c(j,ell).
            mwIndex ell = 1;
            vertex *cur_vertex = head;  // cur_vertex represents the parent at depth d-1
            for (; ell < z_coarse; ++ell) {
                // Invariant: at this point, c(i,1:ell-1) == c(j,1:ell-1)
                if (c[(ell-1)*model_N + i] != c[(ell-1)*model_N + j]) { // c is 0-indexed, so we subtract 1 from levels ell
                    break;
                }
                cur_vertex = cur_vertex->children[(int) c[(ell-1)*model_N + i]].ptr;
            }
            return &cur_vertex->children[(int) c[(ell-1)*model_N + i]].B_row_r[(int) c[(ell-1)*model_N + j]];
        }
        
        /*
         * Computes log(P(E(i,j),|c,z,E_{-(i,j})), according to the current
         * value of the vertex-situated sufficient statistics B_row, as
         * well as c,z_r,z_l.
         * 
         * The B_row's are expected to account for all E's, except E(i,j).
         * 
         * Note that the actor indices i,j are zero-based.
         *
         * This function is primarily called by sample_z().
         */
         double log_E_likelihood(mwIndex i, mwIndex j) const {
            //CHANGE
            const B_t_r* B_el = get_B_element_ptr_r(i,j);
            
            // Likelihood is a ratio of two normalizers with gamma functions;
            // we use an equivalent expression without the gamma functions.
            double likelihood_sum = 0;
            
            for (int r = 0; r < model_R; ++r){
                //bool E_value    = E[r*model_N*model_N + j*model_N + i];
                bool E_value    = bool_lookup(i,j,r); 
                likelihood_sum = log( E_value*(B_el->get<0>()[r]+lambda1) + (1-E_value)*(B_el->get<1>()[r]+lambda2) )
                - log( B_el->get<0>()[r] + B_el->get<1>()[r] + lambda1 + lambda2 );
            }
            //mexErrMsgTxt("error");
            return likelihood_sum;
        }
        
        /*
         * Computes the conditional log-likelihood of edges E related to
         * path c(i,:), log(P(E_{i,.},E_{.,i}|c,z,E_{-i})), according to
         * the current value of the vertex-situated sufficient statistics
         * B_row, as well as c,z_r,z_l.
         *
         * As with log_E_likelihood(), this function expects the B_row's
         * to account for all E's except E_{i,.},E{.,i}.
         *
         * Used by write_path_probs() during sample_c().
         */
        double log_path_E_likelihood(mwIndex i) const {
            
            double ll   = 0;
            
            // The idea is to find the set of B_row elements B_el touched
            // by at least one edge in E_{i,.},E{.,i}. While doing that, we
            // we also count the edges in E_{i,.},E{.,i}. Finally, we
            // compute the likelihood.
            
            // Find the set of B_el's
            //typedef std::map<const B_t*,B_t>    B_row_ptr_map_t;
            //B_row_ptr_map_t                     B_row_ptr_map;
            typedef std::map<const B_t_r*,B_t_r>    B_row_ptr_r_map_t;
            B_row_ptr_r_map_t                     B_row_ptr_r_map;
            
            for (mwIndex j = 0; j < model_N; ++j) {
                if (j==i) continue;
                for (unsigned int swap = 0; swap < 2; ++swap) {
                    const B_t_r* B_el = swap?get_B_element_ptr_r(j,i):get_B_element_ptr_r(i,j);
                    for (mwIndex r = 0; r < model_R; ++r){
                        bool E_value = swap?bool_lookup(j,i,r):bool_lookup(i,j,r);
                        // Add B_el to B_row_ptr_map, and accumulate counts
                        if (E_value)    ++B_row_ptr_r_map[B_el].get<0>()[r];
                        else            ++B_row_ptr_r_map[B_el].get<1>()[r];
                    }
                }
            }
            
            // Add likelihoods from the set of B_el's
            for (B_row_ptr_r_map_t::iterator it = B_row_ptr_r_map.begin(), end = B_row_ptr_r_map.end();
                 it != end;
                 ++it) {
                const B_t_r *B_el = it->first;
                B_t_r &counts     = it->second;
                
                
                for (mwIndex r = 0; r < model_R; ++r){
                    // Likelihood is a ratio of two normalizers
                    ll +=
                        // Inverse normalizer on sufficient stats, excluding E_{i,.},E{.,i} associated with B_el
                        rng_class::ln_gamma(B_el->get<0>()[r] + B_el->get<1>()[r] + lambda1 + lambda2)
                        - rng_class::ln_gamma(B_el->get<0>()[r] + lambda1)
                        - rng_class::ln_gamma(B_el->get<1>()[r] + lambda2)
                        // Normalizer on sufficient stats, including E_{i,.},E{.,i} associated with B_el
                        + rng_class::ln_gamma(B_el->get<0>()[r] + counts.get<0>()[r] + lambda1)
                        + rng_class::ln_gamma(B_el->get<1>()[r] + counts.get<1>()[r] + lambda2)
                        - rng_class::ln_gamma(B_el->get<0>()[r] + B_el->get<1>()[r] + counts.get<0>()[r] + counts.get<1>()[r] + lambda1 + lambda2);
                }
                
            }
            return ll;
        }
        
        /*
         * Computes the conditional log-likelihood log(P(E|c,z)), according
         * to the current value of c,z_r,z_l.
         *
         * Unlike log_E_likelihood() and log_path_E_likelihood(), this
         * function ignores the value of the B_row's. The necessary
         * sufficient statistics are computed while determining which
         * edges are associated with which B_row's.
         *
         * This function is used to compute the marginal log-likelihood via
         * Monte Carlo integration. It is also used to compute the complete
         * log-likelihood log(P(E,c,z)).
         */
        
        double log_all_E_likelihood() const {
            
            // TODO:
            // There is no need to look at every edge; the required sufficient
            // statistics can be found by looking at the B_row's in the hierarchy.
            // Note that ll_sample() will require a call to initialize_gs_ss_observed().
            
            double ll   = 0;
            
            // Find all B_el's and their sufficient statistics by considering all edges E
            //typedef std::map<const B_t*,B_t>    B_row_ptr_map_t;
            //B_row_ptr_map_t                     B_row_ptr_map;
            typedef std::map<const B_t_r*,B_t_r>    B_row_ptr_r_map_t;
            B_row_ptr_r_map_t                     B_row_ptr_r_map;
            for (mwIndex i = 0; i < model_N; ++i) {
                for (mwIndex j = 0; j < model_N; ++j) {
                    if (j==i) continue;
                    //const B_t* B_el = get_B_element_ptr(i,j);
                    const B_t_r* B_el = get_B_element_ptr_r(i,j);
                    for (mwIndex r = 0; r < model_R; ++r){
                        //bool E_value    = E[j*model_N + i];
                        bool E_value = bool_lookup(i,j,r);
                    
                        // Add B_el to B_row_ptr_map, and accumulate counts
                        //if (E_value)    ++B_row_ptr_map[B_el].get<0>();
                        //else            ++B_row_ptr_map[B_el].get<1>();
                        if (E_value)    ++B_row_ptr_r_map[B_el].get<0>()[r];
                        else            ++B_row_ptr_r_map[B_el].get<1>()[r];
                    }
                }
            }
            
            for (B_row_ptr_r_map_t::iterator it = B_row_ptr_r_map.begin(), end = B_row_ptr_r_map.end();
                 it != end;
                 ++it) {
                B_t_r &counts     = it->second;
                
                
                for (mwIndex r = 0; r < model_R; ++r){
                    // Likelihood is a ratio of two normalizers
                    ll +=
                        // Inverse normalizer on sufficient stats, excluding E_{i,.},E{.,i} associated with B_el
                        rng_class::ln_gamma(lambda1 + lambda2)
                        - rng_class::ln_gamma(lambda1)
                        - rng_class::ln_gamma(lambda2)
                        // Normalizer on sufficient stats, including E_{i,.},E{.,i} associated with B_el
                        + rng_class::ln_gamma(counts.get<0>()[r] + lambda1)
                        + rng_class::ln_gamma(counts.get<1>()[r] + lambda2)
                        - rng_class::ln_gamma(counts.get<0>()[r] + counts.get<1>()[r] + lambda1 + lambda2);
                }
                
            }
            
            return ll;
        }
        
        /*
         * Computes the log-prior log(P(c)), based on the current state
         * of the path tree (which serves as the sufficient statistics for
         * paths c).
         *
         * Used to compute the complete log-likelihood log(P(E,c,z)).
         */
        double log_prior_c() const { return log_prior_c_helper(head, 0); }
        double log_prior_c_helper(vertex* node, mwIndex depth) const {
            // Compute the log-probability of the paths at this vertex's
            // CRP, and those of its children
            double ll = 0;
            unsigned int tot_paths = 0;
            for (map<int,vertex::child>::const_iterator it = node->children.begin(), end = node->children.end();
                 it != end;
                 ++it) {
                // Probability of the first path at a table
                ll += log(gamma) - log((tot_paths++)+gamma);
                // Probability of the remaining paths at a table
                for (unsigned int x = 1; x < it->second.pathcount; ++x) ll += log((double) x) - log((tot_paths++)+gamma);
                if (depth < model_K-1) {
                    // Recursively compute the path probabilities at the
                    // child restaurant associated with this table
                    ll += log_prior_c_helper(it->second.ptr, depth+1);
                }
            }
            return ll;
        }
        
        /*
         * Computes the log-prior log(P(z)), based on the current state
         * of the sufficient statistics z_counts.
         *
         * Used to compute the complete log-likelihood log(P(E,c,z)).
         */
        double log_prior_z() const {
            double ll = 0;
            for (mwIndex i = 0; i < model_N; ++i) ll += log_prior_z_helper(i);
            return ll;
        }
        double log_prior_z_helper(mwIndex i) const {
            vector<double> lgc(model_K+2);
            vector<double> lp(model_K+2);
            vector<double> lpv(model_K+2);  // Represents sum(log(V)) up to a given index
            double ll               = 0;            
            unsigned int i_offset   = i * (model_K+1);
            
            // Go over each z associated with actor i
            for (uint16 ell = model_K; ell >= 1; --ell) {   // For each level ell
                for (unsigned int x = 0; x < z_counts[i_offset + ell]; ++x) {   // For each z in z_counts(i,ell)
                    
                    // Compute unnormalized log-probabilities for z being at level 1 through K
                    lpv[0]              = 0;
                    //double prior_sum    = 0;    // Used to compute the probability of drawing a level > K. For this fixed K implementation, we have no need of it.
                    for (uint16 d = 1; d <= model_K; ++d) {
                        // Compute prior log-probability
                        double level_count          = lgc[d] - lgc[d+1];
                        double log_V                = log(m*pi + level_count) - log(pi + lgc[d]);
                        double log_prob             = log_V + lpv[d-1];
                        //prior_sum                   += exp(log_prob);
                        lpv[d]                      = log(1-exp(log_V)) + lpv[d-1];
                        
                        // Store the log-probability
                        lp[d]                       = log_prob;
                    }
                    
                    // Now add the log-normalized-probability for z at its actual level ell
                    double denominator  = 0;
                    for (uint16 d = 1; d <= model_K; ++d) denominator += exp(lp[d]);
                    ll += lp[ell] - log(denominator);
                    
                    // Update lgc
                    for (uint16 d = 1; d <= ell; ++d) ++lgc[d];
                }
            }
            
            return ll;
        }


        
    public:
        /*
         * Constructor.
         */
        //sampler_class(bool *Ei, uint16 *ci, uint16 *z_ri, uint16 *z_li,
        sampler_class(uint16 *ci, uint16 *z_ri, uint16 *z_li,
                 double mi, double pii, double gammai,
                 double lambda1i, double lambda2i,
                 mwSize model_Ni, mwSize model_Ri, mwSize model_Ki,
                 unsigned int SEED, unsigned int num_threadsi, std::unordered_set< int > coordinatesi, std::vector< int > entity_link_countsi)
                 : // Observed and latent variables
                   //E(Ei), 
                   coordinates(coordinatesi), entity_link_counts(entity_link_countsi), c(ci), z_r(z_ri), z_l(z_li),
                   
                   // Model parameters
                   m(mi), pi(pii), gamma(gammai), lambda1(lambda1i),
                   lambda2(lambda2i), model_N(model_Ni), model_R(model_Ri), model_K(model_Ki),
                   
                   // Path sufficient statistics
                   path_probs(model_Ni*model_Ki),       // N*K is an upper bound on #[potential paths] = #[leaf tables] + #[internal nodes including root]
                   
                   // Level sufficient statistics
                   z_counts(model_Ni*(model_Ki+1),0),   // N actors, K+1 levels including level 0
                   level_greater_counts(model_Ki+2),    // K+2 levels including level 0 and K+1
                   level_probs(model_Ki+2),             // K+2 levels including level 0 and K+1
                   level_prod_vector(model_Ki+2),       // K+2 levels including level 0 and K+1
                   
                   // Miscellaneous
                   rng(SEED), num_threads(num_threadsi),
                   
                   // Initialize main-thread-specific data structure for sampling z
                   z_thread_data(rng,level_greater_counts,level_probs,level_prod_vector)
        { create_empty_tree(); iterCount = 0; mexPrintf("sampler_class called\n");}


        
        /*
         * Destructor.
         */
        ~sampler_class() { delete_tree(); }
        
        /*
         * Initializes sufficient statistics for the Gibbs sampler using
         * the current values of c,z_r,z_l,E. Must  be called before
         * performing Gibbs sampling.
         * 
         * Alternatively, call generative_latent() followed by
         * initialize_gs_ss_observed() to initialize using the
         * generative process. This will overrides the input values of c,
         * z_r  and z_l.
         *
         * More specifically, this function:
         * 1. Adds every path c to the path tree.
         * 2. Adds every level z_r, z_l to the level count vector z_counts.
         * 3. Adds every edge E to the blockmatrix sufficient stats B_row's.
         * The first two items are initialized by initialize_gs_ss_latent(),
         * while the last item is initialized by initialize_gs_ss_observed().
         */
        void initialize_gs_ss() { initialize_gs_ss_latent(); initialize_gs_ss_observed(); }
        void initialize_gs_ss_latent() {
            for (mwIndex i = 0; i < model_N; ++i) {
                add_path(i);    // Add c(i)
                for (mwIndex j = 0; j < model_N; ++j) {
                    if (j==i) continue;
                    ++z_counts[i*(model_K+1) + z_r[j*model_N + i]]; // Add z_r(i,j)
                    ++z_counts[i*(model_K+1) + z_l[j*model_N + i]]; // Add z_l(i,j)
                }
            }
        }
        void initialize_gs_ss_observed() {
            
            for (mwIndex i = 0; i < model_N; ++i) {
                for (mwIndex j = 0; j < model_N; ++j) { // Add E(i,j)
                    if (j==i) continue;
                    B_t_r* B_el       = get_B_element_ptr_r(i,j);
                    for (mwIndex r = 0; r < model_R; ++r){
                        //bool E_value    = E[r*model_N*model_N + j*model_N + i];
                        bool E_value    = bool_lookup(i,j,r);
                        if (E_value)    ++B_el->get<0>()[r];
                        else            ++B_el->get<1>()[r];
                    }
                }
            }
        }
        
        /*
         * Path-tree-related functions. Note that depths are 0-based,
         * unlike levels which are 1-based (even though they both mean the
         * same thing).
         */
        void create_empty_tree() {  // Creates an empty tree. Call delete_tree() first if there is an existing tree at the pointer head.
            head                = new vertex;
            pathcount           = 0;    // Empty root restaurant contains 0 paths
            potential_pathcount = 1;    // Empty root restaurant has 1 potential path
        }
        void add_path(mwIndex path_idx) {  // Adds path c(i,:) to the tree. Does not modify blockmatrix B sufficient stats!
            ++pathcount;
            potential_pathcount += add_path_helper(head, 0, path_idx);
        }
        int add_path_helper(vertex* node, mwIndex depth, mwIndex path_idx) {    // Returns #[newly created potential paths associated with this node]
            int child_idx               = (int) c[depth*model_N + path_idx];    // Column-major indexing
            int add_potential_pathcount = 0;
            if (node->children.count(child_idx) == 0) { // We need to create a new child/table
                node->children[child_idx].pathcount             = 0;
                node->children[child_idx].potential_pathcount   = 1;
                ++add_potential_pathcount;                              // Create a potential path corresponding to the new child (specifically, the child's next new table)
            }
            ++node->children[child_idx].pathcount;
            if (depth < model_K-1) {    // If not maxdepth then recurse
                if (node->children[child_idx].ptr == NULL) node->children[child_idx].ptr = new vertex;  // Create child vertex if necessary
                int child_add_potential_pathcount               = add_path_helper(node->children[child_idx].ptr, depth+1, path_idx);
                node->children[child_idx].potential_pathcount   += child_add_potential_pathcount;
                add_potential_pathcount                         += child_add_potential_pathcount;   // Accumulate new potential paths from this child
            }
            return add_potential_pathcount;
        }
        void rm_path(mwIndex path_idx) {  // Removes path c(i,:) from the tree. Does not check if the path actually exists!
                                          // Also does not modify blockmatrix B sufficient stats, except when a path is deleted,
                                          // which also deletes the B_row in it. Nevertheless, such paths are expected to have zero
                                          // sufficient stats before being removed.
            --pathcount;
            potential_pathcount -= rm_path_helper(head, 0, path_idx);
        }
        int rm_path_helper(vertex* node, mwIndex depth, mwIndex path_idx) {     // Returns #[newly destroyed potential paths associated with this node]
            int child_idx               = (int) c[depth*model_N + path_idx];    // Column-major indexing
            int rm_potential_pathcount  = 0;
            --node->children[child_idx].pathcount;
            if (depth < model_K-1) {   // If not maxdepth, then recurse
                int child_rm_potential_pathcount                = rm_path_helper(node->children[child_idx].ptr, depth+1, path_idx);
                node->children[child_idx].potential_pathcount   -= child_rm_potential_pathcount;
                rm_potential_pathcount                          += child_rm_potential_pathcount;    // Accumulate destroyed potential paths from this child
                if (node->children[child_idx].pathcount == 0) delete node->children[child_idx].ptr; // Delete child vertex if it has become empty
            }
            if (node->children[child_idx].pathcount == 0) { // Child has become empty
                node->children.erase(child_idx);    // Destroy map entry
                ++rm_potential_pathcount;           // Destroy potential path corresponding to this child (specifically, the child's next new table)
            }
            return rm_potential_pathcount;
        }
        void write_potential_path(mwIndex path_idx, int pp_idx) {  // Writes a potential path's representation to c(path_idx,:). Does not add/remove paths from the tree!
            write_potential_path_helper(head, 0, path_idx, pp_idx);
        }
        void write_potential_path_helper(vertex* node, mwIndex depth, mwIndex path_idx, int pp_idx) {
            int running_count       = 0;
            int prev_count          = 0;
            bool is_existing_path   = false;
            for (map<int,vertex::child>::const_iterator iter = node->children.begin(); iter != node->children.end(); ++iter) {
                running_count   += (*iter).second.potential_pathcount;
                if (running_count > pp_idx) {
                    is_existing_path            = true; // pp_idx refers to an existing child/table at this vertex/restaurant
                    c[depth*model_N + path_idx] = (*iter).first;    // Assign current path branch to temp storage
                    if ((*iter).second.potential_pathcount > 1) {   // If >1 potential path at this child, then recurse
                        write_potential_path_helper((*iter).second.ptr, depth+1, path_idx, pp_idx-prev_count);
                    } else {    // Only one potential path at this child, so write the rest of it out
                        // For the fixed-K implementation and path representation,
                        // the following loop will never execute any iterations, since
                        // only the leaf vertices will have exactly one potential path.
                        // This loop is merely here for a future nonparameteric implementation.
                        for (mwIndex k = depth+1; k < model_K; ++k) c[k*model_N + path_idx] = 1;    // Assign remaining path branches to 1
                    }
                    break;
                }
                prev_count      = running_count;
            }
            if (!is_existing_path) {    // pp_idx refers to the next new table at this vertex/restaurant
                // Assign the current path branch to the first unused index (starting from 1)
                int unused_idx  = 1;
                for (map<int,vertex::child>::const_iterator iter = node->children.begin(); iter != node->children.end(); ++iter) {
                    if (unused_idx < (*iter).first) break;
                    ++unused_idx;
                }
                c[depth*model_N + path_idx] = unused_idx;
                for (mwIndex k = depth+1; k < model_K; ++k) c[k*model_N + path_idx] = 1;    // Assign remaining path branches to 1
            }
        }
        void print_tree() const { // Prints the tree
            mexPrintf("R(%u,%u)\n",pathcount,potential_pathcount);
            print_tree_helper(head,0);
        }
        void print_tree_helper(vertex* node, mwIndex depth) const {
            for (map<int,vertex::child>::const_iterator iter = node->children.begin(); iter != node->children.end(); ++iter) {
                for (int i = 0; i < depth+1; ++i) mexPrintf("\t\t");
                
                // Standard output
                mexPrintf("%u(%u,%u)\n",(*iter).first,(*iter).second.pathcount,(*iter).second.potential_pathcount);
                 
                if (depth < model_K-1) print_tree_helper((*iter).second.ptr, depth+1);    // Recurse if not at max depth
            }
        }
        void delete_tree() {    // Deletes the entire tree, including the head vertex/restaurant
            delete_tree_helper(head);
            delete head;
        }
        void delete_tree_helper(vertex* node) {
            for (map<int,vertex::child>::const_iterator iter = node->children.begin(); iter != node->children.end(); ++iter) {
                if ((*iter).second.ptr != NULL) {
                    delete_tree_helper((*iter).second.ptr);
                    delete (*iter).second.ptr;
                }
            }
        }

        
        /*
         * Main Gibbs sampler. Samples every latent variable once.
         *
         * If num_threads > 1, sampling levels z is done in parallel.
         * The order in which the zs are block-sampled is identical to the
         * single-threaded version.
         */
        void gs_all() {
        iterCount++; if (iterCount % 50 == 0) print_tree();

        // Sample levels z_r, z_l
            for (mwIndex j = 0; j < model_N; ++j) {
                    for (mwIndex i = 0; i < model_N; ++i) {
                        if (i==j) continue;
                        
                        if (rng.rand() < 0.5){
                        B_t* B_el = get_B_element_ptr(i,j);
                        gs_z(Z_R,i,j,z_thread_data);  // Sample donor level z_{i->j}
                        gs_z(Z_L,i,j,z_thread_data);  // Sample receiver level z_{j<-i} = z_l(i,j)
                        }     
                    }
            }
            
            // Sample paths c sequentially
            /*
            for (mwIndex i=0; i < model_N; ++i){
                gs_c(i);
            }
            */
            
            for (mwIndex i=0; i < model_N; ++i){
                if (entity_link_counts[i] > 10){
                    gs_c(i);
                }else if ((entity_link_counts[i] <= 10) && (entity_link_counts[i]   >= 4)){
                    if (rng.rand() < 0.4){
                        gs_c(i);
                    }
                }else{
                    if (rng.rand() < 0.1){
                        gs_c(i);
                    }
                }
            }
        }
        
        /*
         * Generative process for the latent variables c, z_r, and z_l.
         * Assumes empty sufficient statistics for c and z, i.e.
         * 1. The path tree is empty.
         * 2. z_counts is a zero vector.
         *
         * When this function terminates,
         * 1. All sampled paths will be in the path tree.
         * 2. All levels z_r, z_l will be recorded in z_counts.
         *
         * This function can be used to initialize the Gibbs sampler. It
         * is also used by generative_all() and ll_sample().
         */
        void generative_latent() {
            
            // Sample levels z_r, z_l
            if (num_threads == 1) {
                // Single-threaded version
                for (mwIndex i = 0; i < model_N; ++i) {
                    for (mwIndex j = 0; j < model_N; ++j) {
                        if (i==j) continue;
                        generative_z(Z_R,i,j,z_thread_data);  // Sample donor level z_{i->j}
                        generative_z(Z_L,i,j,z_thread_data);  // Sample receiver level z_{j<-i} = z_l(i,j)
                    }
                }
            }
            
            // Sample paths c sequentially
            for (mwIndex i=0; i<model_N; ++i) generative_c(i);
            
            print_tree();
            
        }

        double get_community_relation(mwIndex i, mwIndex j, mwIndex r){
            B_t_r* B_el       = get_B_element_ptr_r(i,j);
            double community_relation = (B_el->get<0>()[r] + lambda1) / (B_el->get<0>()[r] + B_el->get<1>()[r] + lambda1 + lambda2);
            return community_relation;

        }
        
        /*
         * Functions to add and remove edges associated with path i,
         * namely E_{i,.},E{.,i}, from the B_row sufficient stats.
         *
         * These functions assume that the path tree has allocated the
         * B_row's in question. Hence add_path_B() should be called AFTER
         * add_path(), while rm_path_B() should be called BEFORE rm_path().
         */
        void add_path_B(mwIndex i) { modify_path_B(i,true); }
        void rm_path_B(mwIndex i) { modify_path_B(i,false); }
        void modify_path_B(mwIndex i, bool add) {
            //CHANGE
            
            for (mwIndex j = 0; j < model_N; ++j) {
                if (j==i) continue;
                for (unsigned int swap = 0; swap < 2; ++swap) {
                    B_t_r* B_el       = swap?get_B_element_ptr_r(j,i):get_B_element_ptr_r(i,j);
                    for (mwIndex r = 0; r < model_R; ++r){
                        //bool E_value    = swap?E[r*model_N*model_N + i*model_N + j]:E[r*model_N*model_N + j*model_N + i];
                        bool E_value    = swap?bool_lookup(j,i,r):bool_lookup(i,j,r);
                        // Add/remove E(i,j) to/from B_el
                        if (E_value)    B_el->get<0>()[r] += add?1:-1;
                        else            B_el->get<1>()[r] += add?1:-1;
                    }
                }
            }
        }
        
        /*
         * Gibbs sampler for a specific path c (conditioned on all other
         * variables).
         *
         * This function DOES manipulate the B_row sufficient stats,
         * because they are involved in the likelihood probabilities.
         */
        void gs_c(mwIndex path_idx) {
            // Remove E's associated with c(path_idx) from B_row sufficient stats
            rm_path_B(path_idx);
            // Remove c(path_idx) from the path tree
            rm_path(path_idx);
            // Recursively enumerate each potential path for c(path_idx),
            // writing its relative POSTERIOR probability to path_probs
            write_path_probs(head, 0, 0, pathcount, 0, path_idx, POSTERIOR); // NB: This function uses c(path_idx,:) as temp storage

            rng_class::logprobs_to_relprobs(path_probs, 0, potential_pathcount-1);

            // Sample a potential path index from the discrete distribution path_probs
            size_t pp_idx  = rng.rand_discrete(path_probs, 0, potential_pathcount-1);
            // Write the sampled path to c(path_idx,:)
            write_potential_path(path_idx, pp_idx);
            // Add c(path_idx) to the tree
            add_path(path_idx);
            // Add E's associated with c(path_idx) to B_row sufficient stats
            add_path_B(path_idx);
        }
        
        /*
         * Generative process for a specific path c. This function must
         * be called exactly once for each path index path_idx.
         *
         * This function DOES NOT manipulate the B_row sufficient stats,
         * since it only involves prior probabilities.
         */
        void generative_c(mwIndex path_idx) {
            // Recursively enumerate each potential path for c(path_idx),
            // writing its relative PRIOR probability to path_probs
            write_path_probs(head, 0, 0, pathcount, 0, path_idx, PRIOR); // NB: This function uses c(path_idx,:) as temp storage
            rng_class::logprobs_to_relprobs(path_probs, 0, potential_pathcount-1);
            // Sample a potential path index from the discrete distribution path_probs
            size_t pp_idx  = rng.rand_discrete(path_probs, 0, potential_pathcount-1);
            // Write the sampled path to c(path_idx,:)
            write_potential_path(path_idx, pp_idx);
            // Add c(path_idx) to the tree
            add_path(path_idx);
        }
        
        /*
         * Path sampler helper function. Used to compute all potential
         * path prior/posterior probabilities to temp storage.
         *
         * Computes prior probabilities if prob_mode == PRIOR, and
         * computes posterior probabilities if prob_mode == POSTERIOR.
         */
        void write_path_probs(vertex* node, mwIndex depth, int parent_pp_idx,
                              int parent_pathcount, double log_parent_prob,
                              mwIndex path_idx, prob_mode_t prob_mode)
        {   
            double  log_child_prob;                         // Prior child log-probability (log_parent_prob is the prior parent log-probability)
            int     child_start_pp_idx  = parent_pp_idx;    // Initialize child potential path index
            // Recursively compute prior/posterior probabililties for each used path
            for (map<int,vertex::child>::const_iterator iter = node->children.begin(); iter != node->children.end(); ++iter) {
                log_child_prob              = log((double) (*iter).second.pathcount) - log((double) (parent_pathcount + gamma));    // Child/table relative log-probability
                c[depth*model_N + path_idx] = (*iter).first;    // Assign current path branch (we're using c(path_idx,:) to store the current path)
                if ((*iter).second.potential_pathcount > 1) {   // If >1 potential path at this child, then recurse
                    write_path_probs((*iter).second.ptr, depth+1, child_start_pp_idx,
                                     (*iter).second.pathcount, log_parent_prob + log_child_prob,
                                     path_idx, prob_mode);
                } else {    // Only one potential path at this child, so we calculate its probability
                    for (mwIndex k = depth+1; k < model_K; ++k) {
                        // For the fixed-K implementation and path representation,
                        // this for-loop should never execute. It is merely here for
                        // a future nonparameteric-depth implementation.
                        c[k*model_N + path_idx] = 1;    // Assign remaining path branches to 1
                    }
                    path_probs[child_start_pp_idx]  = (prob_mode==POSTERIOR?log_path_E_likelihood(path_idx):0) + log_parent_prob + log_child_prob;    // Store log-probabilities
                }
                child_start_pp_idx  += (*iter).second.potential_pathcount;  // Advance potential path index
            }
            // Compute prior/posterior probability for the unused path
            log_child_prob              = log((double) gamma) - log((double) (parent_pathcount + gamma));   // Unused child/table relative log-probability
            c[depth*model_N + path_idx] = 0;    // Assign current path branch to an unused number (we reserve 0 for this purpose)
            for (mwIndex k = depth+1; k < model_K; ++k) c[k*model_N + path_idx] = 1;    // Assign remaining path branches to 1 by convention
            // Store log-probabilities
            path_probs[child_start_pp_idx]  = 0;
            if (prob_mode==POSTERIOR) {
                // The unused path is not currently represented in the path tree,
                // hence some of the B_row sufficient stats are unallocated.
                // Although these B_row sufficient stats are implictly 0, the
                // likelihood function log_path_E_likelihood() currently does
                // not take this into account. Hence we shall admit a small hack:
                // we first add_path() to allocate the sufficient stats, call
                // log_path_E_likelihood(), and finally rm_path() to revert the tree.
                add_path(path_idx);
                path_probs[child_start_pp_idx]  += log_path_E_likelihood(path_idx);
                rm_path(path_idx);
            }
            path_probs[child_start_pp_idx]  += log_parent_prob + log_child_prob;
        }
        
        /*
         * Gibbs sampler for a specific level in z_r, z_l (conditioned on
         * all other variables).
         *
         * Samples z_r(i,j) if z_mode == Z_R, or z_l(i,j) if z_mode == Z_L.
         *
         * This function DOES manipulate the B_row sufficient stats,
         * because they are involved in the likelihood probabilities.
         *
         * Requires a thread-specific data structure of type z_thread_data_t.
         */
        void gs_z(z_mode_t z_mode, mwIndex i, mwIndex j, z_thread_data_t &ztd) {
            uint16 &level   = z_mode==Z_R ? z_r[j*model_N + i] : z_l[j*model_N + i]; // Reference to z_r(i,j) or z_l(i,j)
            uint16 i_offset = i * (model_K+1);
            
            // Remove E(i,j) or E(j,i) from B_row sufficient stats
            //B_t* B_el       = z_mode==Z_R ? get_B_element_ptr(i,j) : get_B_element_ptr(j,i);
            B_t_r* B_el       = z_mode==Z_R ? get_B_element_ptr_r(i,j) : get_B_element_ptr_r(j,i);
            //CHANGE
            for (mwIndex r = 0; r < model_R; ++r){
                //bool E_value    = z_mode==Z_R ? E[r*model_N*model_N + j*model_N + i] : E[r*model_N*model_N + i*model_N + j];
                bool E_value    = z_mode==Z_R ? bool_lookup(i,j,r) : bool_lookup(j,i,r);
                if (E_value)    --B_el->get<0>()[r];
                else            --B_el->get<1>()[r];
            }
            // Remove z_r(i,j) or z_l(i,j) from z_counts
            --z_counts[i_offset + level];
            // Sample z_r(i,j) or z_l(i,j)
            sample_z(z_mode, i, j, POSTERIOR, ztd);
            // Add the new value of z_r(i,j) or z_l(i,j) to z_counts
            ++z_counts[i_offset + level];
            // Add E(i,j) or E(j,i) to B_row sufficient stats
            B_el            = z_mode==Z_R ? get_B_element_ptr_r(i,j) : get_B_element_ptr_r(j,i);
            //CHANGE
            for (mwIndex r = 0; r < model_R; ++r){
                //bool E_value    = z_mode==Z_R ? E[r*model_N*model_N + j*model_N + i] : E[r*model_N*model_N + i*model_N + j];
                bool E_value    = z_mode==Z_R ? bool_lookup(i,j,r) : bool_lookup(j,i,r);
                if (E_value)    ++B_el->get<0>()[r];
                else            ++B_el->get<1>()[r];
            }
        }
        
        /*
         * Generative process for a specific level in z_r, z_l. This
         * function must be called exactly once for each z_r(i,j) or
         * z_l(i,j).
         *
         * Samples z_r(i,j) if z_mode == Z_R, or z_l(i,j) if z_mode == Z_L.
         *
         * This function DOES NOT manipulate the B_row sufficient stats,
         * since it only involves prior probabilities.
         *
         * Requires a thread-specific data structure of type z_thread_data_t.
         */
        void generative_z(z_mode_t z_mode, mwIndex i, mwIndex j, z_thread_data_t &ztd) {
            uint16 &level   = z_mode==Z_R ? z_r[j*model_N + i] : z_l[j*model_N + i]; // Reference to z_r(i,j) or z_l(i,j)
            uint16 i_offset = i * (model_K+1);
            
            // Sample z_r(i,j) or z_l(i,j)
            sample_z(z_mode, i, j, PRIOR, ztd);
            // Add the new value of z_r(i,j) or z_l(i,j) to z_counts
            
            ++z_counts[i_offset + level];

        }
        
        /*
         * Samples z_r(i,j) or z_l(i,j), depending on the value of z_mode
         * (either Z_R or Z_L). The samples are taken based on the current
         * level counts z_counts, and written directly to z_r(i,j) or
         * z_l(i,j).
         *
         * Uses prior probabilities if prob_mode == PRIOR, and uses
         * posterior probabilities if prob_mode == POSTERIOR.
         *
         * Requires a thread-specific data structure of type z_thread_data_t.
         */
        void sample_z(z_mode_t z_mode, mwIndex i, mwIndex j,
                      prob_mode_t prob_mode, z_thread_data_t &ztd)
        {   
            // Initialization
            uint16 *this_z          = z_mode == Z_R ? z_r : z_l;
            unsigned int i_offset   = i * (model_K+1);
            rng_class &ext_rng      = ztd.rng;
            vector<double> &lgc     = ztd.level_greater_counts;
            vector<double> &lp      = ztd.level_probs;
            vector<double> &lpv     = ztd.level_prod_vector;    // Represents sum(log(V)) up to a given index

            // Compute level counts >= a given index
            lgc[model_K+1]  = 0;    // No one has level > K
            for (uint16 d = model_K; d >= 1; --d) lgc[d] = lgc[d+1] + z_counts[i_offset + d];

            
            // Compute unnormalized prior/posterior probabilities from level 1 through K
            lpv[0]              = 0;
            double prior_sum    = 0;    // Used to compute the prior probability of drawing a level > K. For this fixed K implementation, we have no need of it.

            //double temp_qqq = rng.rand();
            for (uint16 d = 1; d <= model_K; ++d) {
                // Compute prior log-probability
                double log_V    = log(m*pi + z_counts[i_offset + d]) - log(pi + lgc[d]);
                double log_prob = log_V + lpv[d-1];
                prior_sum       += exp(log_prob);
                lpv[d]          = log(1-exp(log_V)) + lpv[d-1];

                //if (temp_qqq < 0.001) mexPrintf("\ni = %d j = %d level %d prior = %f ", i, j, d, exp(log_prob));
                
                if (prob_mode == POSTERIOR) {
                                               // Add the log-likelihood to log_prob. This block of
                                                // code is not trivially parallelizable, hence we only
                                                // parallelize the generative process on z (which does
                                                // not execute this block).
                    this_z[j*model_N + i]   = d;    // log_E_likelihood() reads the level directly from this_z
                    log_prob                += z_mode == Z_R ? log_E_likelihood(i,j) : log_E_likelihood(j,i);
                    //if (temp_qqq < 0.001) mexPrintf("\tlikelihood = %f\n", z_mode == Z_R ? log_E_likelihood(i,j) : log_E_likelihood(j,i));
                }
                
                // Store the prior/posterior probability
                lp[d]           = log_prob; // Store log-probabilities first, and convert to relative probabilities later
            }

            // Sample the new level z (ignoring the event that z > K)
            rng_class::logprobs_to_relprobs(lp,1,model_K);
            this_z[j*model_N + i]   = static_cast<uint16>( ext_rng.rand_discrete(lp,1,model_K) );   
      
        }
        
        /*
         * Computes the complete log-likelihood log(P(E,c,z)), based on
         * the current state of the path tree and z_counts (but not the
         * B_row's).
         */
        
        double log_complete_likelihood() const {
            double log_likelihood = log_all_E_likelihood() + log_prior_c() + log_prior_z();
            //mexPrintf("%f \n", log_likelihood);
            return log_likelihood;
            //return log_all_E_likelihood() + log_prior_c() + log_prior_z();
        }
        
        /*
         * Functions to query hyperparameter values.
         */
        double get_gamma() const { return gamma; }
        double get_lambda1() const { return lambda1; }
        double get_lambda2() const { return lambda2; }
        double get_m() const { return m; }
        double get_pi() const { return pi; }
        

    };
}
#endif
