#ifndef HB_HPP_
#define HB_HPP_

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
    
    typedef unsigned short uint16;  
    
    enum prob_mode_t {PRIOR, POSTERIOR};    
    enum z_mode_t {Z_R, Z_L};               
    
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
            double max_log = *max_element( distrib.begin()+begin , distrib.begin()+end+1 ); 
            for (size_t i = begin; i <= end; ++i) distrib[i] = exp(distrib[i] - max_log);   
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
     * Collapsed Gibbs sampler class for the model
     */
    class sampler_class {
        
        typedef boost::tuple<unsigned int,unsigned int> B_t;   
        typedef boost::tuple<unsigned int[1000],unsigned int[1000]> B_t_r;
        
        /*
         * Structure to contain community information
         */
        struct vertex {
            struct child {  
                vertex  *ptr;                  
                int pathcount;              
                int potential_pathcount;    
                std::map<int,B_t> B_row; 
                std::map<int,B_t_r> B_row_r;
                child() : ptr(NULL), pathcount(0), potential_pathcount(0) {}
            };
            map<int,child>  children;   
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
                
        // Constants
        std::unordered_set< int > coordinates;
        std::vector< int > entity_link_counts; 
        uint16          *c, *z_r, *z_l; 
        
        // Model parameters
        double          m, pi, gamma, lambda1, lambda2;
        const mwSize    model_N, model_R, model_L;
        
        // Tree data structure (sufficient statistics for paths and blockmatrices B)
        vertex          *head;
        int             pathcount;
        int             potential_pathcount;
        
        // Temporary variables for sampling paths 
        vector<double>  path_probs;     
        
        // Sufficient statistics for levels z
        vector<unsigned int>    z_counts;               
        
        // Main thread temporary variables for sampling levels z
        vector<double>  level_greater_counts;   
        vector<double>  level_probs;            
        vector<double>  level_prod_vector;      
        
        // RNG
        rng_class       rng;
        
        // Thread-related
        unsigned int    num_threads;
        z_thread_data_t z_thread_data;  // Contains references to thread-specific objects necessary for sampling z (for the main thread)
        
        // Miscellaneous/debugging
        int             iterCount;

        /* 
         * Returns true if link exists from entity i to entity j on predicate r, otherwise false
         */
        bool bool_lookup(mwIndex i, mwIndex j, mwIndex r) const{
            return (coordinates.find(r*model_N*model_N + j*model_N + i)  != coordinates.end() );
        }
        
        
        /*
         * Return the pointer to the corresponding B values after Psi coarsening
         */
        /*
        B_t* get_B_element_ptr(mwIndex i, mwIndex j) { return const_cast<B_t*>(get_B_element_ptr_helper(i,j)); }
        const B_t* get_B_element_ptr(mwIndex i, mwIndex j) const { return get_B_element_ptr_helper(i,j); }
        const B_t* get_B_element_ptr_helper(mwIndex i, mwIndex j) const {
            
            unsigned int z_coarse = min(z_r[j*model_N + i],z_l[i*model_N + j]); 
            mwIndex ell = 1;
            vertex *cur_vertex = head;  
            for (; ell < z_coarse; ++ell) {
                if (c[(ell-1)*model_N + i] != c[(ell-1)*model_N + j]) {
                    break;
                }
                cur_vertex = cur_vertex->children[(int) c[(ell-1)*model_N + i]].ptr;
            }
            return &cur_vertex->children[(int) c[(ell-1)*model_N + i]].B_row[(int) c[(ell-1)*model_N + j]];
        }
        */

        /*
         * Return the pointer to the corresponding B values after Psi coarsening
         */
        B_t_r* get_B_element_ptr_r(mwIndex i, mwIndex j) { return const_cast<B_t_r*>(get_B_element_ptr_r_helper(i,j)); }
        const B_t_r* get_B_element_ptr_r(mwIndex i, mwIndex j) const { return get_B_element_ptr_r_helper(i,j); }
        const B_t_r* get_B_element_ptr_r_helper(mwIndex i, mwIndex j) const {
        
            
            unsigned int z_coarse = min(z_r[j*model_N + i],z_l[i*model_N + j]); 
            double rrr = ((double) rand() / (RAND_MAX));

            mwIndex ell = 1;
            vertex *cur_vertex = head;  
            for (; ell < z_coarse; ++ell) {
                if (c[(ell-1)*model_N + i] != c[(ell-1)*model_N + j]) { 
                    break;
                }
                cur_vertex = cur_vertex->children[(int) c[(ell-1)*model_N + i]].ptr;
            }
            return &cur_vertex->children[(int) c[(ell-1)*model_N + i]].B_row_r[(int) c[(ell-1)*model_N + j]];
        }
        
        /*
         * Computes log likelihood
         */
         double log_E_likelihood(mwIndex i, mwIndex j) const {

            const B_t_r* B_el = get_B_element_ptr_r(i,j);
            
            // Likelihood is a ratio of two normalizers with gamma functions;
            // we use an equivalent expression without the gamma functions.
            double likelihood_sum = 0;
            
            for (int r = 0; r < model_R; ++r){
                bool E_value    = bool_lookup(i,j,r); 
                likelihood_sum = log( E_value*(B_el->get<0>()[r]+lambda1) + (1-E_value)*(B_el->get<1>()[r]+lambda2) )
                - log( B_el->get<0>()[r] + B_el->get<1>()[r] + lambda1 + lambda2 );
            }
            return likelihood_sum;
        }
        
        /*
         * Computes the conditional log-likelihood of edges related to
         * path 
         */
        double log_path_E_likelihood(mwIndex i) const {
            
            double ll   = 0;
            
            typedef std::map<const B_t_r*,B_t_r>    B_row_ptr_r_map_t;
            B_row_ptr_r_map_t                     B_row_ptr_r_map;
            
            for (mwIndex j = 0; j < model_N; ++j) {
                if (j==i) continue;
                for (unsigned int swap = 0; swap < 2; ++swap) {
                    const B_t_r* B_el = swap?get_B_element_ptr_r(j,i):get_B_element_ptr_r(i,j);
                    for (mwIndex r = 0; r < model_R; ++r){
                        bool E_value = swap?bool_lookup(j,i,r):bool_lookup(i,j,r);
                        if (E_value)    ++B_row_ptr_r_map[B_el].get<0>()[r];
                        else            ++B_row_ptr_r_map[B_el].get<1>()[r];
                    }
                }
            }
            
            for (B_row_ptr_r_map_t::iterator it = B_row_ptr_r_map.begin(), end = B_row_ptr_r_map.end();
                 it != end;
                 ++it) {
                const B_t_r *B_el = it->first;
                B_t_r &counts     = it->second;
                
                
                for (mwIndex r = 0; r < model_R; ++r){
                    // Likelihood is a ratio of two normalizers
                    ll +=
                        rng_class::ln_gamma(B_el->get<0>()[r] + B_el->get<1>()[r] + lambda1 + lambda2)
                        - rng_class::ln_gamma(B_el->get<0>()[r] + lambda1)
                        - rng_class::ln_gamma(B_el->get<1>()[r] + lambda2)
                        + rng_class::ln_gamma(B_el->get<0>()[r] + counts.get<0>()[r] + lambda1)
                        + rng_class::ln_gamma(B_el->get<1>()[r] + counts.get<1>()[r] + lambda2)
                        - rng_class::ln_gamma(B_el->get<0>()[r] + B_el->get<1>()[r] + counts.get<0>()[r] + counts.get<1>()[r] + lambda1 + lambda2);
                }
                
            }
            return ll;
        }
        
        /*
         * Computes the conditional log-likelihood
         */
        double log_all_E_likelihood() const {
                     
            double ll   = 0;
            
            typedef std::map<const B_t_r*,B_t_r>    B_row_ptr_r_map_t;
            B_row_ptr_r_map_t                     B_row_ptr_r_map;
            for (mwIndex i = 0; i < model_N; ++i) {
                for (mwIndex j = 0; j < model_N; ++j) {
                    if (j==i) continue;
                    const B_t_r* B_el = get_B_element_ptr_r(i,j);
                    for (mwIndex r = 0; r < model_R; ++r){
                        bool E_value = bool_lookup(i,j,r);
                    
                        if (E_value)    ++B_row_ptr_r_map[B_el].get<0>()[r];
                        else            ++B_row_ptr_r_map[B_el].get<1>()[r];
                    }
                }
            }
            
            for (B_row_ptr_r_map_t::iterator it = B_row_ptr_r_map.begin(), end = B_row_ptr_r_map.end(); it != end; ++it) {
                B_t_r &counts     = it->second;
                
                for (mwIndex r = 0; r < model_R; ++r){
                    // Likelihood is a ratio of two normalizers
                    ll +=
                        rng_class::ln_gamma(lambda1 + lambda2)
                        - rng_class::ln_gamma(lambda1)
                        - rng_class::ln_gamma(lambda2)
                        + rng_class::ln_gamma(counts.get<0>()[r] + lambda1)
                        + rng_class::ln_gamma(counts.get<1>()[r] + lambda2)
                        - rng_class::ln_gamma(counts.get<0>()[r] + counts.get<1>()[r] + lambda1 + lambda2);
                }
            }
            
            return ll;
        }
        
        /*
         * Computes the log-prior of paths
         */
        double log_prior_c() const { return log_prior_c_helper(head, 0); }
        double log_prior_c_helper(vertex* node, mwIndex depth) const {
            double ll = 0;
            unsigned int tot_paths = 0;
            for (map<int,vertex::child>::const_iterator it = node->children.begin(), end = node->children.end();
                 it != end;
                 ++it) {
                ll += log(gamma) - log((tot_paths++)+gamma);
                for (unsigned int x = 1; x < it->second.pathcount; ++x) ll += log((double) x) - log((tot_paths++)+gamma);
                if (depth < model_L-1) {
                    ll += log_prior_c_helper(it->second.ptr, depth+1);
                }
            }
            return ll;
        }
        
        /*
         * Computes the log-prior of levels
         */
        double log_prior_z() const {
            double ll = 0;
            for (mwIndex i = 0; i < model_N; ++i) ll += log_prior_z_helper(i);
            return ll;
        }
        double log_prior_z_helper(mwIndex i) const {
            vector<double> lgc(model_L+2);
            vector<double> lp(model_L+2);
            vector<double> lpv(model_L+2);  
            double ll               = 0;            
            unsigned int i_offset   = i * (model_L+1);
            
            for (uint16 ell = model_L; ell >= 1; --ell) {   
                for (unsigned int x = 0; x < z_counts[i_offset + ell]; ++x) {   
                    
                    lpv[0]              = 0;

                    for (uint16 d = 1; d <= model_L; ++d) {

                        double level_count          = lgc[d] - lgc[d+1];
                        double log_V                = log(m*pi + level_count) - log(pi + lgc[d]);
                        double log_prob             = log_V + lpv[d-1];

                        lpv[d]                      = log(1-exp(log_V)) + lpv[d-1];

                        lp[d]                       = log_prob;
                    }
                    
                    double denominator  = 0;
                    for (uint16 d = 1; d <= model_L; ++d) denominator += exp(lp[d]);
                    ll += lp[ell] - log(denominator);

                    for (uint16 d = 1; d <= ell; ++d) ++lgc[d];
                }
            }
            
            return ll;
        }
   
    public:
        /*
         * Constructor.
         */
        sampler_class(uint16 *ci, uint16 *z_ri, uint16 *z_li,
                 double mi, double pii, double gammai,
                 double lambda1i, double lambda2i,
                 mwSize model_Ni, mwSize model_Ri, mwSize model_Li,
                 unsigned int SEED, unsigned int num_threadsi, std::unordered_set< int > coordinatesi, std::vector< int > entity_link_countsi)
                 : 
                   coordinates(coordinatesi), entity_link_counts(entity_link_countsi), c(ci), z_r(z_ri), z_l(z_li),
                   
                   // Model parameters
                   m(mi), pi(pii), gamma(gammai), lambda1(lambda1i),
                   lambda2(lambda2i), model_N(model_Ni), model_R(model_Ri), model_L(model_Li),
                   
                   // Path sufficient statistics
                   path_probs(model_Ni*model_Li),      
                   
                   // Level sufficient statistics
                   z_counts(model_Ni*(model_Li+1),0),   
                   level_greater_counts(model_Li+2),    
                   level_probs(model_Li+2),             
                   level_prod_vector(model_Li+2),       
                   
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
                add_path(i);    
                for (mwIndex j = 0; j < model_N; ++j) {
                    if (j==i) continue;
                    ++z_counts[i*(model_L+1) + z_r[j*model_N + i]]; 
                    ++z_counts[i*(model_L+1) + z_l[j*model_N + i]]; 
                }
            }
        }
        void initialize_gs_ss_observed() {
            
            for (mwIndex i = 0; i < model_N; ++i) {
                for (mwIndex j = 0; j < model_N; ++j) { 
                    if (j==i) continue;
                    B_t_r* B_el       = get_B_element_ptr_r(i,j);
                    for (mwIndex r = 0; r < model_R; ++r){
                        bool E_value    = bool_lookup(i,j,r);
                        if (E_value)    ++B_el->get<0>()[r];
                        else            ++B_el->get<1>()[r];
                    }
                }
            }
        }
        
        /*
         * Path-realted functions
         */
        void create_empty_tree() { 
            head                = new vertex;
            pathcount           = 0;    
            potential_pathcount = 1;    
        }
        void add_path(mwIndex path_idx) {  
            ++pathcount;
            potential_pathcount += add_path_helper(head, 0, path_idx);
        }
        int add_path_helper(vertex* node, mwIndex depth, mwIndex path_idx) {    
            int child_idx               = (int) c[depth*model_N + path_idx];    
            int add_potential_pathcount = 0;
            if (node->children.count(child_idx) == 0) { 
                node->children[child_idx].pathcount             = 0;
                node->children[child_idx].potential_pathcount   = 1;
                ++add_potential_pathcount;                              
            }
            ++node->children[child_idx].pathcount;
            if (depth < model_L-1) {    
                if (node->children[child_idx].ptr == NULL) node->children[child_idx].ptr = new vertex;  
                int child_add_potential_pathcount               = add_path_helper(node->children[child_idx].ptr, depth+1, path_idx);
                node->children[child_idx].potential_pathcount   += child_add_potential_pathcount;
                add_potential_pathcount                         += child_add_potential_pathcount;
            }
            return add_potential_pathcount;
        }
        void rm_path(mwIndex path_idx) {  
            --pathcount;
            potential_pathcount -= rm_path_helper(head, 0, path_idx);
        }
        int rm_path_helper(vertex* node, mwIndex depth, mwIndex path_idx) {    
            int child_idx               = (int) c[depth*model_N + path_idx];    
            int rm_potential_pathcount  = 0;
            --node->children[child_idx].pathcount;
            if (depth < model_L-1) {   
                int child_rm_potential_pathcount                = rm_path_helper(node->children[child_idx].ptr, depth+1, path_idx);
                node->children[child_idx].potential_pathcount   -= child_rm_potential_pathcount;
                rm_potential_pathcount                          += child_rm_potential_pathcount;    
                if (node->children[child_idx].pathcount == 0) delete node->children[child_idx].ptr; 
            }
            if (node->children[child_idx].pathcount == 0) { 
                node->children.erase(child_idx);    
                ++rm_potential_pathcount;           
            }
            return rm_potential_pathcount;
        }
        void write_potential_path(mwIndex path_idx, int pp_idx) {  
            write_potential_path_helper(head, 0, path_idx, pp_idx);
        }
        void write_potential_path_helper(vertex* node, mwIndex depth, mwIndex path_idx, int pp_idx) {
            int running_count       = 0;
            int prev_count          = 0;
            bool is_existing_path   = false;
            for (map<int,vertex::child>::const_iterator iter = node->children.begin(); iter != node->children.end(); ++iter) {
                running_count   += (*iter).second.potential_pathcount;
                if (running_count > pp_idx) {
                    is_existing_path            = true; 
                    c[depth*model_N + path_idx] = (*iter).first;   
                    if ((*iter).second.potential_pathcount > 1) {   
                        write_potential_path_helper((*iter).second.ptr, depth+1, path_idx, pp_idx-prev_count);
                    } else {    
                        for (mwIndex l = depth+1; l < model_L; ++l) c[l*model_N + path_idx] = 1;   
                    }
                    break;
                }
                prev_count      = running_count;
            }
            if (!is_existing_path) {    
                int unused_idx  = 1;
                for (map<int,vertex::child>::const_iterator iter = node->children.begin(); iter != node->children.end(); ++iter) {
                    if (unused_idx < (*iter).first) break;
                    ++unused_idx;
                }
                c[depth*model_N + path_idx] = unused_idx;
                for (mwIndex l = depth+1; l < model_L; ++l) c[l*model_N + path_idx] = 1;    
            }
        }
        void print_tree() const { 
            mexPrintf("R(%u,%u)\n",pathcount,potential_pathcount);
            print_tree_helper(head,0);
        }
        void print_tree_helper(vertex* node, mwIndex depth) const {
            for (map<int,vertex::child>::const_iterator iter = node->children.begin(); iter != node->children.end(); ++iter) {
                for (int i = 0; i < depth+1; ++i) mexPrintf("\t\t");
                
                mexPrintf("%u(%u,%u)\n",(*iter).first,(*iter).second.pathcount,(*iter).second.potential_pathcount);
                 
                if (depth < model_L-1) print_tree_helper((*iter).second.ptr, depth+1);    
            }
        }
        void delete_tree() {  
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
         */
        void gs_all() {
        iterCount++; if (iterCount % 50 == 0) print_tree();

        for (mwIndex j = 0; j < model_N; ++j) {
                for (mwIndex i = 0; i < model_N; ++i) {
                    if (i==j) continue;
                    
                    if (rng.rand() < 0.5){
                        B_t_r* B_el = get_B_element_ptr_r(i,j);
                        gs_z(Z_R,i,j,z_thread_data);  
                        gs_z(Z_L,i,j,z_thread_data);  
                    }     
                }
            }
            
            // Sample paths c sequentially
            for (mwIndex i=0; i < model_N; ++i){
                gs_c(i);
            }

            // Sample paths stochastically
            /*
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
            */
        }
        
        /*
         * Generative process for the latent path and level variables
         */
        void generative_latent() {
            
            if (num_threads == 1) {
                for (mwIndex i = 0; i < model_N; ++i) {
                    for (mwIndex j = 0; j < model_N; ++j) {
                        if (i==j) continue;
                        generative_z(Z_R,i,j,z_thread_data);  
                        generative_z(Z_L,i,j,z_thread_data);  
                    }
                }
            }
            
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
         */
        void add_path_B(mwIndex i) { modify_path_B(i,true); }
        void rm_path_B(mwIndex i) { modify_path_B(i,false); }
        void modify_path_B(mwIndex i, bool add) {
                        
            for (mwIndex j = 0; j < model_N; ++j) {
                if (j==i) continue;
                for (unsigned int swap = 0; swap < 2; ++swap) {
                    B_t_r* B_el       = swap?get_B_element_ptr_r(j,i):get_B_element_ptr_r(i,j);
                    for (mwIndex r = 0; r < model_R; ++r){

                        bool E_value    = swap?bool_lookup(j,i,r):bool_lookup(i,j,r);
                        if (E_value)    B_el->get<0>()[r] += add?1:-1;
                        else            B_el->get<1>()[r] += add?1:-1;
                    }
                }
            }
        }
        
        /*
         * Gibbs sampler for a specific path c (conditioned on all other
         * variables)
         */
        void gs_c(mwIndex path_idx) {

            rm_path_B(path_idx);

            rm_path(path_idx);

            write_path_probs(head, 0, 0, pathcount, 0, path_idx, POSTERIOR);

            rng_class::logprobs_to_relprobs(path_probs, 0, potential_pathcount-1);

            size_t pp_idx  = rng.rand_discrete(path_probs, 0, potential_pathcount-1);

            write_potential_path(path_idx, pp_idx);

            add_path(path_idx);

            add_path_B(path_idx);
        }
        
        /*
         * Generative process for a specific path 
         */
        void generative_c(mwIndex path_idx) {

            write_path_probs(head, 0, 0, pathcount, 0, path_idx, PRIOR); 
            rng_class::logprobs_to_relprobs(path_probs, 0, potential_pathcount-1);

            size_t pp_idx  = rng.rand_discrete(path_probs, 0, potential_pathcount-1);

            write_potential_path(path_idx, pp_idx);

            add_path(path_idx);
        }
        
        /*
         * Path sampler helper function. Used to compute all potential path prior/posterior probabilities to temp storage
         */
        void write_path_probs(vertex* node, mwIndex depth, int parent_pp_idx, int parent_pathcount, double log_parent_prob, mwIndex path_idx, prob_mode_t prob_mode)
        {   
            double  log_child_prob;                         
            int     child_start_pp_idx  = parent_pp_idx;    

            for (map<int,vertex::child>::const_iterator iter = node->children.begin(); iter != node->children.end(); ++iter) {
                log_child_prob = log((double) (*iter).second.pathcount) - log((double) (parent_pathcount + gamma));   
                c[depth*model_N + path_idx] = (*iter).first;    
                if ((*iter).second.potential_pathcount > 1) {   
                    write_path_probs((*iter).second.ptr, depth+1, child_start_pp_idx,
                                     (*iter).second.pathcount, log_parent_prob + log_child_prob,
                                     path_idx, prob_mode);
                } else {    
                    for (mwIndex l = depth+1; l < model_L; ++l) {
                        c[l*model_N + path_idx] = 1; 
                    }
                    path_probs[child_start_pp_idx] = (prob_mode==POSTERIOR?log_path_E_likelihood(path_idx):0) + log_parent_prob + log_child_prob;    
                }
                child_start_pp_idx  += (*iter).second.potential_pathcount;
            }
            
            log_child_prob = log((double) gamma) - log((double) (parent_pathcount + gamma));   
            c[depth*model_N + path_idx] = 0;
            for (mwIndex l = depth+1; l < model_L; ++l) c[l*model_N + path_idx] = 1;
            path_probs[child_start_pp_idx]  = 0;
            if (prob_mode==POSTERIOR) {
                add_path(path_idx);
                path_probs[child_start_pp_idx]  += log_path_E_likelihood(path_idx);
                rm_path(path_idx);
            }
            path_probs[child_start_pp_idx]  += log_parent_prob + log_child_prob;
        }
        
        /*
         * Gibbs sampler for a specific level in level indicators
         */
        void gs_z(z_mode_t z_mode, mwIndex i, mwIndex j, z_thread_data_t &ztd) {
            uint16 &level   = z_mode==Z_R ? z_r[j*model_N + i] : z_l[j*model_N + i]; 
            uint16 i_offset = i * (model_L+1);
            
            B_t_r* B_el       = z_mode==Z_R ? get_B_element_ptr_r(i,j) : get_B_element_ptr_r(j,i);
            for (mwIndex r = 0; r < model_R; ++r){
                bool E_value    = z_mode==Z_R ? bool_lookup(i,j,r) : bool_lookup(j,i,r);
                if (E_value)    --B_el->get<0>()[r];
                else            --B_el->get<1>()[r];
            }
            --z_counts[i_offset + level];
            sample_z(z_mode, i, j, POSTERIOR, ztd);
            ++z_counts[i_offset + level];
            B_el            = z_mode==Z_R ? get_B_element_ptr_r(i,j) : get_B_element_ptr_r(j,i);
            for (mwIndex r = 0; r < model_R; ++r){
                bool E_value    = z_mode==Z_R ? bool_lookup(i,j,r) : bool_lookup(j,i,r);
                if (E_value)    ++B_el->get<0>()[r];
                else            ++B_el->get<1>()[r];
            }
        }
        
        /*
         * Generative process for a specific level in level indicators
         */
        void generative_z(z_mode_t z_mode, mwIndex i, mwIndex j, z_thread_data_t &ztd) {
            uint16 &level   = z_mode==Z_R ? z_r[j*model_N + i] : z_l[j*model_N + i]; 
            uint16 i_offset = i * (model_L+1);
            
            sample_z(z_mode, i, j, PRIOR, ztd);
            
            ++z_counts[i_offset + level];

        }
        
        /*
         * Samples level indicators
         */
        void sample_z(z_mode_t z_mode, mwIndex i, mwIndex j,
                      prob_mode_t prob_mode, z_thread_data_t &ztd)
        {   
            // Initialization
            uint16 *this_z          = z_mode == Z_R ? z_r : z_l;
            unsigned int i_offset   = i * (model_L+1);
            rng_class &ext_rng      = ztd.rng;
            vector<double> &lgc     = ztd.level_greater_counts;
            vector<double> &lp      = ztd.level_probs;
            vector<double> &lpv     = ztd.level_prod_vector;    

            lgc[model_L+1]  = 0;   
            for (uint16 d = model_L; d >= 1; --d) lgc[d] = lgc[d+1] + z_counts[i_offset + d];

            lpv[0]              = 0;
            double prior_sum    = 0;    

            for (uint16 d = 1; d <= model_L; ++d) {
                double log_V    = log(m*pi + z_counts[i_offset + d]) - log(pi + lgc[d]);
                double log_prob = log_V + lpv[d-1];
                prior_sum       += exp(log_prob);
                lpv[d]          = log(1-exp(log_V)) + lpv[d-1];
                
                if (prob_mode == POSTERIOR) {
                    this_z[j*model_N + i] = d;   
                    log_prob += z_mode == Z_R ? log_E_likelihood(i,j) : log_E_likelihood(j,i);
                }
                
                lp[d] = log_prob;
            }

            rng_class::logprobs_to_relprobs(lp,1,model_L);
            this_z[j*model_N + i]   = static_cast<uint16>( ext_rng.rand_discrete(lp,1,model_L) );   
      
        }
        
        /*
         * Computes the complete log-likelihood
         */
        
        double log_complete_likelihood() const {
            double log_likelihood = log_all_E_likelihood() + log_prior_c() + log_prior_z();
            return log_likelihood;
        }
        
        /*
         * Functions to query hyperparameter values.
         */
        double get_gamma() const { return gamma; }
        double get_lambda() const { return lambda1; }
        double get_eta() const { return lambda2; }
        double get_mu() const { return m; }
        double get_sigma() const { return pi; }
        

    };
}
#endif
