functions{
    real partial_sum(int[,] det_slice, 
                     int start, int end, 
                     int n_visit, 
                     vector b0, 
                     vector b1, 
                     vector d0, 
                     vector d1, 
                     vector env_var, 
                     row_vector[] vis_cov, 
                     int[] id_sp, 
                     int[] id_pt, 
                     int[] Q) {
        // indexing
        int len = end - start + 1;
        int r0 = start - 1;
        
        vector[len] lp;
        real logit_psi;
        row_vector[n_visit] logit_theta;
        
        for (r in 1:len){
            logit_psi = b0[id_sp[r0+r]] + b1[id_sp[r0+r]] * env_var[id_pt[r0+r]];
            logit_theta = d0[id_sp[r0+r]] + d1[id_sp[r0+r]] * vis_cov[r0+r];
            if (Q[r0+r] == 1) 
                lp[r] = log_inv_logit(logit_psi) +
                    bernoulli_logit_lpmf(det_slice[r] | logit_theta);
            else lp[r] = log_sum_exp(
                log_inv_logit(logit_psi) +
                    log1m_inv_logit(logit_theta[1]) +
                    log1m_inv_logit(logit_theta[2]) +
                    log1m_inv_logit(logit_theta[3]) +
                    log1m_inv_logit(logit_theta[4]),
                log1m_inv_logit(logit_psi));
        }
        return sum(lp);
    }
}
data {
    int<lower=1> n_tot;
    int<lower=1> n_visit;
    int<lower=1> n_species;
    int<lower=1> n_point;
    
    int<lower=1> id_sp[n_tot];
    int<lower=1> id_pt[n_tot];
    int det[n_tot, n_visit];
    row_vector[n_visit] vis_cov[n_tot];
    int Q[n_tot];
    vector[n_point] env_var;
    int<lower=1> grainsize;
}
parameters {
    // psi: occupancy 
    real mu_b0;
    real<lower=0> sigma_b0;
    vector[n_species] b0_raw;
    
    real mu_b1;
    real<lower=0> sigma_b1;
    vector[n_species] b1_raw;
    
    // theta: detection
    real mu_d0; 
    real<lower=0> sigma_d0;
    vector[n_species] d0_raw;
    
    real mu_d1; 
    real<lower=0> sigma_d1;
    vector[n_species] d1_raw;
    
}
transformed parameters{
    // scaling
    vector[n_species] b0 = mu_b0 + b0_raw * sigma_b0;
    vector[n_species] b1 = mu_b1 + b1_raw * sigma_b1;
    vector[n_species] d0 = mu_d0 + d0_raw * sigma_d0;
    vector[n_species] d1 = mu_d1 + d1_raw * sigma_d1;
}
model {
    target += reduce_sum_static(partial_sum, det, grainsize, n_visit, b0, b1, d0, 
                                d1, env_var, vis_cov, id_sp, id_pt, Q);
                                
    // priors
    mu_b0 ~ normal(0, 10);
    mu_b1 ~ normal(0, 10);
    mu_d0 ~ normal(0, 10);
    mu_d1 ~ normal(0, 10);
    
    sigma_b0 ~ normal(0, 10);
    sigma_b1 ~ normal(0, 10);
    sigma_d0 ~ normal(0, 10);
    sigma_d1 ~ normal(0, 10);
    
    b0_raw ~ normal(0,1);
    b1_raw ~ normal(0,1);
    d0_raw ~ normal(0,1);
    d1_raw ~ normal(0,1);
    
    
}
