data {
    int<lower=1> n_tot; // total number of rows in df
    int<lower=1> n_visit; // width of df (number of visits)
    int<lower=1> n_species;
    int<lower=1> n_point; // number of points sampled at
    
    int<lower=1> id_sp[n_tot]; //an index (length n_tot) for species identity
    int<lower=1> id_pt[n_tot]; //an index (length n_tot) for point identity
    int det[n_tot, n_visit]; //an index (length n_tot) for point identity
    row_vector[n_visit] vis_cov[n_tot]; // visit covariate (time of day)
    vector[n_tot] Q; // indexes whether a species has been observed at the point
    vector[n_point] env_var;
    
}
parameters {
    // occupancy intercept-by-species
    real mu_b0;
    real<lower=0> sigma_b0;
    vector[n_species] b0_raw;
    // occupancy slope-by-species
    real mu_b1;
    real<lower=0> sigma_b1;
    vector[n_species] b1_raw;
    
    // detection intercept-by-species
    real mu_d0; 
    real<lower=0> sigma_d0;
    vector[n_species] d0_raw;
    // detection slope-by-species
    real mu_d1; 
    real<lower=0> sigma_d1;
    vector[n_species] d1_raw;
    
}
transformed parameters{
    // sorting out non-centering
    vector[n_species] b0 = mu_b0 + b0_raw * sigma_b0;
    vector[n_species] b1 = mu_b1 + b1_raw * sigma_b1;
    vector[n_species] d0 = mu_d0 + d0_raw * sigma_d0;
    vector[n_species] d1 = mu_d1 + d1_raw * sigma_d1;
}
model {
    vector[n_tot] lp;
    real logit_psi;
    row_vector[n_visit] logit_theta;
    
    for (i in 1:n_tot){
        logit_psi = b0[id_sp[i]] + b1[id_sp[i]] * env_var[id_pt[i]];
        logit_theta = d0[id_sp[i]] + d1[id_sp[i]] * vis_cov[i];
        
        if (Q[i] == 1) 
            lp[i] = log_inv_logit(logit_psi) +
                bernoulli_logit_lpmf(det[i] | logit_theta[i]);
        else lp[i] = log_sum_exp(
            log_inv_logit(logit_psi) +
                log1m_inv_logit(logit_theta[1]) +
                log1m_inv_logit(logit_theta[2]) +
                log1m_inv_logit(logit_theta[3]) +
                log1m_inv_logit(logit_theta[4]),
            log1m_inv_logit(logit_psi));
    }
    
    target += sum(lp);
    
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
