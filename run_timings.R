## simulate some simple 2D occupancy data
library("dplyr"); library("boot"); library("reshape2"); library("cmdstanr")

# define dimensions
n_site <- 50
n_species <- 50
n_visit <- 4

## hyperparameters
b0_mu <- 0
b1_mu <- .5
b0_sigma <- .5
b1_sigma <- .5

# detection
d0_mu <- 0
d1_mu <- -1
d0_sigma <- .5
d1_sigma <- .5

# parameters
set.seed(100)
b0 <- rnorm(n_species, b0_mu, b0_sigma)
b1 <- rnorm(n_species, b1_mu, b1_sigma)
d0 <- rnorm(n_species, d0_mu, d0_sigma)
d1 <- rnorm(n_species, d1_mu, d1_sigma)

# simulate data
env_var <- rnorm(n_site)
df_sim <- expand.grid(i = 1:n_site, 
                      j = 1:n_visit, 
                      k = 1:n_species) %>%
    as_tibble %>%
    mutate(time = runif(n(), 0, 1), 
           env_var = env_var[i], 
           psi = b0[k] + b1[k]*env_var[i], 
           Z = rbinom(n(), 1, inv.logit(psi)), 
           theta = d0[k] + d1[k]*time, 
           det_sim = rbinom(n(), Z, inv.logit(theta)))

det_sim_wide <- reshape2::dcast(df_sim, i + k ~ j, value.var="det_sim")
time_sim_wide <- reshape2::dcast(df_sim, i + k ~ j, value.var="time")

# run model 
stan_data <- list(n_tot = nrow(det_sim_wide), 
                  n_visit = 4, 
                  n_species = length(unique(det_sim_wide$k)), 
                  n_point = length(unique(det_sim_wide$i)), 
                  id_sp = det_sim_wide$k, 
                  id_pt = det_sim_wide$i, 
                  det = det_sim_wide[,c("1", "2", "3", "4")],
                  vis_cov = time_sim_wide[,c("1", "2", "3", "4")],
                  Q = rowSums(det_sim_wide[,c("1", "2", "3", "4")]), 
                  env_var = env_var, 
                  grainsize = 1)

## Run mod ----
mod <- cmdstan_model("occupancy_example_redsum.stan", 
                     cpp_options = list(stan_threads = T))

replicate(5, {
    samps_8 <- mod$sample(data = stan_data, 
                          chains = 1, 
                          threads_per_chain = 8, 
                          iter_warmup = 1000, 
                          iter_sampling = 1000)
    print(samps_8$time())
    
    samps_4 <- mod$sample(data = stan_data, 
                          chains = 1, 
                          threads_per_chain = 4, 
                          iter_warmup = 1000, 
                          iter_sampling = 1000)
    print(samps_4$time())
    
    samps_2 <- mod$sample(data = stan_data, 
                          chains = 1, 
                          threads_per_chain = 2, 
                          iter_warmup = 1000, 
                          iter_sampling = 1000)
    print(samps_2$time())
    
    samps_1 <- mod$sample(data = stan_data, 
                          chains = 1, 
                          threads_per_chain = 1, 
                          iter_warmup = 1000, 
                          iter_sampling = 1000)
    print(samps_1$time())
    
    data_frame(cpu = c(1, 2, 4, 8),
               time = c(samps_1$time()$chains$total,
                        samps_2$time()$chains$total,
                        samps_4$time()$chains$total,
                        samps_8$time()$chains$total))
}, simplify=F) %>%
    bind_rows(., .id="run")


