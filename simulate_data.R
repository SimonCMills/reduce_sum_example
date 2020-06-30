## simulate some simple 2D occupancy data
library(dplyr); library(ggplot2); library(boot); library(reshape2)

# define dimensions
n_site <- 15
n_species <- 15
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
