Reduce\_sum applied to 2D data
================
22 June 2020

In the 2.23 update, Stan introduced `reduce_sum`, which allows us to
parallelise within-chain computation. The 1D case- where the response
variable is a vector- has a well worked example
([here](https://mc-stan.org/users/documentation/case-studies/reduce_sum_tutorial.html)),
and the speedups for these simple models where most of the computation
can be placed inside the parallel sum component are reasonably well
documented. It’s taken a bit of experimentation to figure out how to
apply `reduce_sum` to a 2D case and I’m not aware of equivalent worked
examples for these cases, or example speedups. This document therefore
aims to provide a minimal example of a 2D application of `reduce_sum`
and take a look at some of the speedups.

There is:

1.  A brief description of the model and the data structure
2.  Fitting this model using `reduce_sum`
3.  A comparison of speedups with increasing number of threads

They are referred to at various points throughout, but see also:

  - The `reduce_sum` documentation:
    <https://mc-stan.org/docs/2_23/stan-users-guide/reduce-sum.html>

  - [The 1D worked example (logistic
    regression)](https://mc-stan.org/users/documentation/case-studies/reduce_sum_tutorial.html),
    which has details that aren’t included here

  - <https://jsocolar.github.io/occupancyModels/> as an
    introduction/explanation to the model in Stan

# (Brief) model description

A model familiar to many ecologists is the occupancy model. They come in
a lot of flavours, but, very broadly, these models are typically
attempting to relate binary presence/absence to characteristics of the
environment and/or species, usually using data collected from a number
of different spatial locations.

Of these models a further broad subset are detection-occupancy models,
which attempt to delinate observed occupancy into separate detection and
occupancy components (i.e. acknowledge that the probability of observing
a given species depends both on the underlying probability of presence
and also how likely it is to be observed, given presence). In order to
do this, two things are required: first there needs to be some form of
replicated sampling, and second we need to be able to reasonably assume
that if a species is observed on a single one of these replicates, it is
‘present’ (in some sense) on every other of the replicates. The
resulting mixture of 0s and 1s then allow for us to say something about
both detection and occupancy.

We’ll take a look at these detection-occupancy models, partly because it
is working with these models that have motivated the need to
parallelise, due to the fact that these datasets can quickly become
large and they aren’t obviously reduced to a 1D observation vector for
modelling, but also they are also somewhat distinct from different
flavours of glm that are already documented.

These models are readily fit in Stan (note though that they are slightly
differently formulated than in JAGS/BUGS, see
[here](https://jsocolar.github.io/occupancyModels/)).

## Data simulation

For our simulated species, there is variation in occupancy both across
species (i.e. some species are more common than others), and also across
an environmental gradient. Across all species, there is an average
tendency for occupancy probability to increase across the environmental
gradient, but there is variation in the strength of this association
across species, such that some species can decrease in occupancy
probability along this gradient. Detection varies across species, with
some species more readily detected than others, and also with a
visit-level covariate, which, for the sake of argument, we’ll call time
of day. Again, there is an average (negative) time of day effect across
species, but species also vary in the strength of this association.

More formally:

  
![\\psi\_{ik} = \\beta\_{0k} +
\\beta\_{1k}.x\_i](https://latex.codecogs.com/png.latex?%5Cpsi_%7Bik%7D%20%3D%20%5Cbeta_%7B0k%7D%20%2B%20%5Cbeta_%7B1k%7D.x_i
"\\psi_{ik} = \\beta_{0k} + \\beta_{1k}.x_i")  

  
![\\theta\_{ijk} = \\alpha\_{0k} +
\\alpha\_{1k}.t\_{ij}](https://latex.codecogs.com/png.latex?%5Ctheta_%7Bijk%7D%20%3D%20%5Calpha_%7B0k%7D%20%2B%20%5Calpha_%7B1k%7D.t_%7Bij%7D
"\\theta_{ijk} = \\alpha_{0k} + \\alpha_{1k}.t_{ij}")  

  
![Z\_{ik} \\sim
bernoulli(logit^{-1}(\\psi\_{ik}))](https://latex.codecogs.com/png.latex?Z_%7Bik%7D%20%5Csim%20bernoulli%28logit%5E%7B-1%7D%28%5Cpsi_%7Bik%7D%29%29
"Z_{ik} \\sim bernoulli(logit^{-1}(\\psi_{ik}))")  

  
![y\_{ijk} \\sim bernoulli(logit^{-1}(\\theta\_{ijk}) \\times
Z)](https://latex.codecogs.com/png.latex?y_%7Bijk%7D%20%5Csim%20bernoulli%28logit%5E%7B-1%7D%28%5Ctheta_%7Bijk%7D%29%20%5Ctimes%20Z%29
"y_{ijk} \\sim bernoulli(logit^{-1}(\\theta_{ijk}) \\times Z)")  

where
![\\psi\_{ik}](https://latex.codecogs.com/png.latex?%5Cpsi_%7Bik%7D
"\\psi_{ik}") is the occupancy probability, which varies according to a
species-level intercept, and a species-level environmental association
(where ![x\_i](https://latex.codecogs.com/png.latex?x_i "x_i") is the
environmental variable).
![\\theta\_{ijk}](https://latex.codecogs.com/png.latex?%5Ctheta_%7Bijk%7D
"\\theta_{ijk}") is the detection component, and varies according to a
species-level intercept, and a species-level time of day effect. All
species-level effects are simulated and modelled as normally-distributed
random effects, i.e. eight hyperparameters in total.

Data are generated from this model in the following script:

``` r
source("simulate_data.R")
```

For the timings, I’ll generate 50 species across 50 sites and 4 visits
(i.e. 10,000 observations in total).

Visually, the model look something like this (for nine example
species):

<div class="figure" style="text-align: center">

<img src="README_files/figure-gfm/figs-1.png" alt="**Figure 1:** Nine example species taken from the full simulation. Solid line indicates the underlying occupancy relationship, while the dashed line indicates the probability of detection, accounting for the species' detectability. As this depends both on the species' detectibility and also a visit-level covariate (time of day), the detectibility is not a constant offset, but rather varies depending on time of day (which varies between 0 and 1). Note the variation in the environmental association, the detection offset, and the strength of the time-of-day effect on detection accross species"  />

<p class="caption">

**Figure 1:** Nine example species taken from the full simulation. Solid
line indicates the underlying occupancy relationship, while the dashed
line indicates the probability of detection, accounting for the species’
detectability. As this depends both on the species’ detectibility and
also a visit-level covariate (time of day), the detectibility is not a
constant offset, but rather varies depending on time of day (which
varies between 0 and 1). Note the variation in the environmental
association, the detection offset, and the strength of the time-of-day
effect on detection accross species

</p>

</div>

## Data structure

Observations inherently have a 3D structure, structured by the point
that was surveyed (point ![i](https://latex.codecogs.com/png.latex?i
"i")), the visit to the point (visit
![j](https://latex.codecogs.com/png.latex?j "j")), and the species that
was observed/unobserved (species
![k](https://latex.codecogs.com/png.latex?k "k")). However, these are
readily collapsed to a 2D structure by collapsing the species dimension
to produce a dataframe that has dimensions (![n\_{points} \\times
n\_{species}](https://latex.codecogs.com/png.latex?n_%7Bpoints%7D%20%5Ctimes%20n_%7Bspecies%7D
"n_{points} \\times n_{species}"),
![n\_{visits}](https://latex.codecogs.com/png.latex?n_%7Bvisits%7D
"n_{visits}")), with each row now indexing a point:species combination.

``` r
head(det_sim_wide)
```

``` 
  i k 1 2 3 4
1 1 1 0 1 0 0
2 1 2 0 0 0 0
3 1 3 0 0 1 0
4 1 4 0 0 0 0
5 1 5 0 0 1 1
6 1 6 0 1 0 0
```

We have two columns that identify the species:point combination,
followed by 4 columns that give the observation (0 or 1) across each of
4 visits to the point.

The time of day predictor has the same dimensions. It could be reduced
to a smaller \[i, j\] array, as time of day only depends upon when the
point was visited, not upon which species were observed, but it’s more
straightforward to just produce it as a array of times with the same
dimensions as the detection data.

``` r
head(time_sim_wide)
```

``` 
  i k         1         2         3          4
1 1 1 0.6718841 0.5165038 0.8644191 0.09028463
2 1 2 0.2200007 0.9218696 0.6630422 0.28670152
3 1 3 0.1859771 0.8414575 0.4018805 0.14375275
4 1 4 0.8825094 0.2849294 0.4753757 0.35831950
5 1 5 0.6178907 0.8492360 0.6639175 0.89314053
6 1 6 0.8250160 0.3381398 0.9950988 0.11576463
```

``` r
head(env_var)
```

    [1] -0.5701107 -0.8914825  0.3680577  2.0535598 -0.3794084  0.2064427

Finally, an environmental variable, that just varies between points.

# The Stan model

The Stan file is a little complicated if it’s not a model you are
already familiar with. The main oddity is the if-statement asking if
Q==1: this just refers to the central assumption mentioned at the
outset, where points that have at least one observation are treated as
though the species is present (but undetected) at all other visits. The
converse, where a species is never observed at a point is more
complicated, because now we have to acknowledge that it might have been
there all along, but never observed, and also that it might simply not
be there. Again <https://jsocolar.github.io/occupancyModels/> explains
this in more detail.

    data {
        int<lower=1> n_tot;
        int<lower=1> n_visit;
        int<lower=1> n_species;
        int<lower=1> n_point;
        
        int<lower=1> id_sp[n_tot];
        int<lower=1> id_pt[n_tot];
        int det[n_tot, n_visit];
        row_vector[n_visit] vis_cov[n_tot];
        vector[n_tot] Q;
        vector[n_point] env_var;
        
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
        
        // Priors
        //...
    }

# `reduce_sum` formulation

To get the parallelisation done, we need to break up the computation
such that it can be passed out to multiple workers. To do this, the
model itself wrapped in `partial_sum` (user-written function), which
will allow the full task of calculating all the log-probabilities to be
broken up into chunks (the size of which are set by the grainsize
argument). `reduce_sum` will automate the choice about how to chunk up
the data, and pass these data chunks out to be computed in parallel.

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

This is largely a repetition of the previous stan model, the only
difference is that the model has been shifted into the `parallel_sum`
function block. Note that a lot of stuff has to get passed to this
function: you could simplify it by doing more computation outside
(e.g. transforming some parameters) and passing a reduced set of
objects, but then you would be shifting more of the computation
*outside* of the bit that will be run in parallel. As the
parallelisation speedup would take a hit from doing this, it’s better to
do it this way, despite the unwieldiness of the ensuing function.

When compiling the model, we need to specify that it is threaded
(cpp\_options), and then specify the threads\_per\_chain at the sampling
step. The model is run with `cmdstanr` (at some point I presume
multi-threading will be rolled out to `rstan` as well).

``` r
library(cmdstanr)
# data
stan_data <- list(n_tot = nrow(det_sim_wide), 
                  n_visit = 4, 
                  n_species = length(unique(det_sim_wide$k)), 
                  n_point = length(unique(det_sim_wide$i)), 
                  id_sp = det_sim_wide$k, 
                  id_pt = det_sim_wide$i, 
                  det = det_sim_wide[,3:6],
                  vis_cov = time_sim_wide[,3:6],
                  Q = rowSums(det_sim_wide[,3:6]), 
                  env_var = env_var, 
                  grainsize = 1)


## Run mod ----
library(cmdstanr)
mod <- cmdstan_model("occupancy_example_redsum.stan", 
                     cpp_options = list(stan_threads = T))

samps <- mod$sample(data = stan_data, 
                    chains = 1, 
                    threads_per_chain = 8, 
                    iter_warmup = 1000, 
                    iter_sampling = 1000)
print(samps$time())
```

# Timings

<div class="figure" style="text-align: center">

<img src="README_files/figure-gfm/fig2-1.png" alt="**Figure 2:** Timings across 1, 2, 4, and 8 cores, using a simulated dataset with 50 species, 50 sites, and 4 visits (i.e. 10,000 observations in total). "  />

<p class="caption">

**Figure 2:** Timings across 1, 2, 4, and 8 cores, using a simulated
dataset with 50 species, 50 sites, and 4 visits (i.e. 10,000
observations in total).

</p>

</div>

Timings:

``` 
  cpu   time speedup
1   1 1860.0    1.00
2   2  801.7    2.32
3   4  660.1    2.82
4   8  337.7    5.51
```

As you would hope, we are getting good speedups by parallelising the
model. While the dataset simulated here is fairly small, with a larger
dataset, a 4-6x speedup on a chain that would otherwise take \>1 week to
run is not trivial at all\! The speedup itself will also depend on the
proportion of the model that can be placed within the `partial_sum`
function (see
[here](https://statmodeling.stat.columbia.edu/2020/05/05/easy-within-chain-parallelisation-in-stan/)).
With increased model complexity, as long as the relative proportions of
‘stuff in the `partial_sum`’ to ‘stuff outside the `partial_sum`’
remain constant, we should expect to see similar speedups. Anecdotally
this seems to be the case, and I’ve seen fairly equivalent speedups
running these models with more complexity (e.g. more terms in the
detection and occupancy components).

For models where almost all the computation can be placed inside the
`partial_sum`, we can achieve a 1:1 speedup (in some cases, marginally
better\!). Given that this model has a bunch of centering of
hyper-parameters outside of the main model block, it is to be expected
that we should see speedups slightly below this line.

The other thing to mention is that when comparing speedups, it’s
important to give the computer a proper job to really see the speedup
from parallelisation. If it only takes 30 seconds to run the model, then
the speedups are all over the place (though the tendency to get faster
with increasing cores is present). Initially I had been simulating 15
species at 15 sites, across 4 visits (i.e. 900 observations)- speedups
were (a) not very repeatable, varying substantially between runs, and
(b) the scaling didn’t look great. Presumably this largely arises
because the overhead paid for parallelising makes up a greater
proportion of the total run time, and it’s difficult to get large
speed-gains on the bit that is being run in parallel.
