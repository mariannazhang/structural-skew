import streamlit as st

# import packages for model
from gorgo import infer, condition, draw_from, keep_deterministic
from gorgo.distributions.builtin_dists import Gaussian, Beta, Uniform, Gamma, Categorical, Binomial

import numpy as np; np.random.seed(42)
import pandas as pd

# import packages for plotting
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

# page title
st.title("Threshold adjustment model")

# set up sidebar
with st.sidebar:
    support_step = st.selectbox('''**Discretization step size**  
                                (smaller step = finer-grained predictions, slower runtime)''', options = (0.25, 0.5, 1), index = 2)
    
    st.write("# Model posterior on Zarpie height (eq. 1)")
    st.latex("P(\mu|z, t, \epsilon) \propto P(z|t, \epsilon, \mu) P(\mu)")
    
    st.write("# Likelihood of observing sample (eq. 2)")
    st.latex("P(z_i|t, \epsilon, \mu) \propto \mathcal{N}(z_i|\mu, \sigma) f(z_i|t, \epsilon)")
    
    
    st.write("# Study 1a parameters")
    
    with st.expander("**Filter parameters**", expanded = True):
        boat_height_range = st.slider("Filter threshold ($t$) (boat height)", 0, 15, value=[5, 10]) 
        
    st.write("# Additional parameters")
    
    with st.expander("**Sample ($z$) parameters**"):
        zarpie_height_1 = st.slider("Zarpie height 1 ($z_1$)", 1, 15, 4)
        zarpie_height_2 = st.slider("Zarpie height 2 ($z_2$)", 1, 15, 5)
        zarpie_height_3 = st.slider("Zarpie height 3 ($z_3$)", 1, 15, 6)
        zarpie_height_4 = st.slider("Zarpie height 4 ($z_4$)", 1, 15, 6)
        zarpie_height_5 = st.slider("Zarpie height 5 ($z_5$)", 1, 15, 7)
        zarpie_height_6 = st.slider("Zarpie height 6 ($z_6$)", 1, 15, 8)

    with st.expander("**Prior ($P(\mu)$) parameters**"):
        pop_mean_prior_mean = st.slider("Prior on population mean ($P(\mu)$), mean", 1, 15, 10)
        pop_mean_prior_sd = st.slider("Prior on population mean ($P(\mu)$), SD", 1, 5, 5)
        pop_sd = st.slider("Prior on population SD", 1, 5, 1)
    
    st.write("# Filter (eq. 3)")
    
    st.latex("f(z_i|t, \epsilon)=\epsilon + \\frac{1-\epsilon}{1+e^{-k(t-z_i)}}")
    
    with st.expander("**Filter parameters**"):
        pass_prob = st.slider('''Filter strength ($\epsilon$)  
                              (pass-through rate when above threshold)''', 0.0, 1.0, 0.1)
        k = st.slider('''Filter steepness ($k$)  
                      (lower $k$ = flatter, higher $k$ = steeper, step-like)''', 0.0, 100.0, 100.0, 0.1)



# set up model

@keep_deterministic
def discretized_gaussian(mean, sd, support=None):
    if support is None:
        support = list(np.arange(-20+mean, 20+mean, step=support_step)) # step needs to be 1, .5, or .25
    return Categorical.from_continuous(
        Gaussian(mean, sd),
        support=support # restrict range to make inference easier
    )

# logistic filter
# k is 0 to infinity, as k approaches inf, filter becomes a step-function
def logistic_filter(zarpie_height, boat_height, pass_prob, k):
    boarding_prob = np.float64(pass_prob + (1 - pass_prob)/(1 + np.e**(-k*(boat_height - zarpie_height))))
    return boarding_prob

# boat boarding model
# requires: boat height, population mean, pass-through probability
@infer
def boat_boarding_model(boat_height, pop_mean, pop_sd, pass_prob, k):
    zarpie_height = discretized_gaussian(pop_mean, pop_sd).sample() # randomly sample Zarpies from prior (Gaussian with mean _, SD 2)
    # step-function filter
    # condition(1 if zarpie_height < boat_height # if Zarpie shorter than boat height, board
    #           else pass_prob) # if Zarpie taller than boat height, board based on pass-through probability
    boarding_prob = logistic_filter(zarpie_height, boat_height, pass_prob, k) # logistic filter
    condition(boarding_prob)
    return zarpie_height

# infer population mean
# requires: boat height, observed Zarpie height
# assume: prior as Gaussian with mean drawn from (10, 5), pass-through probability .1
@infer
def threshold_adjustment_model(boat_height, zarpie_heights, pop_mean_prior_mean = 10, pop_mean_prior_sd = 5, pop_sd = 2, pass_prob = .1, k = 1):
    pop_mean = discretized_gaussian(pop_mean_prior_mean, pop_mean_prior_sd).sample()  # sample a population mean from prior
    # [
    #     condition(boat_boarding_model(boat_height, pop_mean) == zh)
    #     for zh in zarpie_heights
    # ]
    dist = boat_boarding_model(boat_height, pop_mean, pop_sd, pass_prob, k)  # run boat boarding model with given boat height, population mean, etc.
    [dist.observe(zarpie_height) for zarpie_height in zarpie_heights]
    return pop_mean 

# initalize list
threshold_adjustment_res_list = []

# run the model
for boat_height in np.arange(boat_height_range[0], boat_height_range[1], 1):
    threshold_adjustment_res = threshold_adjustment_model(
        zarpie_heights = (
                        zarpie_height_1, 
                        zarpie_height_2, 
                        zarpie_height_3, 
                        zarpie_height_4,
                        zarpie_height_5,
                        zarpie_height_6
                        ),
        boat_height = boat_height, 
        pop_mean_prior_mean = pop_mean_prior_mean, 
        pop_mean_prior_sd = pop_mean_prior_sd,
        pop_sd = pop_sd,
        pass_prob = pass_prob,
        k = k
        )

    # record inferred probability on each height value
    for val, prob in threshold_adjustment_res.items():
        threshold_adjustment_res_list.append(dict(
            boat_height = boat_height,
            value = val,
            prob = prob
        ))

# convert list to df   
threshold_adjustment_res_df = pd.DataFrame(threshold_adjustment_res_list)

# make plot
g1 = sns.relplot(data = threshold_adjustment_res_df,
                x = "value",
                y = "prob",
                hue = "boat_height",
                kind = "line") 
plt.xlim(0, 15)
g1.set_ylabels("posterior probability")
g1.set_xlabels("Zarpie height")
st.pyplot(g1)







# filter plot
boarding_prob_list = []

# look at the stated range of boat heights, look at all possible zarpie heights
for boat_height in np.arange(boat_height_range[0], boat_height_range[1], 1):
    for zarpie_height in np.arange(1, 15):
        # run the logistic filter for boat height = 6
        boarding_prob = logistic_filter(zarpie_height = zarpie_height, boat_height = boat_height, pass_prob = pass_prob, k = k)
        boarding_prob_list.append(dict(
            zarpie_height = zarpie_height,
            boat_height = boat_height,
            pass_prob = pass_prob,
            k = k,
            boarding_prob = boarding_prob
        ))
            
# convert to df
boarding_prob_df = pd.DataFrame(boarding_prob_list)

g2 = sns.relplot(data = boarding_prob_df,
                 x = "zarpie_height",
                 y = "boarding_prob",
                 hue = "boat_height",
                 kind = "line")
g2.set_ylabels("boarding probability")
plt.ylim(0, 1)
g2.set_xlabels("Zarpie height")
plt.xlim(0, 15)
# g.figure.suptitle(f"Logistic filter (pass-through prob: {pass_prob}, k: {k})")
plt.title(f"Logistic filter \n(pass-through prob: {pass_prob}, k: {k})")
# g.figure.subplots_adjust(top = 0.9)
st.pyplot(g2)

# add simple model plot
