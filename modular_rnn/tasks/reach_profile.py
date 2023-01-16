from jax import jit
from tensorflow_probability.substrates import jax as tfp

# median_reaction_time = 23
# speed_curve_mean = 29.
# speed_curve_std = 12.
    
speed_gaussian = tfp.distributions.Normal(loc = 29., scale = 12.)

@jit
def speed_curve(x):
    return speed_gaussian.prob(x - 23)

# actual distance because cdf is the integral of the pdf.
@jit
def extent_curve(x):
    return speed_gaussian.cdf(x - 23)
