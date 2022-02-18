# Bayesian NNs using MCMC

## Methodology

### The Prior ~ `N(0, 1)`

According to [Section IV-C3 of Hands-on Bayesian Neural Networks](https://arxiv.org/pdf/2007.06823.pdf), the prior is analogous to a regularizer seen in traditional gradient descent based MLPs.\
I picked the prior as a simple Normal distribution with `mean: 0` and `stdev: 1`, so that the weights of the MLP are discouraged from having high magnitudes.

### The Likelihood ~ `N(label, 0.0001)`

The likelihood encourages the MLP to produce outputs close to the ground labels.\
Thus, I've picked likelihood to be a Normal distribution `centered about the labels` with `stdev: 0.0001`. A lower standard deviation implies a greater strictness of the likelihood function. I found `0.0001` to give good training performance.

### The MCMC Procedure - Metropolis-Hastings

I've used the Metropolis-Hastings algorithm for sampling the MLP weights. The transition matrix uses a Normal distribution `centered about each scaler` inside the weight matrix with `stdev: 1`.

The Metropolis-Hastings algorithm implemented has a `burn in accuracy` threshold before it starts recording the sampled weights. It also has a `patience` metric, which is the maximum number of attempts the chain will make to grow by a single step, before giving up.

## Dataset and Model

The dataset is a noisy `2D XOR` distribution.

The model is an MLP with: \
Inputs: `2`\
Hidden layers: `3x3`\
Outputs: `1`

## Results

### Training/Sampling Accuracy

<image src='readme-images/training graph.jpg' alt='training graph'/>\
Sampling was done with a burn in accuracy of `87.5%`, and a patience of `8000 attempts`.

### Test Accuracy with Metropolis-Hastings

<image src='./readme-images/test-results-mcmc.jpg' alt='mcmc test results'/>

Average standard deviation across all data points: `0.1069`\
Average accuracy: `92.85 %`\
Accuracy stdev: `3.4613 %`

### Comparison with Random Sampler

<image src='./readme-images/test-results-random.jpg' alt='random sampler test results'/>

Average standard deviation across all data points: `0.238`\
Average accuracy: `53.15 %`\
Accuracy stdev: `4.1008 %`
