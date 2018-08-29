"""
The inference generator model is similar to variational autoencoders.
The latents must be inferred from the ConvLSTM model. Latents are the
hidden variables which ultimately tries to capture the distribution of
input. The way we learn the distribution is simple. KL divergence between
the prior and posterior must be minimized, hence forms the part of our criterion.
The latent variable z is sampling from learned posterior distribution. As we use
a sequential model for generation, prior can be calculated by auto-regressive density.
"""

SCALE = 4

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


"""
We use a ConvLSTM model for generation. It is similar to the
LSTM model but linear layers are replaced by Convolutional
layers.
"""
class Conv2dLSTMCell(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernal_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()
        self.in_channels = input_channels
        self.out_channels = output_channels

        kwargs = dict(kernal_size=kernal_size, stride=stride, padding=padding)

        self.forget = nn.Conv2d(self.in_channels, self.out_channels **kwargs)
        self.input = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)
        self.output = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)
        self.state = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)

    def forward(self, input, states):
        """
        input -> input to be passed
        states -> (hidden, cell) 
        returns next hidden, cell pair
        """

        (hidden, cell) = states

        forget_gate = F.sigmoid(self.forget(input))
        input_gate = F.sigmoid(self.input(input))
        output_gate = F.sigmoid(self.output(input))
        state_gate = F.sigmoid(self.state(input))

        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * F.tanh(cell)

        return hidden, cell


"""
The generator network is LSTM like network
with Conv layers and latent variables to learn
the distributions
"""
class GeneratorNetwork(nn.Module):
    """
    x_dim -> number of input channels
    v_dim -> dimensions of viewpoint vector
    r_dim -> dimension of representation
    z_dim -> dimension of latent variable
    h_dim -> dimension of hidden dimensions
    L -> Number of layers in which latent variables would be sequentially refined
    """
    
    def __init__(self, x_dim, v_dim, r_dim, z_dim=64, h_dim=128, L=12):
        super(GeneratorNetwork, self).__init__()
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Core layers consists of inference and generator layers:
        # Inference layers gives posterior distribution from which we sample latent variable
        # Generator layer gives prior distribution and generates the prediction given input and latent variable
        # Gist is inference and generator behave like variational autoencoders

        self.inference_core = Conv2dLSTMCell(h_dim + x_dim + v_dim + r_dim, h_dim, kernal_size=5, stride=1, padding=2)
        self.generator_core = Conv2dLSTMCell(v_dim + r_dim + z_dim, h_dim, kernal_size=5, stride=1, padding=2)

        # To obtain posterior and prior we use another Convolutional layers for each
        # Output is 2 x no. of dimensions of latent variable to accomodate mean and std. deviation of
        # the distributions. We just split the output tensor in half to get mean and std. dev which can
        # be used later for sampling
        self.posterior_density = nn.Conv2d(h_dim, 2 * z_dim, kernel_size=5, stride=1, padding=2)
        self.prior_density = nn.Conv2d(h_dim, 2 * z_dim, kernel_size=5, stride=1, padding=2)

        # Generative density
        self.observation_density = nn.Conv2d(h_dim, x_dim, kernel_size=1, stride=1, padding=0)

        # Upsampling/Downsampling primitives
        self.upsample = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=SCALE, stride=SCALE, padding=0)
        self.downsample = nn.Conv2d(x_dim, x_dim, kernel_size=SCALE, stride=SCALE, padding=0)

    def forward(self, x, v, r):
        """
        Attempt to reconstruct an image for given arbritary viewpoint given 
        x -> input image
        v -> viewpoint
        r -> representation of the scene for which we want to generate new images
        """
        batch_size, _, h, w = x.size()
        kl_div = 0

        # Increase dimensions:
        v = v.view(batch_size, -1, 1, 1).repeat(1, 1, h//SCALE, w//SCALE)
        if r.size(2) != h//SCALE:
            r = r.repeat(1, 1, h//SCALE, w//SCALE)

        # Reset hidden and cell state
        hidden_g, hidden_i = self._init(x, batch_size, h, w), self._init(x, batch_size, h, w)
        cell_g, cell_i = self._init(x, batch_size, h, w), self._init(x, batch_size, h, w)

        u = self._init(x, batch_size, h, w)

        # Downsample x by applying conv
        x = self.downsample(x)

        # LSTM Loop!
        for _ in range(self.L):

            # Get Prior distribution            
            prior_distribution = self._get_distribution(hidden_g)

            # Inference state update:
            hidden_i, cell_i = self.inference_core(torch.cat([hidden_g, x, v, r], dim=1), [hidden_i, cell_i])

            # Get Posterior distribution
            posterior_distribution = self._get_distribution(hidden_i, type='posterior')

            # Sample posterior to get latent variable
            z = posterior_distribution.rsample()

            # Generator
            hidden_g, cell_g = self.generator_core(torch.cat([z, v, r], dim=1), [hidden_g, cell_g])

            u = self.upsample(hidden_g) + u

            # Calculating the KL divergence
            kl_div += kl_divergence(posterior_distribution, prior_distribution)

        x_mu = self.observation_density(u)
        return F.sigmoid(x_mu), kl_div


    def sample(self, x_shape, v, r):
        """
        Sample from the prior distribution to generate new image,
        given any arbritary viewpoint and scene representation.
        x_shape -> (h, w) of the image to be generated
        v -> viewpoint
        r -> representation
        """
        h, w = x_shape
        batch_size = v.size(0)

        # Increase the dimensions of the viewpoint and representation
        v = v.view(batch_size, -1, 1, 1).repeat(1, 1, h//SCALE, w//SCALE)
        if r.size(2) != h//SCALE:
            r = r.view(batch_size, -1, 1, 1).repeat(1, 1, h//SCALE, w//SCALE)

        hidden_g, cell_g = self._init(v, batch_size, h, w), self._init(v, batch_size, h, w)

        u = v.new_zeros((batch_size, self.h_dim, h, w))

        for _ in range(self.L):
            
            # Calculate the prior distribution
            prior_distribution = self._get_distribution(hidden_g)

            # Sample from the prior distribution
            z = prior_distribution.sample()

            # Calculate u:
            hidden_g, cell_g = self.generator_core(torch.cat([z, v, r], dim=1), [hidden_g, cell_g])
            u = self.upsample(hidden_g) + u

        x_mu = self.observation_density(u)
        return F.sigmoid(x_mu)

    def _init(self, x, bs, h, w):
        return x.new_zeros((bs, self.h_dim, h//SCALE, w//SCALE))

    def _get_distribution(self, inputs, type='prior'):
        if type == 'prior':
            o = self.prior_density(inputs)
        elif type == 'posterior':
            o = self.posterior_density(inputs)
        else:
            raise ValueError('Invalid distribution name.')
        p_mu, p_std = torch.split(o, self.z_dim, dim=1)
        return Normal(p_mu, F.softplus(p_std))
