import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from representation import RepresentationNetwork
from generator import GeneratorNetwork

class GenerativeQueryNetwork(nn.Module):
    
    def __init__(self, x_dim, v_dim, r_dim, h_dim, z_dim, L=12):
        """
        x_dim -> Number of channels in input image
        v_dim -> Dimensions of viewpoint
        r_dim -> Dimensions of representation
        h_dim -> Dimensions of hidden
        z_dim -> Channels in latent variable
        L -> Number of refinements in the density
        """
        super(GenerativeQueryNetwork, self).__init__()
        self.r_dim = r_dim

        self.generator = GeneratorNetwork(x_dim, v_dim, r_dim, z_dim, h_dim, L)
        self.representation = RepresentationNetwork(x_dim, v_dim, r_dim)

    def forward(self, images, viewpoints):
        """
        images -> input images [b, m, c, h, w]
        viewpoints -> tensor containing viewpoint [b, m, k]
        m is the number of observations in the given scene
        """
        batch_size, m, *_ = viewpoints.size()

        # Sample random number of views for a scene
        n_views = random.randint(2, m-1)

        indices = torch.randperm(m)
        # Split the images randomly into representation and query network.
        # As we feed the images with repr idx to the representation network and 
        # the images with query_idx to the query(generator) network.
        representation_idx, query_idx = indices[:n_views], indices[n_views:]

        x, v = images[:, representation_idx], viewpoints[:, representation_idx]

        # Merge b and m dimensions
        x, v = self._compress(x), self._compress(v)

        # Fetch the representation of all images:
        _repr = self.representation(x, v)

        # Expand b and m dimensions again(for summing against each scene)
        _repr = self._expand(_repr, batch_size, n_views)

        # Summing representation across the number of observations dimensions
        # To get the final representations
        r = torch.sum(_repr, dim=1)

        # Pass the randomly selected images and viewpoints into the generative network
        x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]
        x_mu, kl = self.generator(x_q, v_q)

        return [x_mu, x_q, r, kl]

    def sample(self, context_images, context_view, viewpoint, sigma):
        """
        Sample when context image and viewpoints are given
        """
        batch_size, n_views, _, h, w = context_images.size()

        context_images, context_view = self._compress(context_images), self._compress(context_view)

        _repr = self.representation(context_images, context_view)

        _repr = self._expand(_repr, batch_size, n_views)

        r = torch.sum(_repr, dim=1)

        x_mu = self.generator.sample((h, w), viewpoint, r)
        x_sample = Normal(x_mu, sigma).sample()

        return x_sample

    def _compress(self, x):
        _, _, *x_dims = x.size()
        return x.view((-1, *x_dims))

    def _expand(self, x, bs, n_views):
        _, *x_dims = x.size()
        return x.view((bs, n_views, *x_dims))
