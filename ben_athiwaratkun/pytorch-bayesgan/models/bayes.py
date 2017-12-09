import torch
#from distributions import Normal
from torch.autograd import Variable

class NoiseLoss(torch.nn.Module):
  # need the scale for noise standard deviation
  # scale = noise  std
  def __init__(self, params, scale=None, observed=None):
    super(NoiseLoss, self).__init__()
    # initialize the distribution for each parameter
    #self.distributions = []
    self.noises = []
    for param in params:
      noise = 0*param.data.cuda() # will fill with normal at each forward
      self.noises.append(noise)
    if scale is not None:
      self.scale = scale
    else:
      self.scale = 1.
    self.observed = observed

  def forward(self, params, scale=None, observed=None):
    # scale should be sqrt(2*alpha/eta)
    # where eta is the learning rate and alpha is the strength of drag term
    if scale is None:
      scale = self.scale
    if observed is None:
      observed = self.observed

    assert scale is not None, "Please provide scale"
    noise_loss = 0.0
    for noise, var in zip(self.noises, params):
      # This is scale * z^T*v
      # The derivative wrt v will become scale*z
      _noise = noise.normal_(0,1)
      noise_loss += scale*torch.sum(Variable(_noise)*var)
    noise_loss /= observed
    return noise_loss

class PriorLoss(torch.nn.Module):
  # negative log Gaussian prior
  def __init__(self, prior_std=1., observed=None):
    super(PriorLoss, self).__init__()
    self.observed = observed
    self.prior_std = prior_std

  def forward(self, params, observed=None):
    if observed is None:
      observed = self.observed
    prior_loss = 0.0
    for var in params:
      prior_loss += torch.sum(var*var/(self.prior_std*self.prior_std))
    prior_loss /= observed
    return prior_loss