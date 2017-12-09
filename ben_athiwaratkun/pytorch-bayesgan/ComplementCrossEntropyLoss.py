# Loss for G in semi supervised setting
import torch
#import torch.functional as F

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

class ComplementCrossEntropyLoss(torch.nn.Module):
  # Note: This is the cross entropy of the sum of all probabilities of other indices, except for the 
  # This is used in Bayesian GAN semi-supervised learning
  def __init__(self, except_index=None, weight=None, ignore_index=-100, size_average=True, reduce=True):
    super(ComplementCrossEntropyLoss, self).__init__()
    self.except_index = except_index
    self.weight = weight
    self.ignore_index = ignore_index
    self.size_average = size_average
    self.reduce = reduce

  def forward(self, input, target=None):
    # Use target if not None, else use self.except_index
    if target is not None:
      _assert_no_grad(target)
    else:
      assert self.except_index is not None
      target = torch.autograd.Variable(torch.LongTensor(input.data.shape[0]).fill_(self.except_index).cuda())
    result = torch.nn.functional.nll_loss(
      torch.log(1. - torch.nn.functional.softmax(input) + 1e-4), 
      target, weight=self.weight, 
      size_average=self.size_average, 
      ignore_index=self.ignore_index)
    return result