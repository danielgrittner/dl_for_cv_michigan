from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device='cpu'):
  """
  Generate a PyTorch Tensor of uniform random noise.

  Input:
  - batch_size: Integer giving the batch size of noise to generate.
  - noise_dim: Integer giving the dimension of noise to generate.
  
  Output:
  - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
    random noise in the range (-1, 1).
  """
  noise = None
  ##############################################################################
  # TODO: Implement sample_noise.                                              #
  ##############################################################################
  
  noise = 2 * (torch.rand((batch_size, noise_dim), dtype=dtype, device=device) - 0.5)

  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################

  return noise


def discriminator():
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement discriminator.                                           #
  ############################################################################
  
  input_dim = 784
  hidden_dim = 256
  
  model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.LeakyReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LeakyReLU(),
    nn.Linear(hidden_dim, 1)
  )
  
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################
  return model


def generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement generator.                                               #
  ############################################################################
  
  hidden_dim = 1024
  out_dim = 784

  model = nn.Sequential(
    nn.Linear(noise_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, out_dim),
    nn.Tanh()
  )

  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################
  return model  


def discriminator_loss(logits_real, logits_fake):
  """
  Computes the discriminator loss described above.
  
  Inputs:
  - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
  - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Returns:
  - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
  """
  loss = None
  ##############################################################################
  # TODO: Implement discriminator_loss.                                        #
  ##############################################################################
  
  N = logits_real.size()[0]
  
  true_labels = torch.ones_like(logits_real)
  false_labels = torch.zeros_like(logits_fake)
  
  loss = nn.functional.binary_cross_entropy_with_logits(logits_real, true_labels, reduction='sum')
  loss += nn.functional.binary_cross_entropy_with_logits(logits_fake, false_labels, reduction='sum')
  loss /= N
  
  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return loss


def generator_loss(logits_fake):
  """
  Computes the generator loss described above.

  Inputs:
  - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Returns:
  - loss: PyTorch Tensor containing the (scalar) loss for the generator.
  """
  loss = None
  ##############################################################################
  # TODO: Implement generator_loss.                                            #
  ##############################################################################
  
  N = logits_fake.size()[0]
  true_labels = torch.ones_like(logits_fake)
  loss = nn.functional.binary_cross_entropy_with_logits(logits_fake, true_labels, reduction='sum')
  loss /= N
  
  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return loss


def get_optimizer(model):
  """
  Construct and return an Adam optimizer for the model with learning rate 1e-3,
  beta1=0.5, and beta2=0.999.
  
  Input:
  - model: A PyTorch model that we want to optimize.
  
  Returns:
  - An Adam optimizer for the model with the desired hyperparameters.
  """
  optimizer = None
  ##############################################################################
  # TODO: Implement optimizer.                                                 #
  ##############################################################################
  
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
  
  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
  """
  Compute the Least-Squares GAN loss for the discriminator.
  
  Inputs:
  - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
  - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Outputs:
  - loss: A PyTorch Tensor containing the loss.
  """
  loss = None
  ##############################################################################
  # TODO: Implement ls_discriminator_loss.                                     #
  ##############################################################################
  
  N = scores_real.size()[0]
  diff_real = scores_real - 1
  loss = 0.5 * (diff_real * diff_real).sum() / N
  loss += 0.5 * (scores_fake * scores_fake).sum() / N
  
  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return loss


def ls_generator_loss(scores_fake):
  """
  Computes the Least-Squares GAN loss for the generator.
  
  Inputs:
  - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Outputs:
  - loss: A PyTorch Tensor containing the loss.
  """
  loss = None
  ##############################################################################
  # TODO: Implement ls_generator_loss.                                         #
  ##############################################################################
  
  N = scores_fake.size()[0]
  diff = scores_fake - 1
  loss = 0.5 * (diff * diff).sum() / N
  
  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return loss


def build_dc_classifier():
  """
  Build and return a PyTorch nn.Sequential model for the DCGAN discriminator implementing
  the architecture in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement build_dc_classifier.                                     #
  ############################################################################
  
  model = nn.Sequential(
    nn.Unflatten(1, (1, 28, 28)),
    nn.Conv2d(1, 32, kernel_size=5, stride=1),
    # (N, 32, 24, 24)
    nn.LeakyReLU(),
    nn.MaxPool2d(2, stride=2),
    # (N, 32, 12, 12)
    nn.Conv2d(32, 64, kernel_size=5, stride=1),
    # (N, 64, 8, 8)
    nn.LeakyReLU(),
    nn.MaxPool2d(2, stride=2),
    # (N, 64, 4, 4)
    nn.Flatten(),
    # (N, 64*4*4)
    nn.Linear(64 * 4 * 4, 64 * 4 * 4),
    # (N, 64*4*4)
    nn.LeakyReLU(),
    nn.Linear(64 * 4 * 4, 1)
    # (N, 1)
  )
  
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################
  return model


def build_dc_generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the DCGAN generator using
  the architecture described in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement build_dc_generator.                                      #
  ############################################################################
  
  model = nn.Sequential(
    nn.Linear(noise_dim, 1024),
    # (N, 1024)
    nn.ReLU(),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, 128*7*7),
    # (N, 7*7*128)
    nn.ReLU(),
    nn.BatchNorm1d(128*7*7),
    nn.Unflatten(1, (128, 7, 7)),
    # (N, 128, 7, 7)
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding_mode='zeros', padding=1),
    # (N, 64, 14, 14)
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding_mode='zeros', padding=1),
    # (N, 1, 28, 28)
    nn.Tanh(),
    nn.Flatten()
    # (N, 1*28*28)
  )
  
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################
  return model
