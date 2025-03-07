import torch
from pyknos.nflows.transforms import PointwiseAffineTransform
from torch import nn
from typing import Optional
import warnings

from normflows.distributions.base import Uniform, UniformGaussian
from normflows.flows import (
    Permute,
    InvertibleAffine,
    LULinearPermute
)
# from sbi.neural_nets.flow import build_maf, build_nsf
# from sbmfi.inference.normflows_patch import (
#     CircularAutoregressiveRationalQuadraticSpline,
#     CircularCoupledRationalQuadraticSpline,
#     EmbeddingConditionalNormalizingFlow,
#     DiagGaussianScale
# )
# from normflows.flows.neural_spline.wrapper import (
#     CoupledRationalQuadraticSpline,
#     AutoregressiveRationalQuadraticSpline
# )
from normflows.flows import (
    CircularAutoregressiveRationalQuadraticSpline,
    AutoregressiveRationalQuadraticSpline,
    CircularCoupledRationalQuadraticSpline,
    CoupledRationalQuadraticSpline
)
from normflows.core import ConditionalNormalizingFlow
from normflows.distributions.base import DiagGaussian

from sbmfi.core.coordinater import FluxCoordinateMapper
from sbmfi.core.simulator import _BaseSimulator
from sbmfi.priors.uniform import _BasePrior
import numpy as np
import tqdm
from typing import Dict
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)
import inspect
import types


class DiagGaussianScale(DiagGaussian):
    def __init__(self, shape, trainable=True, scale=0.3):
        super().__init__(shape, trainable)
        log_scale = torch.log(torch.ones(1, *self.shape) * scale)
        if trainable:
            self.log_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer("log_scale", log_scale)


def _embedinator(fn):
    def inner(*args, **kwargs):
        self = args[0]
        context = kwargs.get('context', None)
        if context is None and len(args) > 2:
            context = args[-1]
            args = args[:-1]
        if (context is not None) and (self._embnet is not None):
            context = self._embnet(context)
            kwargs['context'] = context
        return fn(*args, **kwargs)
    return inner


def _class_embedinator(decorator):  # @_class_embedinator(_embedinator)
    # TODO make a class with embedinator
    def decorate(cls):
        for attr in dir(cls): # there's propably a better way to do this
            cls_attr = getattr(cls, attr)
            if isinstance(cls_attr, types.FunctionType):
                if 'context' in inspect.getfullargspec(cls_attr).args:
                    setattr(cls, attr, decorator(cls_attr))
        return cls
    return decorate


class Flow_Dataset(Dataset):
    def __init__(self, data: torch.Tensor, theta: torch.Tensor, standardize=True):
        if theta.shape[0] != data.shape[0]:
            raise ValueError

        if data.ndim == 3:
            n, n_obs, n_d = data.shape
            if theta.ndim == 2:
                theta = theta.tile(n_obs, 1, 1).transpose(0, 1)

        if (data.ndim == 3) and (theta.ndim == 3):
            data = data.view(n * n_obs, n_d)
            n_t = theta.shape[-1]
            theta = theta.contiguous().view(n * n_obs, n_t)

        self.data_mean = data.mean(0, keepdims=True)
        self.data_std = data.std(0, keepdims=True)

        if standardize:
            data = (data - self.data_mean) / self.data_std
            # data = (data * self.data_std) + self.data_mean # reverse

        self.data = data
        self.theta = theta

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.theta[idx]


def flow_constructor(
        fcm: FluxCoordinateMapper,
        coordinate_id,
        embedding_net=None,
        num_context_channels=None,
        autoregressive=True,
        num_blocks=4,
        num_hidden_channels=5,
        num_bins=10,
        dropout_probability=0.0,
        num_transforms = 3,
        init_identity=True,
        use_lu=True,
        mixing_id='lu',
        p=None,
        device='cpu',
):
    # prior_flow just makes a normalizing flow that matches samples from a prior
    #   thus not needing to fuck around with context = conditioning on data

    n_theta = len(fcm.theta_id(coordinate_id))

    if not torch.cuda.is_available():
        device = 'cpu'

    if coordinate_id not in ['cylinder', 'rounded']:
        raise ValueError(f'can only flow for cylinder and rounded')
    elif coordinate_id == 'rounded':
        ind = list(range(n_theta - fcm._nx, n_theta))
        scale = torch.ones(n_theta)
        scale[-fcm._nx:] *= 2  # need to pass the width!
        base = UniformGaussian(ndim=n_theta, ind=ind, scale=scale)
        base.scale.to(device)
        warnings.warn('Watch out! Base distribution for xch fluxes is U[-1,1]. '
                      'You need to manually scale the exchange fluxes, the currently have fcm._rho_bounds')
    elif coordinate_id == 'cylinder':
        base = Uniform(shape=n_theta, low=-1.0, high=1.0)
        base.low = base.low.to(device)
        base.high = base.high.to(device)

    transforms = []
    for i in range(num_transforms):
        common_kwargs = dict(
            num_input_channels=n_theta,
            num_blocks=num_blocks,
            num_hidden_channels=num_hidden_channels,
            num_context_channels=num_context_channels,
            num_bins=num_bins,
            tail_bound=1.0,
            activation=nn.ReLU,
            dropout_probability=dropout_probability,
            init_identity=init_identity,
        )
        if coordinate_id == 'cylinder':
            # this is the index of \theta, the coordinate that we model with a circular flow
            common_kwargs['ind_circ'] = [0]
        if (coordinate_id == 'cylinder') and autoregressive:
            transform = CircularAutoregressiveRationalQuadraticSpline(
                **common_kwargs,
                permute_mask=True,
            )
        elif (coordinate_id == 'cylinder') and not autoregressive:
            transform = CircularCoupledRationalQuadraticSpline(
                **common_kwargs,
                reverse_mask=False,
                mask=None,
            )
        elif (coordinate_id == 'rounded') and autoregressive:
            transform = AutoregressiveRationalQuadraticSpline(**common_kwargs)
        elif (coordinate_id == 'rounded') and not autoregressive:
            transform = CoupledRationalQuadraticSpline(**common_kwargs)

        if mixing_id == 'lu':
            mixer = LULinearPermute(num_channels=n_theta, identity_init=init_identity)
        elif mixing_id == 'shuffle':
            mixer = Permute(num_channels=n_theta, mode='shuffle')
        elif mixing_id == 'affine':
            mixer = InvertibleAffine(num_channels=n_theta, use_lu=use_lu)
        else:
            raise ValueError(f'not a valid mixing_id: {mixing_id}')

        transform_sequence = [transform]
        if mixing_id is not None:
            transform_sequence = [transform, mixer]

        transforms.extend(transform_sequence)

    if mixing_id is not None:
        transforms = transforms[:-1]

    flow = ConditionalNormalizingFlow(q0=base, flows=transforms, p=p)
    flow.to(device=device)
    return flow


def flow_trainer(
    flow,
    dataset,
    optimizer=torch.optim.Adam,
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=64,
    max_iter=100,
    show_progress=True,
):
    optim = optimizer(flow.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if max_iter is None:
        max_iter = len(train_loader)

    if show_progress:
        pbar = tqdm.tqdm(total=max_iter, ncols=100, desc='loss')
    losses = []
    try:
        for i, (x, y) in enumerate(train_loader):
            loss = flow.forward_kld(y, context=x)
            optim.zero_grad()
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optim.step()
            losses.append(loss.to('cpu').data.numpy())
            if show_progress:
                pbar.update(1)
                pbar.set_postfix(forward_kld=np.round(losses[-1], 5))
            if i == max_iter:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if show_progress:
            pbar.close()
    return np.array(losses)


if __name__ == "__main__":
    pass
