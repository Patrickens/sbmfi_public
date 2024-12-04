import normflows
import torch
from normflows.utils.splines import rational_quadratic_spline
# from sbi.neural_nets.flow import build_maf, build_nsf
from functools import partial
from pyknos.nflows.transforms import PointwiseAffineTransform
from torch import Tensor, nn, relu, tanh, tensor, uint8
from typing import Optional
from sbi.utils.torchutils import create_alternating_binary_mask
from normflows.distributions.base import BaseDistribution, Uniform, UniformGaussian, DiagGaussian
from normflows.flows import Permute, LULinearPermute
from sbmfi.inference.normflows_patch import (
    CircularAutoregressiveRationalQuadraticSpline,
    CircularCoupledRationalQuadraticSpline,
    EmbeddingConditionalNormalizingFlow,
    DiagGaussianScale,
    Flow_Dataset
)
from torch.utils.data import Dataset, DataLoader, random_split
from normflows.flows.neural_spline.wrapper import (
    CoupledRationalQuadraticSpline,
    AutoregressiveRationalQuadraticSpline
)
from sbmfi.core.polytopia import FluxCoordinateMapper
from sbmfi.core.model import LabellingModel
from sbmfi.core.simulator import _BaseSimulator
from sbmfi.inference.bayesian import _BaseBayes
from sbmfi.inference.priors import _BasePrior
import math
import numpy as np
import tqdm
from typing import Dict, Union, Any
from ray import tune
import os
from sbmfi.settings import SIM_DIR
import shutil
from pathlib import Path
import ray
from ray import tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from ray.tune import Callback

def flow_constructor(
        fcm: FluxCoordinateMapper = None,
        circular=True,
        embedding_net=None,
        num_context_channels=None,
        autoregressive=True,
        num_blocks=4,
        num_hidden_channels=20,
        num_bins=10,
        dropout_probability=0.0,
        num_transforms = 3,
        init_identity=True,
        permute=None,
        p=None,
        scale=0.3,
):
    # prior_flow just makes a normalizing flow that matches samples from a prior
    #   thus not needing to fuck around with context = conditioning on data

    if fcm._bound is None:
        raise ValueError('needs to have a tail_bound!')

    if circular and not (fcm._sampler.basis_coordinates == 'cylinder'):
        raise ValueError('needs to have cylinder base_coordinates or logit ya schmuckington')
    elif not circular and not (fcm._sampler.basis_coordinates == 'rounded'):
        raise ValueError('we need to have roonded coordenates ya goon')

    n_theta = len(fcm.theta_id)

    if circular:
        if (fcm._nx > 0) and fcm.logit_xch_fluxes:
            ind = list(range(n_theta - fcm._nx))
            scale = torch.ones(n_theta)
            scale[:-fcm._nx] *= fcm._bound * 2  # need to pass the width!
            base = UniformGaussian(ndim=len(fcm.theta_id), ind=ind, scale=scale)
        else:
            base = Uniform(shape=len(fcm.theta_id), low=-fcm._bound, high=fcm._bound)
    else:
        if (fcm._nx > 0) and not fcm.logit_xch_fluxes:
            ind = list(range(n_theta - fcm._nx, n_theta))
            scale = torch.ones(n_theta)
            scale[-fcm._nx:] *= fcm._bound * 2  # need to pass the width!
            base = UniformGaussian(ndim=len(fcm.theta_id), ind=ind, scale=scale)
        else:
            base = DiagGaussianScale(n_theta, trainable=True, scale=scale)

    transforms = []
    for i in range(num_transforms):
        common_kwargs = dict(
            num_input_channels=n_theta,
            num_blocks=num_blocks,
            num_hidden_channels=num_hidden_channels,
            num_context_channels=num_context_channels,
            num_bins=num_bins,
            tail_bound=fcm._bound,
            activation=nn.ReLU,
            dropout_probability=dropout_probability,
            init_identity=init_identity,
        )
        if circular:
            common_kwargs['ind_circ'] = [0]
        if circular and autoregressive:
            transform = CircularAutoregressiveRationalQuadraticSpline(
                **common_kwargs,
                permute_mask=True,
            )
        elif circular and not autoregressive:
            transform = CircularCoupledRationalQuadraticSpline(
                **common_kwargs,
                reverse_mask=False,
                mask=None,
            )
        elif not circular and autoregressive:
            transform = AutoregressiveRationalQuadraticSpline(**common_kwargs)
        else:
            transform = CoupledRationalQuadraticSpline(**common_kwargs)

        if permute == 'lu':
            perm = LULinearPermute(num_channels=n_theta, identity_init=init_identity)
        elif permute == 'shuffle':
            perm = Permute(num_channels=n_theta, mode='shuffle')

        transform_sequence = [transform]
        if permute is not None:
            transform_sequence = [transform, perm]

        transforms.extend(transform_sequence)

    if permute is not None:
        transforms = transforms[:-1]

    flow = EmbeddingConditionalNormalizingFlow(q0=base, flows=transforms, embedding_net=embedding_net, p=p)
    return flow


class MFA_Flow(tune.Trainable):
    def _prior_flow_step(self):
        theta = self._prior.sample((self._batch_size,))
        loss = self._flow.forward_kld(theta)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            self._optimizer.step()
        return {'forward_kld': loss.to('cpu').data.numpy().item()}

    def _posterior_flow_step(self):
        raise NotImplementedError

    def _get_data_loaders(self):
        pass

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Dict]:
        pass

    def setup(self, config: Dict, prior=None, simulator=None):
        self._prior = prior
        self._simulator = simulator

        # TODO load data and use a batch to parametrize z-scoring
        embedding = config.get('data_embedding')
        embedding_net = None
        if embedding == 'z_score_trainable':
            embedding_net = PointwiseAffineTransform  # TODO shift and scale are registered as buffers????
            # register as parameters
            raise NotImplementedError
        # TODO come up with other embedding nets?

        prior_flow = config.get('prior_flow', True)

        self._flow = flow_constructor(
            fcm=prior._fcm,
            simulator=simulator,
            embedding_net=embedding_net,
            prior_flow=prior_flow,
            autoregressive=config.get('autoregressive', True),
            num_blocks=config.get('num_blocks', 2),
            num_hidden_channels=config.get('num_hidden_channels', 10),
            num_bins=config.get('num_bins', 10),
            dropout_probability=config.get('dropout_probability', 0.1),
            use_batch_norm=config.get('use_batch_norm', False),
            num_transforms=config.get('num_transforms', 2),
            init_identity=config.get('init_identity', True),
        )

        self._optimizer = torch.optim.Adam(
            self._flow.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        self._batch_size = config.get('batch_size', 512)

        if prior_flow:
            self.step = self._prior_flow_step
        else:
            self.step = self._posterior_flow_step


class CVStopper(TrialPlateauStopper):

    def __init__(
            self,
            metric: str,
            std: float = 0.01,
            num_results: int = 4,
            grace_period: int = 4,
            metric_threshold: Optional[float] = None,
            mode: Optional[str] = None,
            max_kld = 5.0,
    ):
        super().__init__(metric, std, num_results, grace_period, metric_threshold, mode)
        self._cv = self._std
        self._max_kld = max_kld

    def __call__(self, trial_id: str, result: Dict):
        metric_result = result.get(self._metric)

        if metric_result  > self._max_kld:
            return True

        self._trial_results[trial_id].append(metric_result)
        self._iter[trial_id] += 1

        # If still in grace period, do not stop yet
        if self._iter[trial_id] < self._grace_period:
            return False

        # If not enough results yet, do not stop yet
        if len(self._trial_results[trial_id]) < self._num_results:
            return False

        # If metric threshold value not reached, do not stop yet
        if self._metric_threshold is not None:
            if self._mode == "min" and metric_result > self._metric_threshold:
                return False
            elif self._mode == "max" and metric_result < self._metric_threshold:
                return False

        # Calculate stdev of last `num_results` results
        try:
            current_std = np.std(self._trial_results[trial_id])
            current_mu = np.mean(self._trial_results[trial_id])
            current_cv = current_std / abs(current_mu)
        except Exception:
            current_cv = float("inf")

        # If stdev is lower than threshold, stop early.
        return current_cv < self._cv


def main(
        prior: _BasePrior,
        simulator: _BaseSimulator = None,
        tune_id: str = 'test',
        prior_flow=True,
        num_samples=200,
        max_t=300,

):
    if not prior_flow:
        raise NotImplementedError('have yet to think about doing posteriors')

    param_space = {
        'prior_flow': prior_flow,
        'autoregressive': tune.choice((True, False)),
        'num_blocks': tune.randint(2, 5),
        'num_hidden_channels': tune.qrandint(10, 100, 5),
        'num_bins': tune.qrandint(4, 20, 4),
        'dropout_probability': tune.uniform(0.0, 0.5),
        'use_batch_norm': tune.choice((True, False)),
        'num_transforms': tune.randint(2, 5),
        'init_identity': tune.choice((True, False)),
        'learning_rate': tune.loguniform(1e-5, 3e-1),
        'weight_decay': tune.loguniform(1e-7, 1e-5),
        'batch_size': tune.choice(2 ** np.arange(4, 10)),
    }
    whatune = (
        'autoregressive', 'num_blocks', 'num_hidden_channels', 'num_bins', 'dropout_probability', 'use_batch_norm',
        'num_transforms', 'learning_rate', 'batch_size'
    )
    param_space = {k: v for k, v in param_space.items() if k in whatune}
    local_dir = os.path.join(SIM_DIR, 'ray_logs')
    dirpath = Path(local_dir) / tune_id
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    callback = None  # TODO for posterior trainine, useful to do some calibration automatically!

    metric = 'forward_kld'
    grace_period = 5
    tuner = tune.Tuner(
        trainable=tune.with_parameters(MFA_Flow, prior=prior, simulator=simulator),
        # trainable=tune.with_resources(
        #     tune.with_parameters(MFA_Flow, prior=prior, simulator=simulator),
        #     resources={"cpu": 1, "gpu": 0, "memory": 1e9},
        # ),
        tune_config=tune.TuneConfig(
            search_alg=None,  # NB defaults to random search, think of doing BOHB or BayesOptSearch
            metric=metric,
            mode='min',
            scheduler=tune.schedulers.ASHAScheduler(
                max_t=max_t,
                grace_period=grace_period,
                reduction_factor=2
            ),
            num_samples=num_samples,
        ),
        run_config=ray.air.RunConfig(
            local_dir=local_dir,
            name=tune_id,
            callbacks=callback,
            stop=CVStopper(  # NB checks for convergence based on CV
                metric=metric,
                std=0.005,  # TODO THIS SHOULD ACTUALLY THE CV!
                num_results=20,
                grace_period=grace_period,
                metric_threshold=None,
                mode='min',
                max_kld=5.0,
            ),
            log_to_file=True,
            # checkpoint_config=ray.air.CheckpointConfig(
            #     checkpoint_at_end=True
            # )
        ),
        param_space=param_space,
    )
    result = tuner.fit()
    return result


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
    from sbmfi.models.small_models import spiro
    from sbmfi.core.polytopia import sample_polytope
    from sbmfi.inference.priors import UniNetFluxPrior

    mdv_ds = torch.load(r"C:\python_projects\sbmfi\mdv_ds.pt")
    model, kwargs = spiro(
        backend='torch', v2_reversible=True, v5_reversible=False, build_simulator=True, which_measurements='com',
        which_labellings=['A', 'B']

    )
    up = UniNetFluxPrior(model.flux_coordinate_mapper, cache_size=100000)
    fcm_cyl = FluxCoordinateMapper(
        model,
        kernel_basis='svd',  # basis for null-space of simplified polytope
        basis_coordinates='cylinder',  # which variables will be considered free (basis or simplified)
        logit_xch_fluxes=False,  # whether to logit exchange fluxes
        hemi_sphere=False,
        scale_bound=1.0,
    )

    mdv_flow = flow_constructor(
        fcm=fcm_cyl,
        circular=True,
        embedding_net=None,
        num_context_channels=mdv_ds.data.shape[-1],
        autoregressive=True,
        num_blocks=4,
        num_hidden_channels=20,
        num_bins=10,
        dropout_probability=0.0,
        num_transforms=3,
        init_identity=True,
        permute=None,
        p=None,
        scale=0.3,
    )

    # flow_trainer(mdv_flow, mdv_ds, max_iter=50)
    #
    x, y = mdv_ds[[123456]]
    num_samples = 50
    xx = torch.tile(x, (num_samples, 1))
    samples = mdv_flow.sample(num_samples=num_samples, context=x)

