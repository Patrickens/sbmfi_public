import numpy as np
import copy
# np.seterr(all='raise')  # throws errors with np.nextafter(0, 1, 'float32')
import scipy
from scipy.special import expit, logit
import random
import inspect
import math

def _conditional_torch_import():
    # if 'torch' in sys.modules:
    #     return
    try:
        global torch
        import torch
        version = int(torch.__version__.split('.')[1])
    except ImportError as e:
        print('torch not installed, cannot use this backend')
        raise e
    global _NP_TORCH_DTYPE

    _NP_TORCH_DTYPE = {
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
        np.double: torch.double,
    }
    return version


def _merge_duplicate_indices(indices, values):
    if values.size == 0:
        return indices, values
    uniq_indices, where, counts = np.unique(indices, axis=0, return_counts=True, return_index=True)
    new_values = values[where]
    for i in np.where(counts > 1)[0]:
        aka = (uniq_indices[i] == indices).all(1)
        new_values[i] = values[aka].sum()
    return uniq_indices, new_values


def torch_auto_jacobian(inputs, outputs, create_graph=False, squeeze=False):
    """
    TODO: https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html
        check whether torch has a better/ faster implementation of this via the link above
    Stolen from: https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa#gistcomment-2955749
    Computes the jacobian of outputs with respect to inputs

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """

    if inputs.ndim == 1:
        nbatch, nin = 1, inputs.shape[0]
    elif inputs.ndim == 2:
        nbatch, nin = inputs.shape
    else:
        raise NotImplementedError

    nout = outputs.shape[-1]

    jac = torch.zeros(size=(nbatch, nin, nout), dtype=torch.double)
    for i, out in enumerate(outputs.view(-1)):
        col_i = torch.autograd.grad(out, inputs, retain_graph=True, create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            if inputs.ndim == 1:
                jac[i//nout, :, i%nout] = col_i
            else:
                jac[i//nout, :, i%nout] = col_i[i//nout, :]

    if create_graph:
        jac.requires_grad_()
    if squeeze:
        return jac.squeeze(0)
    return jac

_SQRT2PI = math.sqrt(2 * math.pi)
_ONEBYSQRT2PI = 1.0 / _SQRT2PI
_SQRT2 = math.sqrt(2)


class NumpyBackend(object):
    _DEFAULT_FKWARGS = {
        'LU': {'overwrite_a': True, 'check_finite': False},
        'solve': {'trans': 0, 'overwrite_b': True, 'check_finite': False},
    }
    _AUTO_DIFF = False
    _BATCH_PROCESSING = True

    def __init__(self, seed=None, dtype=np.double, **kwargs):
        self._rng = np.random.default_rng(seed=seed)
        self._def_dtype = dtype

    def get_tensor(self, shape, indices, values, squeeze, dtype, device):
        if shape is not None:
            if (values is not None) and values.size:
                if dtype is None:
                    dtype = values.dtype
                    if dtype in (np.float32, np.float64):
                        dtype = self._def_dtype  # maing sure that sbi works
            elif dtype is None:
                dtype = self._def_dtype
            A = np.zeros(shape=shape, dtype=dtype)
            if (indices is not None) and indices.size:
                indices, values = _merge_duplicate_indices(indices=indices, values=values)
                indices = tuple(col for col in indices.T)
                A[indices] = values
        else:
            if dtype is None:
                dtype = values.dtype
                if dtype in (np.float32, np.float64):
                    dtype = self._def_dtype  # maing sure that sbi works
            A = np.array(values, dtype=dtype)

        if (A.ndim == 3) and squeeze:
            A = A.squeeze(0)
        return A

    @staticmethod
    def LU(A, **kwargs):
        # this is fucking slow! This has to do with the fact that a fortran object is passed around;
        #  for our use-case (solving only once or at most a len(linsys_reactions) times, this is not worth it
        # return scipy.linalg.lu(a=A, **kwargs) # this is also fekkin slow...

        if A.ndim == 3:
            return [NumpyBackend.LU(A[i, :, :], **kwargs) for i in range(A.shape[0])]
        return scipy.linalg.lu_factor(a=A, **kwargs)

    @staticmethod
    def vecopy(A):
        return A.copy()

    @staticmethod
    def solve(LU, b, **kwargs):
        # lu_solve is only useful with lu_factor, which is horribly slow
        # P, L, U = LU
        # z = P.T @ b
        # y = scipy.linalg.solve_triangular(L, z, lower=True, **kwargs)
        # x = scipy.linalg.solve_triangular(U, y, lower=False, **kwargs)
        # return x

        if b.ndim == 3:
            solution = np.zeros(b.shape)
            for i in range(b.shape[0]):
                solution[i, :, :] = NumpyBackend.solve(LU=LU[i], b=b[i, :, :], **kwargs)
            return solution
        return scipy.linalg.lu_solve(lu_and_piv=LU, b=b, **kwargs)  # TODO: overwrite_b=False, trans=0, check_finite=False

    @staticmethod
    def add_at(x, y, indices, stoich):
        np.add.at(x, indices[:, 0], stoich * np.prod(y[indices[:, 1:]], axis=1))
        return x

    @staticmethod
    def dadd_at(x, y, indices, stoich):
        sub_indices = np.arange(1, indices.shape[1])
        for i in sub_indices:
            np.add.at(x, indices[:, 0], np.prod(y[indices[:, sub_indices[sub_indices != i]]], axis=1) * x[indices[:, i]] * stoich)
        return x

    @staticmethod
    def convolve(a, v):
        if a.ndim == 2:
            solution = np.zeros((a.shape[0], a.shape[1] + v.shape[1] - 1))
            for i in range(a.shape[0]):
                solution[i, :] = NumpyBackend.convolve(a[i, :], v[i, :])
            return solution
        return np.convolve(a=a, v=v)

    @staticmethod
    def nonzero(A):
        nonzero_indices = A.nonzero()
        return np.array(nonzero_indices, dtype=int).T, A[nonzero_indices]

    @staticmethod
    def tonp(A):
        return A

    @staticmethod
    def set_to(A, vals):
        if isinstance(vals, (int, float)):
            A[:] = vals
        else:
            A[:] = vals[:]
        return A

    @staticmethod
    def permutax(A, *args):
        return A.transpose(*args)

    @staticmethod
    def transax(A, dim0, dim1):
        return np.swapaxes(A, dim0, dim1)

    @staticmethod
    def unsqueeze(A, dim):
        return np.expand_dims(A, dim)

    @staticmethod
    def cat(As, dim=0):
        return np.concatenate(As, axis=dim)

    @staticmethod
    def max(A, dim=None, keepdims=False):
        return A.max(dim, keepdims=keepdims)

    @staticmethod
    def min(A, dim=None, keepdims=False):
        return A.min(dim, keepdims=keepdims)

    @staticmethod
    def view(A, shape):
        return A.reshape(shape)

    @staticmethod
    def logsumexp(A, dim=0, keepdims=False):
        return scipy.special.logsumexp(A, dim, keepdims=keepdims)

    @staticmethod
    def atan2( x, y):
        return

    @staticmethod
    def triu_indices(n, k):
        return np.triu_indices(n=n, k=k)

    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = self._def_dtype
        return np.zeros(shape, dtype)

    def ones(self, shape, dtype=None):
        if dtype is None:
            dtype = self._def_dtype
        return np.ones(shape, dtype)

    def randn(self, shape, dtype=np.float64):
        return self._rng.standard_normal(shape, dtype=dtype)

    def randu(self, shape, dtype=np.float64):
        return self._rng.random(shape, dtype=dtype)

    def randperm(self, n):
        return self._rng.permutation(n)

    def multinomial(self, n, p):
        counts = self._rng.multinomial(1, p, size=(n, *p.shape[:-1]))
        return np.where(counts)[1]

    def choice(self, n, tot, replace=False):
        return self._rng.choice(tot, n, replace=replace)

    # def categorical(self, sample_shape, probs=None, logits=None):
    #     if logits is not None:
    #         if probs is not None:
    #             raise ValueError
    #         probs = np.exp(logits)
    #     probs = probs / probs.sum(-1) # make sure sums to 1
    #     return self._rng.multinomial(n=1, pvals=probs, size=sample_shape)


class FactorExTorchBackend():
    # necessary so that everything in a batch is computed except for the ones that fail, with lu_factor the whole batch fails
    @staticmethod
    def LU(A, **kwargs):
        return torch.linalg.lu_factor_ex(A)

    @staticmethod
    def solve(LU, b, **kwargs):
        if b.ndim == 1:
            b = torch.atleast_2d(b).T
        return torch.lu_solve(b, *LU[:2])


class NonDiffTorchBackend():
    """TODO use this backend if torch<1.10.0, since there lu_solve is not differentiable yet"""
    # torch.lu_solve(b, *LU) is currently not differentiable
    #   this will soon be solved: https://github.com/pytorch/pytorch/pull/61681
    @staticmethod
    def LU(A, **kwargs):
        return A

    @staticmethod
    def solve(LU, b, **kwargs):
        # NOTE: return 1d thing if were working with cumos, otherwise return 2d thing...
        # TODO: make use of the returned LU for jacobians...
        if b.ndim == 1:
            b = torch.atleast_2d(b).T
        X = torch.linalg.solve(LU, b)
        # X = X.squeeze()
        # if X.dim() < 2:
        #     X = X.unsqueeze(0)
        return X


class TorchBackend(object):
    # https://github.com/torch/torch7/wiki/Torch-for-Numpy-users

    # torch.lu_solve(b, *LU) is currently not differentiable
    #   this will soon be solved: https://github.com/pytorch/pytorch/pull/61681
    _DEFAULT_FKWARGS = {
        'LU': {},
        'solve': {},
    }
    _AUTO_DIFF = True
    _BATCH_PROCESSING = True

    def __init__(self, seed=None, solver='lu_solve_ex', device='cpu', dtype=np.double, **kwargs):
        version = _conditional_torch_import()

        self._def_dtype = _NP_TORCH_DTYPE[dtype]
        self._device = torch.device('cpu')
        if (torch.cuda.is_available()) and ('cuda' in device):
            self._device = torch.device(device)

        self._rng = torch.Generator(self._device)
        if isinstance(seed, int):
            self._rng.manual_seed(seed)

        if (version < 10) or (solver == 'lu_solve_nondiff'):
            TorchBackend.LU = staticmethod(NonDiffTorchBackend.LU)
            TorchBackend.solve = staticmethod(NonDiffTorchBackend.solve)
        elif solver == 'lu_solve_ex':
            TorchBackend.LU = staticmethod(FactorExTorchBackend.LU)
            TorchBackend.solve = staticmethod(FactorExTorchBackend.solve)
        elif solver != 'lu_solve':
            raise ValueError('not a legal solver option')

        if self._def_dtype == torch.double:
            def_tens_type = torch.DoubleTensor
        elif self._def_dtype == torch.float32:
            def_tens_type = torch.FloatTensor

        torch.set_default_tensor_type(def_tens_type)
        torch.autograd.set_detect_anomaly(True)

    def get_tensor(self, shape, indices, values, squeeze, dtype, device):
        if shape is not None:
            if (values is not None) and values.size:
                if dtype is None:
                    dtype = values.dtype.type
                    if dtype in (np.float32, np.float64):
                        dtype = self._def_dtype  # maing sure that sbi works
            elif dtype is None:
                dtype = self._def_dtype

            if not isinstance(dtype, torch.dtype):
                dtype = _NP_TORCH_DTYPE[dtype]

            if device is None:
                device = self._device

            A = torch.zeros(size=shape, dtype=dtype, device=device)
            if (indices is not None) and indices.size:
                indices, values = _merge_duplicate_indices(indices=indices, values=values)
                indices = torch.as_tensor(indices, dtype=torch.int64, device=device)
                indices = tuple(col for col in indices.T)
                A[indices] = torch.as_tensor(values, device=device)
        else:
            if dtype is None:
                if isinstance(values, np.ndarray):
                    dtype = values.dtype.type
                else:
                    dtype = values.dtype

                if not isinstance(dtype, torch.dtype):
                    dtype = _NP_TORCH_DTYPE[dtype]
                if dtype in (torch.float32, torch.float64):
                    dtype = self._def_dtype

            if not isinstance(dtype, torch.dtype):
                dtype = _NP_TORCH_DTYPE[dtype]

            A = torch.as_tensor(values, device=device, dtype=dtype)
        if (A.ndim == 3) and squeeze:
            A = A.squeeze(0)
        return A

    @staticmethod
    def LU(A, **kwargs):# NOTE: this is currently differentiable via autograd! torch>0.10.0
        # return torch.lu(A) # TODO check whether this is desirable?
        return torch.linalg.lu_factor(A)

    @staticmethod
    def solve(LU, b, **kwargs):
        if b.ndim == 1:
            b = torch.atleast_2d(b).T
        return torch.lu_solve(b, *LU)

    @staticmethod
    def vecopy(A):
        return A.clone()

    @staticmethod
    def add_at(x, y, indices, stoich):
        x.index_add_(0, indices[:, 0], stoich * torch.prod(y[indices[:, 1:]], dim=1))
        return x

    @staticmethod
    def dadd_at(x, y, indices, stoich):
        sub_indices = torch.arange(1, indices.shape[1])
        for i in sub_indices:
            x.index_add_(0, indices[:, 0],
                         torch.prod(y[indices[:, sub_indices[sub_indices != i]]], axis=1) * x[indices[:, i]] * stoich)
        return x

    @staticmethod
    def convolve(x, y):
        if x.ndim == 1:
            x = x.view(1, 1, -1)
            y = y.view(1, 1, -1).flip(2)
        elif x.ndim == 2:
            x = x.unsqueeze(0)
            y = y.unsqueeze(1).flip(2)
        else:
            raise ValueError(f'only up to 2D tensors!')
        # padding = torch.min(torch.tensor([v1.shape[-1], v2.shape[-1]])).item() - 1
        return torch.conv1d(x, y, padding=y.size(2) - 1, groups=x.size(1)).squeeze()

    @staticmethod
    def nonzero(A):
        nonzero_indices = torch.nonzero(A)
        indices = tuple(col for col in nonzero_indices.T)
        return nonzero_indices, A[indices]

    @staticmethod
    def tonp(A):
        if torch.is_tensor(A):
            return A.to(device='cpu', copy=False).detach().numpy()
        return A

    @staticmethod
    def view(A, shape):
        return A.view(shape)

    @staticmethod
    def diff(inputs, outputs):
        return torch_auto_jacobian(inputs=inputs, outputs=outputs, create_graph=False).detach()

    @staticmethod
    def permutax(A, *args):
        return A.permute(*args)

    @staticmethod
    def transax(A, dim0, dim1):
        return A.transpose(dim0, dim1)

    @staticmethod
    def unsqueeze(A, dim):
        return A.unsqueeze(dim)

    @staticmethod
    def cat(As, dim):
        return torch.cat(As, dim)

    @staticmethod
    def max(A, dim=None, keepdims=False):
        if dim is not None:
            return A.max(dim, keepdims=keepdims).values
        return A.max()

    @staticmethod
    def min(A, dim=None, keepdims=False):
        if dim is not None:
            return A.min(dim, keepdims=keepdims).values
        return A.min()

    @staticmethod
    def logsumexp(A, dim=0, keepdims=False):
        return torch.logsumexp(A, dim, keepdims)

    @staticmethod
    def triu_indices(n, k):
        indices = torch.triu_indices(row=n, col=n, offset=k)
        return indices[0], indices[1]

    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = self._def_dtype
        elif dtype in _NP_TORCH_DTYPE:
            dtype = _NP_TORCH_DTYPE[dtype]
        return torch.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        if dtype is None:
            dtype = self._def_dtype
        return torch.ones(shape, dtype=dtype)

    def multinomial(self, n, p, replace=True):
        return torch.multinomial(input=p, num_samples=n, generator=self._rng, replacement=replace)

    def randn(self, shape, dtype=None):
        if dtype is None:
            dtype = self._def_dtype
        elif not isinstance(dtype, torch.dtype):
            dtype = _NP_TORCH_DTYPE[dtype]
        return torch.randn(shape, generator=self._rng, dtype=dtype)

    def randu(self, shape, dtype=np.double):
        if dtype is None:
            dtype = self._def_dtype
        elif not isinstance(dtype, torch.dtype):
            dtype = _NP_TORCH_DTYPE[dtype]
        return torch.rand(shape, generator=self._rng, dtype=dtype, device=self._device)

    def randperm(self, n):
        return torch.randperm(n, generator=self._rng)

    def choice(self, n, tot, replace=False):
        probs = torch.ones(tot) / tot
        return self.multinomial(n, probs, replace=replace)

    # def categorical(self, sample_shape, probs=None, logits=None):
    #     return torch.distributions.Categorical(probs, logits).sample(sample_shape)


class CupyBackend(object):
    # TODO: make a cupy backend: https://cupy.dev/
    pass

_2PI = 2 * math.pi
_SQRT2PI = math.sqrt(_2PI)
_ONEBYSQRT2PI = 1.0 / _SQRT2PI
_SQRT2 = math.sqrt(2)
_1_SQRT2 = 1.0 / _SQRT2
_LN2PI_2 = math.log(_2PI) / 2.0

class LinAlg(object):

    _SAME_SIGNATURE = [
        # these functions have the same signature in numpy and torch, thus we can dynamically add them
        'exp', 'log10', 'log', 'atleast_2d', 'diag', 'trace', 'allclose', 'where', 'arange', 'divide',
        'prod', 'diagonal', 'tile', 'sqrt', 'isclose', 'sum', 'mean', 'amax', 'linspace', 'cov', 'split',
        'linalg.svd', 'linalg.norm', 'linalg.pinv', 'linalg.cholesky', 'eye', 'stack', 'minimum', 'maximum',
        'cumsum', 'argmin', 'argmax', 'clip', 'special.erf', 'special.erfinv', 'special.expit', 'special.logit',
        'argsort', 'unique', 'cov', 'split', 'arctan2', 'sin', 'cos', 'sign', 'diff', 'nansum', 'isnan', 'float_power'
    ]

    def __getstate__(self):
        return_dict = self.__dict__.copy()
        functions = self._fill_functions(self._backwargs['backend'])
        return_dict['_BACKEND'] = None
        return_dict = {k: v for k, v in return_dict.items() if k not in functions}
        return return_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
        kwargs = state['_backwargs']
        backend = kwargs['backend']
        if backend == 'numpy':
            self._BACKEND = NumpyBackend(**kwargs)
        elif backend == 'torch':
            self._BACKEND = TorchBackend(**kwargs)
        functions = self._fill_functions(self._backwargs['backend'])
        self.__dict__.update(functions)

    def __eq__(self, other):
        if not isinstance(other, LinAlg):
            return False
        return self._backwargs == other._backwargs

    def __init__(
            self,
            backend:str,
            batch_size: int = 1,
            solver: str = 'lu_solve_ex', # solver to use for the linear system A_tot • x = b and for Jacobians
            device: str = 'cpu',
            fkwargs: dict = None,
            auto_diff: bool = False,
            seed: int = None,
            dtype=np.double
    ):
        random.seed(seed)
        np.random.seed(seed)

        if dtype not in (np.double, np.float64, np.float32, np.single):
            raise ValueError('not a supported default float type')

        self._backwargs = {'backend': backend, 'seed': seed, 'solver': solver, 'device': device, 'dtype': dtype}

        if backend == 'numpy':
            self._BACKEND = NumpyBackend(seed=seed, dtype=dtype)
        elif backend == 'torch':
            self._BACKEND = TorchBackend(seed=seed, solver=solver, device=device, dtype=dtype)
        else:
            raise ValueError('not a valid backend, you bellend')

        functions = self._fill_functions(backend)
        self.__dict__.update(functions)

        if fkwargs is None:
            fkwargs = {}

        self._auto_diff = False
        if self._BACKEND._AUTO_DIFF and auto_diff:
            self._auto_diff = auto_diff

        self._batch_size = 1
        if self._BACKEND._BATCH_PROCESSING and (batch_size > 1):
            self._batch_size = int(batch_size)

        kwargs = copy.deepcopy(self._BACKEND._DEFAULT_FKWARGS)
        for function_name, function_kwargs in kwargs.items():
            user_function_kwargs = fkwargs.get(function_name)
            if user_function_kwargs:
                function_kwargs.update(user_function_kwargs)
        self._fkwargs = kwargs

    def _fill_functions(self, backend):
        # TODO make function partial and pass self._device in torch
        functions = {}
        for fname in LinAlg._SAME_SIGNATURE:
            pack_func = fname.split('.')
            n = len(pack_func)
            if n == 1:
                if backend == 'numpy':
                    package = np
                elif backend == 'torch':
                    package = torch
            elif n == 2:
                if backend == 'torch':
                    package = torch.__dict__[pack_func[0]]
                elif backend == 'numpy':
                    if pack_func[0] == 'special':
                        package = scipy.special
                    else:
                        package = np.__dict__[pack_func[0]]
                fname = pack_func[1]
            function = package.__dict__[fname]
            functions[fname] = function
        return functions

    @property
    def backend(self):
        if isinstance(self._BACKEND, NumpyBackend):
            return 'numpy'
        elif isinstance(self._BACKEND, TorchBackend):
            return 'torch'

    def get_tensor(self, shape=None, indices=None, values=None, squeeze=False, dtype=None, device=None):
        # TODO make the default shape (0, ) and the default dtype np.float64!
        return self._BACKEND.get_tensor(shape, indices, values, squeeze, dtype, device)

    def LU(self, A, **kwargs):
        return self._BACKEND.LU(A, **{**self._fkwargs['LU'], **kwargs})

    def vecopy(self, A):
        return self._BACKEND.vecopy(A)

    def solve(self, LU, b, **kwargs):
        return self._BACKEND.solve(LU, b, **{**self._fkwargs['solve'], **kwargs})

    def add_at(self, x, y, indices, stoich):
        return self._BACKEND.add_at(x, y, indices, stoich)

    def dadd_at(self, x, y, indices, stoich):
        return self._BACKEND.dadd_at(x, y, indices, stoich)

    def convolve(self, a, v):
        # TODO: https://en.wikipedia.org/wiki/Toeplitz_matrix#Discrete_convolution
        #   Toeplitz discrete convolution might be a better option
        #   We would have to store a Toeplitz matrix for every ConvolutedEmu object
        #   this is a head-ache and not terribly 'clean' and Im not sure whether this
        #   would actually make anything much faster
        return self._BACKEND.convolve(a, v)

    def nonzero(self, A):
        return self._BACKEND.nonzero(A)

    def tonp(self, A):
        return self._BACKEND.tonp(A)

    def view(self, A, shape):
        return self._BACKEND.view(A, shape)

    def set_to(self, A, vals):
        return NumpyBackend.set_to(A, vals)

    def diff(self, inputs, outputs):
        return self._BACKEND.diff(inputs, outputs)

    def randn(self, shape, dtype=None):
        return self._BACKEND.randn(shape, dtype)

    def randu(self, shape, dtype=None):
        return self._BACKEND.randu(shape, dtype)

    def randperm(self, n):
        return self._BACKEND.randperm(n)

    def permutax(self, A, *args):
        return self._BACKEND.permutax(A, *args)

    def transax(self, A, dim0=-2, dim1=-1):
        return self._BACKEND.transax(A, dim0, dim1)

    def unsqueeze(self, A, dim):
        return self._BACKEND.unsqueeze(A, dim)

    def cat(self, As, dim=0):
        return self._BACKEND.cat(As, dim)

    def choice(self, n, tot, replace=False):
        return self._BACKEND.choice(n, tot, replace)

    def categorical(self, sample_shape, probs=None, logits=None):
        return self._BACKEND.categorical(sample_shape, probs, logits)

    def sample_hypersphere(self, shape, radius=1.0):
        rnd = self.randu(shape)
        return rnd / self.norm(rnd, 2, -1, True)

    def _compute_xi(self, A, mu=0.0, std=1.0):
        return (A - mu) / std

    def norm_pdf(self, A, mu=0.0, std=1.0):
        xi = self._compute_xi(A, mu, std)
        return _ONEBYSQRT2PI * (1.0 / std) * self.exp(-xi**2 * 0.5)

    def norm_log_pdf(self, A, mu=0.0, std=1.0):
        xi = self._compute_xi(A, mu, std)
        return -_LN2PI_2 - self.log(std) - xi**2 * 0.5

    def norm_cdf(self, A, mu=0.0, std=1.0):
        xi = self._compute_xi(A, mu, std)
        return 0.5 + self.erf(xi * _1_SQRT2) * 0.5

    def norm_inv_cdf(self, u, mu=0.0, std=1.0):
        return mu + std * _SQRT2 * self.erfinv(2 * u - 1.0)

    def trunc_norm_pdf(self, A, lo, hi, mu=0.0, std=1.0):
        norm_pdf = self.norm_pdf(A, mu, std)
        alpha = self.norm_cdf(lo, mu, std)
        beta  = self.norm_cdf(hi, mu, std)
        return norm_pdf / (beta - alpha)

    def trunc_norm_pdf2(self, A, lo, hi, mu=0.0, std=1.0):
        xi = (A - mu) / std
        alpha = (lo - mu) / std
        beta = (hi - mu) / std
        A = 0.5 * (1 + self.erf((alpha / math.sqrt(2))))
        B = 0.5 * (1 + self.erf((beta / math.sqrt(2))))
        norm_pdf = (1 / math.sqrt(2* math.pi)) * self.exp(-0.5 * xi**2)
        return norm_pdf / (std * (B - A))

    def trunc_norm_log_pdf(self, A, lo, hi, mu=0.0, std=1.0, *args, **kwargs):
        norm_log_pdf = self.norm_log_pdf(A, mu, std)
        alpha = self.norm_cdf(lo, mu, std)
        beta = self.norm_cdf(hi, mu, std)
        return norm_log_pdf - self.log((beta - alpha))

    def trunc_norm_cdf(self, A, lo, hi, mu=0.0, std=1.0):
        norm_cdf = self.norm_cdf(A, mu, std)
        alpha = self.norm_cdf(lo, mu, std)
        beta  = self.norm_cdf(hi, mu, std)
        return (norm_cdf - alpha) / (beta - alpha)

    def trunc_norm_inv_cdf(self, u, lo, hi, mu=0.0, std=1.0, return_log_prob=False):
        alpha = self.norm_cdf(lo, mu, std)
        beta  = self.norm_cdf(hi, mu, std)
        uu = alpha + u * (beta - alpha)
        samples = self.norm_inv_cdf(uu, mu, std)
        if return_log_prob:
            log_prob = self.norm_log_pdf(samples, mu, std) - self.log(beta - alpha) # TODO perhaps there are still double computations somewhere?
            return samples, log_prob
        return samples

    def unif_inv_cdf(self, u, lo=0.0, hi=1.0, return_log_prob=False):
        diff = hi - lo
        samples = lo + u * diff
        if return_log_prob:
            log_probs = self.log(self.ones(u.shape) / diff)
            return samples, log_probs
        return samples

    def unif_log_pdf(self, A, lo=0.0, hi=1.0, mu=0.0, *args, **kwargs):
        if isinstance(mu, float):
            out_shape = A.shape
        else:
            out_shape = tuple(np.maximum(A.shape, mu.shape))
        return self.log(self.ones(out_shape) / (hi - lo))

    def sample_bounded_distribution(self, shape: tuple, lo, hi, mu=0.0, std=0.1, which='unif', return_log_prob=False):
        if not (lo.shape == hi.shape):
            raise ValueError(f'lo.shape: {lo.shape}, hi.shape: {hi.shape}')
        u = self.randu(shape=(*shape, *lo.shape))
        if which == 'unif':
            return self.unif_inv_cdf(u, lo, hi, return_log_prob)
        elif which == 'gauss':
            # truncated multivariate normal sampling
            # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
            # publication: Efficient Sampling Methods for Truncated Multivariate
            #   Normal and Student-t Distributions Subject to Linear
            #   Inequality Constraints
            return self.trunc_norm_inv_cdf(u, lo, hi, mu, std, return_log_prob)
        else:
            raise ValueError

    def triu_indices(self, n, k):
        return self._BACKEND.triu_indices(n, k)

    def bounded_distribution_log_prob(self, x, lo, hi, mu, std=0.1, which='unif', old_is_new=False, unsqueeze=True):
        if not (lo.shape == hi.shape):  # TODO should work with float lo and hi
            raise ValueError

        if not x.ndim == mu.ndim:
            raise ValueError('conjolo')
            # this means we passed old particles and we need to unsqueeze here!

        if unsqueeze:
            mu = self.unsqueeze(mu, 1)
            x = self.unsqueeze(x, 0)

        if which == 'unif':
            pdf = self.unif_log_pdf
        elif which == 'gauss':
            pdf = self.trunc_norm_log_pdf
        else:
            raise ValueError

        if old_is_new:
            rows, cols = self.triu_indices(x.shape[1], k=1)
            if unsqueeze:
                uptri_x = x[0, cols]
                uptri_mu = mu[rows, 0]
            else:
                lo = lo[rows]
                hi = hi[rows]
                uptri_x = x[rows, cols]
                uptri_mu = mu[cols, rows]
            uptri_probs = pdf(uptri_x, lo, hi, uptri_mu, std)
            out_shape = tuple(np.maximum(x.shape, mu.shape))
            log_probs = self.get_tensor(shape=out_shape)
            log_probs[rows, cols] = uptri_probs
            return log_probs + self.transax(log_probs, dim0=0, dim1=1)
        else:
            return pdf(x, lo, hi, mu, std)

    def multinomial(self, n, p):
        return self._BACKEND.multinomial(n, p)

    def max(self, A, dim=None, keepdims=False):
        return self._BACKEND.max(A, dim, keepdims)

    def min(self, A, dim=None, keepdims=False):
        return self._BACKEND.min(A, dim, keepdims)

    def logsumexp(self, A, dim=0):
        return self._BACKEND.logsumexp(A, dim)

    def zeros(self, shape, dtype=None):
        return self._BACKEND.zeros(shape, dtype)

    def ones(self, shape, dtype=None):
        return self._BACKEND.ones(shape, dtype)

    def tensormul_T(self, A, x, dim0=-2, dim1=-1):  # TODO add b argument that adds to x after multiplication?
        return self.transax(A @ self.transax(x, dim0=dim0, dim1=dim1), dim0=dim0, dim1=dim1)

    def eval_std_normal(self, x):
        return x ** 2 - _SQRT2PI

    def cartesian(self, A, ):
        pass

    def min_pos_max_neg(self, alpha, return_what=1, keepdims=False):
        inf = float('inf')
        if return_what > -1:
            alpha_max = self.vecopy(alpha)
            alpha_max[alpha_max <= 0.0] = inf
            alpha_max = self.min(alpha_max, -1, keepdims)
            if return_what == 1:
                return alpha_max
        if return_what < 1:
            alpha_min = self.vecopy(alpha)
            alpha_min[alpha_min >= 0.0] = -inf
            alpha_min = self.max(alpha_min, -1, keepdims)
            if return_what == -1:
                return alpha_min
        return alpha_min, alpha_max

    def scale(self, A, lo, hi, rev=False):
        if rev:
            return A * (hi - lo) + lo
        return (A - lo) / (hi - lo)

    def unsqueeze_like(self, A, like):
        n_unsqueezes = like.ndim - A.ndim
        return A[(None,) * n_unsqueezes + (...,)]


if __name__ == "__main__":
    import pickle, timeit, cProfile, torch

    nl = LinAlg(backend='numpy')
    p = nl.randu((5, ))
    p = p / p.sum()

    tl = LinAlg(backend='torch')
    tp = tl.get_tensor(values=p)
    rands = tl.randu(59) * 6 - 3

    print(tl.min_pos_max_neg(rands, return_what=0))
    print(rands)

    # a = torch.zeros((3,3,3))
    # b = torch.zeros((3,3,3,5))
    # l = LinAlg(backend='torch')
    # l.unsqueeze_like(a, b)

    # n_dim = 2
    # dim_slicer = slice(0, n_dim) if n_dim > 1 else 0
    # n_el = 2
    #
    # lo = l.get_tensor(values=np.array([
    #     [0.0, 0.5],
    #     [0.0, 0.5],
    # ])[dim_slicer, :n_el])
    # hi = l.get_tensor(values=np.array([
    #     [1.5, 1.0],
    #     [1.5, 1.0],
    # ])[dim_slicer, :n_el])
    # mu = l.get_tensor(values=np.array([
    #     [0.2, 0.55],
    #     [0.2, 0.55],
    # ])[dim_slicer, :n_el])
    # shape = (3,)
    # std = l.get_tensor(values=np.array(0.1))
    # which = 'unif'
    #
    # dang = l.sample_bounded_distribution(shape=shape, lo=lo, hi=hi, mu=mu, std=std, which=which)
    # log_probs = l.proposal_log_probs(dang, lo, hi, std, which)

