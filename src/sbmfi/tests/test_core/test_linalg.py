import math
import numpy as np
import scipy.special
import pytest

# Import torch if available.
try:
    import torch
except ImportError:
    torch = None

# Import the LinAlg class from your module.
# Adjust the import if your file name is different.
from sbmfi.core.linalg import LinAlg


# -------------------------------------------------------------------
# Fixtures for instantiating the LinAlg object for each backend.
# -------------------------------------------------------------------
@pytest.fixture(params=["numpy", "torch"])
def linalg(request):
    """
    Fixture returning a LinAlg instance for each backend.
    """
    return LinAlg(backend=request.param, seed=123)


# -------------------------------------------------------------------
# Test get_tensor (creation with and without indices)
# -------------------------------------------------------------------
def test_get_tensor(linalg):
    shape = (3, 3)
    # Use indices to set the diagonal entries.
    indices = np.array([[0, 0], [1, 1], [2, 2]])
    values = np.array([1, 2, 3])
    tensor = linalg.get_tensor(shape=shape, indices=indices, values=values,
                               squeeze=False, dtype=np.float64, device=None)
    if linalg.backend == "numpy":
        np.testing.assert_array_equal(np.diag(tensor), [1, 2, 3])
    else:
        np.testing.assert_array_equal(tensor.diag().cpu().numpy(), np.array([1, 2, 3]))


# -------------------------------------------------------------------
# Test LU factorization and solving linear systems
# -------------------------------------------------------------------
def test_lu_solve(linalg):
    A = np.array([[3, 1],
                  [1, 2]], dtype=np.float64)
    b = np.array([9, 8], dtype=np.float64)
    LU = linalg.LU(A)
    x = linalg.solve(LU, b)
    # Compute expected solution using NumPy's solver.
    sol = np.linalg.solve(A, b)
    if linalg.backend == "numpy":
        np.testing.assert_allclose(x, sol, rtol=1e-5)
    else:
        np.testing.assert_allclose(x.cpu().numpy(), sol, rtol=1e-5)


# -------------------------------------------------------------------
# Test add_at and dadd_at operations
# -------------------------------------------------------------------
def test_add_at(linalg):
    # Create a 1D tensor of zeros.
    x = linalg.zeros((5,))
    # Let y be a simple array.
    y = np.array([1, 2, 3, 4, 5])
    # indices: first column holds the index to update; second column selects a value from y.
    indices = np.array([[0, 1], [1, 2]])
    stoich = 2.0
    # Expected: x[0] += 2 * y[1] and x[1] += 2 * y[2]
    expected = np.zeros(5)
    expected[0] += 2 * y[1]
    expected[1] += 2 * y[2]
    res = linalg.add_at(x.copy(), y, indices, stoich)
    if linalg.backend == "numpy":
        np.testing.assert_array_equal(res, expected)
    else:
        np.testing.assert_array_equal(res.cpu().numpy(), expected)


def test_dadd_at(linalg):
    # Use a simple vector and check that the output shape remains unchanged.
    x = linalg.ones((5,))
    y = np.arange(1, 6)
    indices = np.array([[0, 1], [1, 2]])
    stoich = 1.5
    res = linalg.dadd_at(x.copy(), y, indices, stoich)
    if linalg.backend == "numpy":
        assert res.shape == x.shape
    else:
        assert res.cpu().numpy().shape == x.shape


# -------------------------------------------------------------------
# Test convolution
# -------------------------------------------------------------------
def test_convolve(linalg):
    a = np.array([1, 2, 3])
    v = np.array([0, 1])
    conv_result = linalg.convolve(a, v)
    expected = np.convolve(a, v)
    if linalg.backend == "numpy":
        np.testing.assert_allclose(conv_result, expected)
    else:
        np.testing.assert_allclose(conv_result.cpu().numpy(), expected)


# -------------------------------------------------------------------
# Test nonzero
# -------------------------------------------------------------------
def test_nonzero(linalg):
    A = np.array([[0, 1],
                  [2, 0]])
    indices, values = linalg.nonzero(A)
    # Expected nonzero indices (order might differ).
    expected_indices = [tuple(row) for row in np.array([[0, 1], [1, 0]])]
    res_indices = sorted([tuple(row) for row in indices])
    assert res_indices == sorted(expected_indices)
    # Expected nonzero values.
    expected_values = np.array([1, 2])
    np.testing.assert_array_equal(np.sort(values), np.sort(expected_values))


# -------------------------------------------------------------------
# Test tonp (conversion to NumPy array)
# -------------------------------------------------------------------
def test_tonp(linalg):
    if linalg.backend == "torch":
        t = linalg.get_tensor(shape=(2, 2), indices=None,
                              values=np.array([[1, 2], [3, 4]]),
                              squeeze=False, dtype=np.float64)
        np_arr = linalg.tonp(t)
        np.testing.assert_array_equal(np_arr, np.array([[1, 2], [3, 4]]))
    else:
        t = np.array([[1, 2], [3, 4]])
        np_arr = linalg.tonp(t)
        np.testing.assert_array_equal(np_arr, t)


# -------------------------------------------------------------------
# Test set_to (setting all elements)
# -------------------------------------------------------------------
def test_set_to(linalg):
    A = linalg.zeros((3, 3))
    A = linalg.set_to(A, 5)
    expected = np.full((3, 3), 5)
    if linalg.backend == "numpy":
        np.testing.assert_array_equal(A, expected)
    else:
        np.testing.assert_array_equal(A.cpu().numpy(), expected)


# -------------------------------------------------------------------
# Test random generation functions (randn, randu, randperm)
# -------------------------------------------------------------------
def test_rand_shapes(linalg):
    r1 = linalg.randn((4, 4))
    r2 = linalg.randu((4, 4))
    if linalg.backend == "numpy":
        assert r1.shape == (4, 4)
        assert r2.shape == (4, 4)
    else:
        assert tuple(r1.shape) == (4, 4)
        assert tuple(r2.shape) == (4, 4)


def test_randperm(linalg):
    perm = linalg.randperm(10)
    if linalg.backend == "numpy":
        assert set(perm) == set(range(10))
    else:
        assert set(perm.cpu().numpy()) == set(range(10))


# -------------------------------------------------------------------
# Test min and max (with return_indices)
# -------------------------------------------------------------------
def test_min_max(linalg):
    arr = np.array([[3, 1, 2],
                    [4, 0, 5]])
    min_val, min_idx = linalg.min(arr, dim=1, keepdims=False, return_indices=True)
    expected_min = np.argmin(arr, axis=1)
    np.testing.assert_array_equal(min_idx, expected_min)

    max_val, max_idx = linalg.max(arr, dim=1, keepdims=False, return_indices=True)
    expected_max = np.argmax(arr, axis=1)
    np.testing.assert_array_equal(max_idx, expected_max)


# -------------------------------------------------------------------
# Test logsumexp
# -------------------------------------------------------------------
def test_logsumexp(linalg):
    arr = np.array([1, 2, 3])
    result = linalg.logsumexp(arr, dim=0)
    expected = scipy.special.logsumexp(arr)
    if linalg.backend == "numpy":
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    else:
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5)


# -------------------------------------------------------------------
# Test basic tensor operations: permutax, transax, unsqueeze, cat.
# -------------------------------------------------------------------
def test_tensor_ops(linalg):
    if linalg.backend == "numpy":
        A = np.array([[1, 2],
                      [3, 4]])
        B = linalg.permutax(A, 1, 0)
        np.testing.assert_array_equal(B, A.T)
        C = linalg.transax(A, 0, 1)
        np.testing.assert_array_equal(C, A.T)
        D = linalg.unsqueeze(A, 0)
        np.testing.assert_array_equal(D, np.expand_dims(A, 0))
        E = linalg.cat([A, A], dim=0)
        np.testing.assert_array_equal(E, np.concatenate([A, A], axis=0))
    else:
        A = torch.tensor([[1, 2],
                          [3, 4]])
        B = linalg.permutax(A, 1, 0)
        np.testing.assert_array_equal(B.cpu().numpy(), A.t().cpu().numpy())
        C = linalg.transax(A, 0, 1)
        np.testing.assert_array_equal(C.cpu().numpy(), A.t().cpu().numpy())
        D = linalg.unsqueeze(A, 0)
        np.testing.assert_array_equal(D.cpu().numpy(), A.unsqueeze(0).cpu().numpy())
        E = linalg.cat([A, A], dim=0)
        np.testing.assert_array_equal(E.cpu().numpy(), torch.cat([A, A], dim=0).cpu().numpy())


# -------------------------------------------------------------------
# Test probability functions: norm_pdf, norm_log_pdf, norm_cdf, norm_inv_cdf.
# -------------------------------------------------------------------
def test_probability_functions(linalg):
    x = np.array([0.0, 0.5, 1.0])
    pdf = linalg.norm_pdf(x, mu=0.0, std=1.0)
    log_pdf = linalg.norm_log_pdf(x, mu=0.0, std=1.0)
    cdf = linalg.norm_cdf(x, mu=0.0, std=1.0)
    inv_cdf = linalg.norm_inv_cdf(cdf, mu=0.0, std=1.0)
    np.testing.assert_allclose(inv_cdf, x, rtol=1e-5)
    # Check that the log PDF equals log(pdf).
    np.testing.assert_allclose(log_pdf, np.log(pdf), rtol=1e-5)
