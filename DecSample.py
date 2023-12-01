import jax
import jax.numpy as jnp

from jax import vmap, jit
from functools import partial

from MyTyping import *


class SampleOracle:
    """
    This class implements the abstract framework of sample oracle for decentralized federated optimization problem.
    ----------
    dim_feature: int
        dim_feature = dim(x).
    M: int
        M = # devices.
    """
    def __init__(self,
                 dim_feature: int,
                 M: int,
                 jax_key: KeyArray,
                 ):
        self.dim_feature = dim_feature
        self.M = M
        self.jax_key = jax_key
        self.jax_key_backup = jnp.copy(jax_key)
        return

    def sample(self, batch_size: int) -> Array:
        # Return shape (M, batch_size, dim_feature)
        raise NotImplemented('Abstract method.')

    def sample_with_seed(self, batch_size: int, seed: int) -> Array:
        # Return shape (M, batch_size, dim_feature)
        raise NotImplemented('Abstract method.')

    def get_full_data(self, size: int = None, expectation_flag=False) -> Array:
        raise NotImplemented('Abstract method.')

    def reset(self, jax_key: KeyArray = None):
        if jax_key is not None:
            self.jax_key = jax_key
        else:
            self.jax_key = jnp.copy(self.jax_key_backup)
        return


class FlowDataSampler:
    def __init__(self,
                 dim_feature: int,
                 M: int,
                 ):
        self.dim_feature = dim_feature
        self.M = M
        return

    def sample(self, batch_size: int, jax_key: KeyArray, expectation_flag=False) -> Array:
        # Return shape (M, batch_size, dim_feature)
        raise NotImplemented('Abstract method.')


class FlowSampleOracle(SampleOracle):
    """
    This class implements the SampleOracle based on a given data flow.
    ----------
    flow_data_sampler: FlowDataSampler
    """
    def __init__(self,
                 flow_data_sampler: FlowDataSampler,
                 full_data_default_size: int,
                 jax_key: KeyArray,
                 ):
        self.flow_data_sampler = flow_data_sampler
        self.full_data_default_size = full_data_default_size
        self.full_data_first_call_flag = False
        self.full_data = None
        super().__init__(dim_feature=flow_data_sampler.dim_feature, M=flow_data_sampler.M, jax_key=jax_key)
        return

    def sample(self, batch_size: int, expectation_flag=False) -> Array:
        # Return shape (M, batch_size, dim_feature)
        self.jax_key, subkey = jax.random.split(self.jax_key)
        return self.flow_data_sampler.sample(batch_size, subkey, expectation_flag)

    # Sample with a specific seed.
    def sample_with_seed(self, batch_size: int, seed: int) -> Array:
        # Return shape (M, batch_size, dim_feature)
        tmp_jax_key = jax.random.PRNGKey(seed)
        return self.flow_data_sampler.sample(batch_size, tmp_jax_key)

    def get_full_data(self, size: int = None, expectation_flag=False) -> Array:
        # Return shape (M, size, dim_feature)
        if size is None:
            size = self.full_data_default_size
        else:
            self.full_data_first_call_flag = False
            self.full_data = None
        if not self.full_data_first_call_flag:
            self.full_data = self.sample(size, expectation_flag)
        self.full_data_first_call_flag = True
        return self.full_data


class HomoDatasetSampleOracle(SampleOracle):
    """
    This class implements the SampleOracle based on a homogeneous dataset.
    ----------
    datasets: Array
        Jax array with shape (M, n, dim_feature)
    """
    def __init__(self,
                 datasets: ArrayLike,
                 jax_key: KeyArray,
                 ):
        # Runtime type validation:
        if not isinstance(datasets, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {type(datasets)}")
        if isinstance(datasets, ndarray):
            datasets = jnp.array(datasets)

        M, self.each_dataset_size, dim_feature = datasets.shape
        self.datasets = datasets

        super().__init__(dim_feature=dim_feature, M=M, jax_key=jax_key)
        return

    def sample(self, batch_size: int) -> Array:
        # Return shape (M, batch_size, dim_feature)
        auxilary_idx = jnp.arange(self.M)[:, None].repeat(batch_size, axis=1)
        self.jax_key, subkey = jax.random.split(self.jax_key)
        idx = jax.random.randint(subkey, shape=(self.M, batch_size), minval=0, maxval=self.each_dataset_size)
        return self.datasets[auxilary_idx, idx, :]

    def get_full_data(self, size: int = None, expectation_flag=False) -> Array:
        assert not expectation_flag
        if size is None or size > self.each_dataset_size:
            return self.datasets
        else:
            return self.datasets[:, :size, :]


# Generate random w with given sparsity and threshold.
def generate_sparse_random_w(sparsity: int, dim_param: int, threshold: float,
                             jax_key: KeyArray,
                             # sparse_pos_random_flag=False
                             ) -> Array:
    key, subkey = jax.random.split(jax_key)
    nonzero_indices = jnp.arange(0, sparsity)

    # Add threshold to abs of N(0, 1).
    nonzero_values = jax.random.normal(key, shape=(sparsity,))
    nonzero_values_sign = jnp.sign(nonzero_values)
    nonzero_values_abs = jnp.abs(nonzero_values) + threshold
    nonzero_values = nonzero_values_sign * nonzero_values_abs
    w = jnp.zeros((dim_param,)).at[nonzero_indices].set(nonzero_values)
    return w


# Return i.i.d. samples from N(mean, cov).
def sample_normal_dataset(mean: Array, cov: Array, M: int, each_device_size: int, jax_key: KeyArray) -> Array:
    normal_data = jax.random.multivariate_normal(key=jax_key, mean=mean, cov=cov, shape=(M, each_device_size))
    return normal_data  # Shape (M, each_device_size, dim_param)


# Return gaussian mixture samples with given dec mean and cov.
@partial(jit, static_argnames=('each_device_size',))
def sample_gaussian_mixture_dataset(dec_mean: Array, dec_cov: Array,
                                    each_device_size: int, jax_key: KeyArray) -> Array:
    # dec_mean: shape (M, dim_feature)
    # dec_cov: shape (M, dim_feature, dim_feature)
    M = dec_mean.shape[0]
    dim_feature = dec_mean.shape[1]
    dec_normal_data = \
        jax.random.multivariate_normal(jax_key, mean=dec_mean, cov=dec_cov, shape=(each_device_size, M)).swapaxes(0, 1)
    return dec_normal_data  # Shape (M, each_device_size, dim_feature)


# This class assume [0]: Y, [1]: 1 (if bias_flag == True), [2:]: X.
# y_{m, i} = func(w_m, x_{m, i}) + epsilon_{m, i}.
# Note: dim_feature include bias and y!!!
class FlowGaussianMixtureRegressionDataSampler(FlowDataSampler):
    def __init__(self,
                 dim_feature: int,
                 dim_param: int,
                 M: int,
                 func: Callable[[Array, Array], float],
                 dec_param: Array,
                 dec_x_mean: Array,
                 dec_x_cov: Array,
                 dec_error_var: Array,
                 bias_flag: bool = True
                 ):
        self.dim_covariate_feature = dim_feature - 1
        if bias_flag:
            self.dim_covariate_feature_without_bias = self.dim_covariate_feature - 1
        else:
            self.dim_covariate_feature_without_bias = self.dim_covariate_feature
        assert dec_param.shape == (M, dim_param)
        assert dec_x_mean.shape == (M, self.dim_covariate_feature_without_bias)
        assert dec_x_cov.shape == (M, self.dim_covariate_feature_without_bias, self.dim_covariate_feature_without_bias)
        assert dec_error_var.shape == (M,)

        self.dec_x_mean = dec_x_mean
        self.dec_x_cov = dec_x_cov
        self.dec_error_var = dec_error_var

        self.dim_param = dim_param
        self.func = func  # y_{m, i} = func(w_m, x_{m, i}) + epsilon_{m, i}.
        self.batch_func = vmap(func, in_axes=(None, 0), out_axes=0)
        self.dec_batch_func = jit(vmap(self.batch_func, in_axes=(0, 0), out_axes=0))
        self.dec_param = dec_param
        self.bias_flag = bias_flag
        super().__init__(dim_feature=dim_feature, M=M)
        return

    def sample(self, batch_size: int, jax_key: KeyArray, expectation_flag=False) -> Array:
        assert not expectation_flag
        # Return shape (M, batch_size, dim_feature)
        datasets = jnp.empty((self.M, batch_size, self.dim_feature), dtype=float)
        jax_key_iterator = iter(jax.random.split(jax_key, num=10))
        X = jnp.ones((self.M, batch_size, self.dim_covariate_feature), dtype=float).at[:, :, int(self.bias_flag):].set(
            sample_gaussian_mixture_dataset(dec_mean=self.dec_x_mean, dec_cov=self.dec_x_cov,
                                            each_device_size=batch_size, jax_key=next(jax_key_iterator))
        )
        epsilon = jnp.sqrt(self.dec_error_var)[:, None] * \
                  jax.random.normal(key=next(jax_key_iterator), shape=(self.M, batch_size))
        Y = self.dec_batch_func(self.dec_param, X) + epsilon
        datasets = datasets.at[:, :, 0].set(Y)
        datasets = datasets.at[:, :, 1:].set(X)
        return datasets

    # Do not add noise epsilon.
    def sample_noiseless_variate(self, batch_size: int, jax_key: KeyArray) -> Array:
        # Return shape (M, batch_size, dim_feature)
        datasets = jnp.empty((self.M, batch_size, self.dim_feature), dtype=float)
        jax_key_iterator = iter(jax.random.split(jax_key, num=10))
        X = jnp.ones((self.M, batch_size, self.dim_covariate_feature), dtype=float).at[:, :, int(self.bias_flag):].set(
            sample_gaussian_mixture_dataset(dec_mean=self.dec_x_mean, dec_cov=self.dec_x_cov,
                                            each_device_size=batch_size, jax_key=next(jax_key_iterator))
        )
        Y = self.dec_batch_func(self.dec_param, X)
        datasets = datasets.at[:, :, 0].set(Y)
        datasets = datasets.at[:, :, 1:].set(X)
        return datasets


class FlowGaussianMixtureBinaryClassificationDataSampler(FlowDataSampler):
    def __init__(self,
                 dim_feature: int,
                 dim_param: int,
                 M: int,
                 func: Callable[[Array, Array], float],  # Should map into [0, 1].
                 dec_param: Array,
                 dec_x_mean: Array,
                 dec_x_cov: Array,
                 bias_flag: bool = True
                 ):
        self.dim_covariate_feature = dim_feature - 1
        if bias_flag:
            self.dim_covariate_feature_without_bias = self.dim_covariate_feature - 1
        else:
            self.dim_covariate_feature_without_bias = self.dim_covariate_feature
        assert dec_param.shape == (M, dim_param)
        assert dec_x_mean.shape == (M, self.dim_covariate_feature_without_bias)
        assert dec_x_cov.shape == (M, self.dim_covariate_feature_without_bias, self.dim_covariate_feature_without_bias)

        self.dec_x_mean = dec_x_mean
        self.dec_x_cov = dec_x_cov

        self.dim_param = dim_param
        self.func = func  # y_{m, i} ~ Ber(func(w_m, x_{m, i})).
        self.batch_func = vmap(func, in_axes=(None, 0), out_axes=0)
        self.dec_batch_func = jit(vmap(self.batch_func, in_axes=(0, 0), out_axes=0))
        self.dec_param = dec_param
        self.bias_flag = bias_flag
        super().__init__(dim_feature=dim_feature, M=M)
        return


    def sample(self, batch_size: int, jax_key: KeyArray, expectation_flag=False) -> Array:
        # Return shape (M, batch_size, dim_feature)
        datasets = jnp.empty((self.M, batch_size, self.dim_feature), dtype=float)
        jax_key_iterator = iter(jax.random.split(jax_key, num=10))
        X = jnp.ones((self.M, batch_size, self.dim_covariate_feature), dtype=float).at[:, :, int(self.bias_flag):].set(
            sample_gaussian_mixture_dataset(dec_mean=self.dec_x_mean, dec_cov=self.dec_x_cov,
                                            each_device_size=batch_size, jax_key=next(jax_key_iterator))
        )
        if not expectation_flag:
            Y = jax.random.bernoulli(key=next(jax_key_iterator),
                                     p=self.dec_batch_func(self.dec_param, X),
                                     shape=(self.M, batch_size))
            Y = jnp.array(Y, dtype=float)
        else:
            # Expectation version.
            Y = self.dec_batch_func(self.dec_param, X)

        datasets = datasets.at[:, :, 0].set(Y)
        datasets = datasets.at[:, :, 1:].set(X)
        return datasets


class FlowGaussianMixtureRegressionSampleOracle(FlowSampleOracle):
    def __init__(self,
                 jax_key: KeyArray,
                 full_data_default_size: int,
                 dim_feature: int,
                 dim_param: int,
                 M: int,
                 func: Callable[[Array, Array], float],
                 dec_param: Array,
                 dec_x_mean: Array,
                 dec_x_cov: Array,
                 dec_error_var: Array,
                 bias_flag: bool = True,
                 ):
        flow_data_sampler = FlowGaussianMixtureRegressionDataSampler(
            dim_feature=dim_feature,
            dim_param=dim_param,
            M=M,
            func=func,
            dec_param=dec_param,
            dec_x_mean=dec_x_mean,
            dec_x_cov=dec_x_cov,
            dec_error_var=dec_error_var,
            bias_flag=bias_flag,
        )
        super().__init__(flow_data_sampler=flow_data_sampler,
                         full_data_default_size=full_data_default_size, jax_key=jax_key)
        return


class FlowGaussianMixtureBinaryClassificationSampleOracle(FlowSampleOracle):
    def __init__(self,
                 jax_key: KeyArray,
                 full_data_default_size: int,
                 dim_feature: int,
                 dim_param: int,
                 M: int,
                 func: Callable[[Array, Array], float],
                 dec_param: Array,
                 dec_x_mean: Array,
                 dec_x_cov: Array,
                 bias_flag: bool = True,
                 ):
        flow_data_sampler = FlowGaussianMixtureBinaryClassificationDataSampler(
            dim_feature=dim_feature,
            dim_param=dim_param,
            M=M,
            func=func,
            dec_param=dec_param,
            dec_x_mean=dec_x_mean,
            dec_x_cov=dec_x_cov,
            bias_flag=bias_flag,
        )
        super().__init__(flow_data_sampler=flow_data_sampler,
                         full_data_default_size=full_data_default_size, jax_key=jax_key)
        return


class FlowHomoGaussianRegressionHeterParamSampleOracle(FlowGaussianMixtureRegressionSampleOracle):
    def __init__(self,
                 jax_key: KeyArray,
                 full_data_default_size: int,
                 dim_feature: int,
                 dim_param: int,
                 M: int,
                 func: Callable[[Array, Array], float],
                 dec_param: Array,
                 x_mean: Array,
                 x_cov: Array,
                 error_var: float,
                 bias_flag: bool = True,
                 ):
        dec_x_mean = jnp.repeat(x_mean[None, :], repeats=M, axis=0)
        dec_x_cov = jnp.repeat(x_cov[None, :, :], repeats=M, axis=0)
        dec_error_var = error_var * jnp.ones((M,))
        super().__init__(
                 jax_key=jax_key,
                 full_data_default_size=full_data_default_size,
                 dim_feature=dim_feature,
                 dim_param=dim_param,
                 M=M,
                 func=func,
                 dec_param=dec_param,
                 dec_x_mean=dec_x_mean,
                 dec_x_cov=dec_x_cov,
                 dec_error_var=dec_error_var,
                 bias_flag=bias_flag,
        )
        return


class FlowHomoGaussianBinaryClassificationHeterParamSampleOracle(FlowGaussianMixtureBinaryClassificationSampleOracle):
    def __init__(self,
                 jax_key: KeyArray,
                 full_data_default_size: int,
                 dim_feature: int,
                 dim_param: int,
                 M: int,
                 func: Callable[[Array, Array], float],
                 dec_param: Array,
                 x_mean: Array,
                 x_cov: Array,
                 bias_flag: bool = True,
                 ):
        dec_x_mean = jnp.repeat(x_mean[None, :], repeats=M, axis=0)
        dec_x_cov = jnp.repeat(x_cov[None, :, :], repeats=M, axis=0)
        super().__init__(
                 jax_key=jax_key,
                 full_data_default_size=full_data_default_size,
                 dim_feature=dim_feature,
                 dim_param=dim_param,
                 M=M,
                 func=func,
                 dec_param=dec_param,
                 dec_x_mean=dec_x_mean,
                 dec_x_cov=dec_x_cov,
                 bias_flag=bias_flag,
        )
        return


class FlowHomoGaussianSparseGLMSampleOracle(FlowHomoGaussianRegressionHeterParamSampleOracle):
    def __init__(self,
                 jax_key: KeyArray,
                 full_data_default_size: int,
                 sparsity: int,
                 dim_param: int,
                 M: int,
                 x_nonzero_mean: Array,
                 x_nonzero_cov: Array,
                 x_zero_mean: Array,
                 x_zero_cov: Array,
                 nonzero_link_func: Callable[[float], float],
                 zero_link_func: Callable[[float], float],
                 error_var: float,
                 param_scale=1.,
                 nonzero_param_threshold=1.,
                 param_mode='all_one',
                 bias_flag=True,
                 ):
        assert x_nonzero_mean.shape == (sparsity,)
        assert x_nonzero_cov.shape == (sparsity, sparsity)
        self.sparsity = sparsity

        # Set x_mean and x_cov.
        x_mean = jnp.concatenate((x_nonzero_mean, x_zero_mean))
        x_cov = jnp.block([
            [x_nonzero_cov, jnp.zeros((x_nonzero_mean.size, x_zero_mean.size), dtype=float)],
            [jnp.zeros((x_zero_mean.size, x_nonzero_mean.size), dtype=float), x_zero_cov]
        ])
        # Set GLM func.
        func = lambda w, x: nonzero_link_func(jnp.dot(w[:sparsity], x[:sparsity])) + \
                            zero_link_func(jnp.dot(w[sparsity:], x[sparsity:]))

        # Set decentralized non-zero parameters.
        jax_key, subkey = jax.random.split(jax_key)
        nonzero_param = jax.random.normal(key=subkey, shape=(1, sparsity))
        nonzero_param_sign = jnp.sign(nonzero_param)
        nonzero_param_abs = jnp.abs(nonzero_param) + nonzero_param_threshold
        nonzero_param = nonzero_param_sign * nonzero_param_abs
        nonzero_param *= param_scale

        dec_nonzero_param = jnp.repeat(nonzero_param, repeats=M, axis=0)

        if param_mode == 'all_one':
            Md2 = M // 2
            assert M % 2 == 0
            ones = jnp.ones((sparsity,)) * param_scale
            dec_nonzero_param = dec_nonzero_param.at[:Md2].add(ones)
            dec_nonzero_param = dec_nonzero_param.at[Md2:].add(-ones)
        elif param_mode == 'random':
            jax_key, subkey = jax.random.split(jax_key)
            dec_nonzero_param += (jax.random.normal(key=subkey, shape=(M, sparsity)) * param_scale)
        else:
            raise NotImplementedError

        # Set decentralized zero parameters.
        if param_mode == 'all_one':
            Md2 = M // 2
            assert M % 2 == 0
            zero_param = jnp.ones((dim_param - sparsity,))  # L^p norm is (sparsity)^(1/p)
            dec_zero_param = jnp.empty(shape=(M, dim_param - sparsity))
            dec_zero_param = dec_zero_param.at[:Md2].set(jnp.repeat(zero_param[None, :], repeats=Md2, axis=0))
            dec_zero_param = dec_zero_param.at[Md2:].set(jnp.repeat(-zero_param[None, :], repeats=Md2, axis=0))
            dec_zero_param *= param_scale
        elif param_mode == 'random':
            # Use Dirichlet(1) for simplicity.
            jax_key, subkey = jax.random.split(jax_key)
            dec_zero_param = jax.random.dirichlet(key=subkey, alpha=jnp.ones((M,)), shape=(dim_param - sparsity,)).T
            dec_zero_param -= 1. / M
            dec_zero_param *= param_scale
        else:
            raise NotImplementedError

        dec_param = jnp.empty((M, dim_param)).at[:, :sparsity].set(dec_nonzero_param)
        dec_param = dec_param.at[:, sparsity:].set(dec_zero_param)

        super().__init__(
                 jax_key=jax_key,
                 full_data_default_size=full_data_default_size,
                 dim_feature=dim_param+1,
                 dim_param=dim_param,
                 M=M,
                 func=func,
                 dec_param=dec_param,
                 x_mean=x_mean,
                 x_cov=x_cov,
                 error_var=error_var,
                 bias_flag=bias_flag,
        )
        return


class FlowHomoGaussianSparseLinearBinaryClassificationSampleOracle(FlowHomoGaussianBinaryClassificationHeterParamSampleOracle):
    def __init__(self,
                 jax_key: KeyArray,
                 full_data_default_size: int,
                 sparsity: int,
                 dim_param: int,
                 M: int,
                 x_mean: Array,
                 x_cov: Array,
                 func: Callable[[float], float],
                 param_scale=1.,
                 nonzero_param_threshold=1.,
                 param_mode='all_one',
                 bias_flag=False,
                 ):
        assert x_mean.shape == (dim_param,)
        assert x_cov.shape == (dim_param, dim_param)
        assert bias_flag is False

        self.sparsity = sparsity

        # Set decentralized non-zero parameters.
        jax_key, subkey = jax.random.split(jax_key)
        nonzero_param = jax.random.normal(key=subkey, shape=(1, sparsity))
        nonzero_param_sign = jnp.sign(nonzero_param)
        nonzero_param_abs = jnp.abs(nonzero_param) + nonzero_param_threshold
        nonzero_param = nonzero_param_sign * nonzero_param_abs
        nonzero_param *= param_scale

        # Order: beta_1, ..., beta_L, beta_-1, ..., beta_-L;
        # beta_i = (gamma_i, delta_i); beta_-i = (gamma_i, -delta_i);
        # L = Md2 = M // 2
        dec_nonzero_param = jnp.repeat(nonzero_param, repeats=M, axis=0)

        Md2 = M // 2
        assert M % 2 == 0
        if param_mode == 'all_one':
            Md4 = M // 4
            assert M % 4 == 0
            ones = jnp.ones((sparsity,)) * param_scale
            dec_nonzero_param = dec_nonzero_param.at[:Md4].add(ones)
            dec_nonzero_param = dec_nonzero_param.at[Md4: 2*Md4].add(-ones)
            dec_nonzero_param = dec_nonzero_param.at[2*Md4: 3*Md4].add(ones)
            dec_nonzero_param = dec_nonzero_param.at[3*Md4:].add(-ones)
        elif param_mode == 'random':
            jax_key, subkey = jax.random.split(jax_key)
            dec_nonzero_param_perturb = jax.random.normal(key=subkey, shape=(Md2, sparsity)) * param_scale
            dec_nonzero_param_perturb = jnp.concatenate([dec_nonzero_param_perturb, dec_nonzero_param_perturb], axis=0)
            dec_nonzero_param = dec_nonzero_param + dec_nonzero_param_perturb
        else:
            raise NotImplementedError

        # Set decentralized zero parameters.
        if param_mode == 'all_one':
            zero_param = jnp.ones((dim_param - sparsity,))  # L^p norm is (sparsity)^(1/p)
            dec_zero_param = jnp.empty(shape=(M, dim_param - sparsity))
            dec_zero_param = dec_zero_param.at[:Md2].set(jnp.repeat(zero_param[None, :], repeats=Md2, axis=0))
            dec_zero_param = dec_zero_param.at[Md2:].set(jnp.repeat(-zero_param[None, :], repeats=Md2, axis=0))
            dec_zero_param *= param_scale
        elif param_mode == 'random':
            jax_key, subkey = jax.random.split(jax_key)

            dec_zero_param = jax.random.normal(key=subkey, shape=(Md2, dim_param - sparsity)) * param_scale
            dec_zero_param = jnp.concatenate([dec_zero_param, -dec_zero_param], axis=0)
        else:
            raise NotImplementedError

        dec_param = jnp.empty((M, dim_param)).at[:, :sparsity].set(dec_nonzero_param)
        dec_param = dec_param.at[:, sparsity:].set(dec_zero_param)

        # GLM
        result_func = lambda w, x: func(jnp.inner(w, x))

        super().__init__(
                 jax_key=jax_key,
                 full_data_default_size=full_data_default_size,
                 dim_feature=dim_param+1,
                 dim_param=dim_param,
                 M=M,
                 func=result_func,
                 dec_param=dec_param,
                 x_mean=x_mean,
                 x_cov=x_cov,
                 bias_flag=bias_flag,
        )
        return
