import jax.numpy as jnp

from warnings import warn
from jax import jit, grad, value_and_grad, jacfwd, jacrev, vmap

from MyTyping import *

from DecSample import SampleOracle
from utilities import check_gossip, sparsify, count_float_nonzero, Lp_norm


grad_norm_eps = 1e-5


class DecentralizedOptProblem:
    """
    This class implements the abstract framework of decentralized federated optimization problem.
    ----------
    F: Callable[[Array, Array], float]
        F(w; x).
    name: str
        Name of the optimization problem.
    dim_feature: int
        dim_feature = dim(x).
    dim_param: int
        dim_param = dim(w).
    M: int
        M = # devices.
    U: Array
        Gossip matrix, an M*M symmetric double stochastic matrix.
    w_0: Array
        Initial w.
    sample_oracle: SampleOracle
        Instance of SampleOracle class.
    w_star: Array
        Optimal w.
    obj_star: float
        Optimal value.
    """
    def __init__(self,
                 F: Callable[[Array, Array], float],
                 sample_oracle: SampleOracle,
                 name: str,
                 dim_param: int,
                 U: Array,
                 w_0: Array,
                 w_star: Array = None,
                 obj_star: float = None,
                 ):
        self.M = sample_oracle.M
        self.dim_feature = sample_oracle.dim_feature
        assert dim_param > 0
        check_gossip(U, self.M)
        assert w_0.shape == (dim_param,)
        if w_star is not None:
            assert w_star.shape == (dim_param,)

        test_sample = sample_oracle.sample(1)
        assert test_sample.shape == (self.M, 1, self.dim_feature)
        test_F = F(w_0, test_sample[0][0])

        self.name = name
        self.dim_param = dim_param
        self.U = U
        self.w_0 = w_0
        self.sample_oracle = sample_oracle
        self.w_star = w_star
        self.obj_star = obj_star

        grad_F = grad(F, argnums=0)
        hess_F = jacfwd(jacrev(F, argnums=0), argnums=0)
        value_grad_F = value_and_grad(F, argnums=0)

        batch_avg_F = lambda w, batch: jnp.average(vmap(F, in_axes=(None, 0), out_axes=0)(w, batch))
        batch_avg_grad_F = lambda w, batch: jnp.average(vmap(grad_F, in_axes=(None, 0), out_axes=0)(w, batch), axis=0)
        batch_avg_hess_F = lambda w, batch: jnp.average(vmap(hess_F, in_axes=(None, 0), out_axes=0)(w, batch), axis=0)
        batch_avg_val_grad_F = lambda w, batch: \
            tuple(jnp.average(result, axis=0) for result in
                  vmap(value_grad_F, in_axes=(None, 0), out_axes=(0, 0))(w, batch))

        dec_batch_avg_F = vmap(batch_avg_F, in_axes=(0, 0), out_axes=0)
        dec_batch_avg_grad_F = vmap(batch_avg_grad_F, in_axes=(0, 0), out_axes=0)
        dec_batch_avg_hess_F = vmap(batch_avg_hess_F, in_axes=(0, 0), out_axes=0)
        dec_batch_avg_val_grad_F = vmap(batch_avg_val_grad_F, in_axes=(0, 0), out_axes=(0, 0))

        # All these functions have been wrapped by jit by default.
        self.lambda_F = jit(F)
        self.lambda_grad_F = jit(grad_F)
        self.lambda_hess_F = jit(hess_F)
        self.lambda_val_grad_F = jit(value_grad_F)
        # Batch version, do not support non-batch input.
        self.lambda_batch_avg_F = jit(batch_avg_F)
        self.lambda_batch_avg_grad_F = jit(batch_avg_grad_F)
        self.lambda_batch_avg_hess_F = jit(batch_avg_hess_F)
        self.lambda_batch_avg_val_grad_F = jit(batch_avg_val_grad_F)
        # Decentralize version, do not support other versions of input.
        self.lambda_dec_batch_avg_F = jit(dec_batch_avg_F)
        self.lambda_dec_batch_avg_grad_F = jit(dec_batch_avg_grad_F)
        self.lambda_dec_batch_avg_hess_F = jit(dec_batch_avg_hess_F)
        self.lambda_dec_batch_avg_val_grad_F = jit(dec_batch_avg_val_grad_F)

        self.F_exec_time = 0
        self.grad_F_exec_time = 0
        self.hess_F_exec_time = 0

        self.w_star = None
        self.obj_star = None

    # Non-batch F.
    def F(self, w: Array, xi: Array) -> float:
        self.F_exec_time += 1
        return self.lambda_F(w, xi)

    # Non-batch grad F w.r.t. w.
    def grad_F(self, w: Array, xi: Array) -> Array:
        self.grad_F_exec_time += 1
        return self.lambda_grad_F(w, xi)

    # Non-batch hess F w.r.t. w.
    def hess_F(self, w: Array, xi: Array) -> Array:
        self.hess_F_exec_time += 1
        return self.lambda_hess_F(w, xi)

    # Non-batch value and grad F w.r.t. w.
    def val_and_grad_F(self, w: Array, xi: Array) -> Tuple[float, Array]:
        self.F_exec_time += 1
        self.grad_F_exec_time += 1
        return self.lambda_val_grad_F(w, xi)

    # Batch F.
    def batch_avg_F(self, w: Array, batch: Array) -> Array:
        batch_size = batch.shape[0]
        self.F_exec_time += batch_size
        return self.lambda_batch_avg_F(w, batch)

    # Batch grad F w.r.t. w.
    def batch_avg_grad_F(self, w: Array, batch: Array) -> Array:
        batch_size = batch.shape[0]
        self.grad_F_exec_time += batch_size
        return self.lambda_batch_avg_grad_F(w, batch)

    # Batch hess F w.r.t. w.
    def batch_avg_hess_F(self, w: Array, batch: Array) -> Array:
        batch_size = batch.shape[0]
        self.hess_F_exec_time += batch_size
        return self.lambda_batch_avg_hess_F(w, batch)

    # Batch value and grad F w.r.t. w.
    def batch_avg_val_and_grad_F(self, w: Array, batch: Array) -> Tuple[Array, Array]:
        batch_size = batch.shape[0]
        self.F_exec_time += batch_size
        self.grad_F_exec_time += batch_size
        return self.lambda_batch_avg_val_grad_F(w, batch)

    # Decentralized batch F.
    def dec_batch_avg_F(self, dec_w: Array,  dec_batch: Array) -> Array:
        batch_size = dec_batch.shape[1]
        self.F_exec_time += batch_size * self.M
        return self.lambda_dec_batch_avg_F(dec_w, dec_batch)

    # Decentralized batch grad F w.r.t. w.
    def dec_batch_avg_grad_F(self, dec_w: Array, dec_batch: Array) -> Array:
        batch_size = dec_batch.shape[1]
        self.grad_F_exec_time += batch_size * self.M
        return self.lambda_dec_batch_avg_grad_F(dec_w, dec_batch)

    # Decentralized batch hess F w.r.t. w.
    def dec_batch_avg_hess_F(self, dec_w: Array, dec_batch: Array) -> Array:
        batch_size = dec_batch.shape[1]
        self.hess_F_exec_time += batch_size * self.M
        return self.lambda_dec_batch_avg_hess_F(dec_w, dec_batch)

    # Decentralized batch value and grad F w.r.t. w.
    def dec_batch_avg_val_and_grad_F(self, dec_w: Array, dec_batch: Array) -> Tuple[Array, Array]:
        batch_size = dec_batch.shape[1]
        self.F_exec_time += batch_size * self.M
        self.grad_F_exec_time += batch_size * self.M
        return self.lambda_dec_batch_avg_val_grad_F(dec_w, dec_batch)

    # Evaluate objective function, which may be very slow.
    def evaluate_obj_and_grad_norm(self, w: Array, expectation_flag=False) -> Tuple[float, float]:
        # We assume each device has same number of samples.
        full_x = self.sample_oracle.get_full_data(expectation_flag=expectation_flag).reshape(-1, self.dim_feature)
        obj = self.lambda_batch_avg_F(w, full_x)
        grad_norm = jnp.linalg.norm(self.lambda_batch_avg_grad_F(w, full_x))
        return float(obj), float(grad_norm)

    # Summary current parameters.
    def summary_result(self, w: Array, sparsity: int, mirror_map_q: float, expectation_flag=False) -> Dict:
        result_dict = {}
        result_dict["obj"], result_dict["grad_norm"] = self.evaluate_obj_and_grad_norm(w, expectation_flag)
        print(f'Obj: {result_dict["obj"]}.')
        print(f'Grad norm: {result_dict["grad_norm"]}.')
        if self.obj_star is None:
            pass
            # warn('We do not know obj_star.')
        else:
            # Obj - Obj^*
            result_dict["obj_gap"] = result_dict["obj"] - float(self.obj_star)
            print(f'Obj gap: {result_dict["obj_gap"]}.')

        # First few terms are not zero by default.
        sparse_w = sparsify(w, sparsity)
        tp = count_float_nonzero(sparse_w[:sparsity])
        fp = sparsity - tp
        fn = sparsity - tp
        result_dict["sparse_recover_cnt"] = tp
        result_dict["sparse_recover_ratio"] = tp / sparsity
        result_dict["sparse_recover_f1"] = (2 * tp) / (2 * tp + fp + fn)
        print(f'Sparse recover: {result_dict["sparse_recover_cnt"]}/{sparsity}={result_dict["sparse_recover_ratio"]}.')

        if self.w_star is None:
            pass
            # warn('We do not know w_star.')
        else:
            # Summary non-zero elements.
            # norm{w-w^*}_1.
            result_dict["l1_norm_gap"] = float(jnp.linalg.norm(w - self.w_star, ord=1))
            result_dict["sparse_l1_norm_gap"] = float(jnp.linalg.norm(sparse_w - self.w_star, ord=1))
            print(f'L1 norm gap: {result_dict["l1_norm_gap"]}.')
            print(f'Sparse L1 norm gap: {result_dict["sparse_l1_norm_gap"]}.')

            # norm{w-w^*}_2.
            result_dict["l2_norm_gap"] = float(jnp.linalg.norm(w - self.w_star))
            result_dict["sparse_l2_norm_gap"] = float(jnp.linalg.norm(sparse_w - self.w_star))
            print(f'L2 norm gap: {result_dict["l2_norm_gap"]}.')
            print(f'Sparse L2 norm gap: {result_dict["sparse_l2_norm_gap"]}.')

            # norm{w-w^*}_q^2.
            result_dict["lq_norm_gap_square"] = float(Lp_norm(w - self.w_star, p=mirror_map_q)) ** 2.
            result_dict["sparse_lq_norm_gap_square"] = float(Lp_norm(sparse_w - self.w_star, p=mirror_map_q)) ** 2.
            print(f'Lq norm gap square: {result_dict["lq_norm_gap_square"]}.')
            print(f'Sparse Lq norm gap square: {result_dict["sparse_lq_norm_gap_square"]}.')
        return result_dict

    # Reset counters.
    def reset_counters(self):
        self.F_exec_time = 0
        self.grad_F_exec_time = 0
        self.hess_F_exec_time = 0
        return

    def set_optimal_solution(self, w_star: Array, expectation_flag=False):
        assert w_star.shape == (self.dim_param,)
        self.w_star = w_star
        self.obj_star, grad_norm_star = self.evaluate_obj_and_grad_norm(w_star, expectation_flag)
        if grad_norm_star > grad_norm_eps:
            warn(f'Grad norm too large: {grad_norm_star}!')
        return
