import jax.numpy as jnp

from jax import jit, vmap

from MyTyping import *
from utilities import Lp_norm, replace_nan_with_zero, sparsify, project_onto_l1_ball_center_w
from DecOptProblem import DecentralizedOptProblem


INIT_C_SEED = 2147483647


class DecOptMethod:
    def __init__(self,
                 dec_opt_problem: DecentralizedOptProblem,
                 *args,
                 **kwargs
                 ):
        self.dec_opt_problem = dec_opt_problem

        self.M = dec_opt_problem.M
        self.U = dec_opt_problem.U
        self.dim_feature = dec_opt_problem.dim_feature

        self.sample_oracle = dec_opt_problem.sample_oracle
        return


class CentraSGD(DecOptMethod):
    def __init__(self,
                 dec_opt_problem: DecentralizedOptProblem,
                 client_step_size: float,
                 batch_size: int,
                 *args,
                 **kwargs
                 ):
        super().__init__(
            dec_opt_problem=dec_opt_problem,
        )
        assert client_step_size > 0
        assert batch_size > 0

        self.client_step_size = client_step_size
        self.batch_size = batch_size
        self.w = dec_opt_problem.w_0
        self.dec_w = jnp.repeat(self.w[None, :], repeats=self.M, axis=0)
        return

    def client_local_update(self):
        batch = self.sample_oracle.sample(self.batch_size).reshape(-1, self.dim_feature)
        batch_avg_grad = self.dec_opt_problem.batch_avg_grad_F(w=self.w, batch=batch)
        self.w -= self.client_step_size * batch_avg_grad
        self.dec_w = jnp.repeat(self.w[None, :], repeats=self.M, axis=0)
        return


class SparseCentraSGD(DecOptMethod):
    def __init__(self,
                 dec_opt_problem: DecentralizedOptProblem,
                 client_step_size: float,
                 batch_size: int,
                 sparsity: int,
                 *args,
                 **kwargs
                 ):
        super().__init__(
            dec_opt_problem=dec_opt_problem,
        )
        assert client_step_size > 0
        assert batch_size > 0

        self.client_step_size = client_step_size
        self.batch_size = batch_size
        self.sparsity = sparsity
        self.sparse_w = dec_opt_problem.w_0[:sparsity]
        self.dim_param = int(dec_opt_problem.w_0.size)
        self.w = jnp.zeros((self.dim_param,)).at[:self.sparsity].set(self.sparse_w)
        self.dec_w = jnp.repeat(self.w[None, :], repeats=self.M, axis=0)
        return

    def client_local_update(self):
        batch = self.sample_oracle.sample(self.batch_size).reshape(-1, self.dim_feature)
        extend_w = jnp.zeros((self.dim_param,)).at[:self.sparsity].set(self.sparse_w)
        batch_avg_grad = self.dec_opt_problem.batch_avg_grad_F(w=extend_w, batch=batch)
        sparse_grad = batch_avg_grad[:self.sparsity]
        self.sparse_w -= self.client_step_size * sparse_grad
        self.w = jnp.zeros((self.dim_param,)).at[:self.sparsity].set(self.sparse_w)
        self.dec_w = jnp.repeat(self.w[None, :], repeats=self.M, axis=0)
        return



class DecSCAFFOLD(DecOptMethod):
    def __init__(self,
                 dec_opt_problem: DecentralizedOptProblem,
                 gradient_tracking_flag: bool,
                 client_step_size: float,
                 server_step_size: float,
                 local_iteration_number: int,
                 batch_size: int,
                 *args,
                 **kwargs
                 ):
        super().__init__(
            dec_opt_problem=dec_opt_problem,
        )
        assert client_step_size > 0 and server_step_size > 0
        assert local_iteration_number > 0 and batch_size > 0

        self.gradient_tracking_flag = gradient_tracking_flag
        self.client_step_size = client_step_size
        self.server_step_size = server_step_size
        self.local_iteration_number = local_iteration_number
        self.batch_size = batch_size

        self.dec_w = jnp.repeat(dec_opt_problem.w_0[None, :], repeats=self.M, axis=0)
        if self.gradient_tracking_flag:
            dec_batch = self.sample_oracle.sample_with_seed(batch_size, seed=INIT_C_SEED)
            dec_batch_avg_grad = self.dec_opt_problem.dec_batch_avg_grad_F(dec_w=self.dec_w, dec_batch=dec_batch)
            self.dec_c = self.U @ dec_batch_avg_grad - dec_batch_avg_grad

        return

    def client_local_update(self):
        old_dec_w = jnp.copy(self.dec_w)
        for k in range(self.local_iteration_number):
            dec_batch = self.sample_oracle.sample(self.batch_size)
            dec_batch_avg_grad = self.dec_opt_problem.dec_batch_avg_grad_F(dec_w=self.dec_w, dec_batch=dec_batch)
            if self.gradient_tracking_flag:
                dec_batch_avg_grad += self.dec_c
            self.dec_w -= self.client_step_size * dec_batch_avg_grad
        if self.gradient_tracking_flag:
            dec_delta = (1. / (self.local_iteration_number * self.client_step_size)) * (self.dec_w - old_dec_w)
            self.dec_c += (dec_delta - self.U @ dec_delta)
        lr = self.server_step_size
        self.dec_w = self.U @ (old_dec_w * (1. - lr) + self.dec_w * lr)
        return


class DecFedDAGT(DecOptMethod):
    def __init__(self,
                 dec_opt_problem: DecentralizedOptProblem,
                 gradient_tracking_flag: bool,
                 client_step_size: float,
                 server_step_size: float,
                 local_iteration_number: int,
                 batch_size: int,
                 mirror_map_p: float,
                 *args,
                 **kwargs
                 ):
        super().__init__(
            dec_opt_problem=dec_opt_problem,
        )
        assert client_step_size > 0 and server_step_size > 0
        assert local_iteration_number > 0 and batch_size > 0
        assert mirror_map_p >= 2.

        self.gradient_tracking_flag = gradient_tracking_flag
        self.client_step_size = client_step_size
        self.server_step_size = server_step_size
        self.local_iteration_number = local_iteration_number
        self.batch_size = batch_size
        self.mirror_map_p = mirror_map_p
        self.mirror_map_q = mirror_map_p / (mirror_map_p - 1.)

        self.h = lambda w: (1. / (2. * (self.mirror_map_q - 1.))) * (Lp_norm(w, self.mirror_map_q) ** 2.)
        self.h_conjugate = lambda z: (1. / (2. * (self.mirror_map_p - 1.))) * (Lp_norm(z, self.mirror_map_p) ** 2.)
        self.mirror_map = lambda w: (1. / (self.mirror_map_q - 1.)) * \
                                    (Lp_norm(w, self.mirror_map_q) ** (2. - self.mirror_map_q)) * \
                                    (jnp.abs(w) ** (self.mirror_map_q - 1.)) * (jnp.sign(w))
        # Numerical stable version.
        def inverse_mirror_map(z: Array) -> Array:
            z_p_norm = Lp_norm(z, self.mirror_map_p)
            normalized_z = replace_nan_with_zero(z / z_p_norm)
            return (1. / (self.mirror_map_p - 1.)) * z_p_norm * (jnp.abs(normalized_z) ** (self.mirror_map_p - 1.)) \
                * jnp.sign(z)
        self.inverse_mirror_map = jit(inverse_mirror_map)

        self.dec_inverse_mirror_map = jit(vmap(self.inverse_mirror_map, in_axes=0, out_axes=0))

        self.dec_w = jnp.repeat(dec_opt_problem.w_0[None, :], repeats=self.M, axis=0)
        self.dec_z = replace_nan_with_zero(jnp.repeat(self.mirror_map(dec_opt_problem.w_0)[None, :], repeats=self.M, axis=0))

        self.dec_c = jnp.zeros_like(self.dec_w)
        if self.gradient_tracking_flag:
            dec_batch = self.sample_oracle.sample_with_seed(batch_size, INIT_C_SEED)
            dec_batch_avg_grad = self.dec_opt_problem.dec_batch_avg_grad_F(dec_w=self.dec_w, dec_batch=dec_batch)
            self.dec_c = self.U @ dec_batch_avg_grad - dec_batch_avg_grad
        return

    def client_local_update(self):
        old_dec_z = jnp.copy(self.dec_z)
        for k in range(self.local_iteration_number):
            # Here, nan is exactly 0.
            current_dec_w = replace_nan_with_zero(self.dec_inverse_mirror_map(self.dec_z))
            dec_batch = self.sample_oracle.sample(self.batch_size)
            dec_batch_avg_grad = self.dec_opt_problem.dec_batch_avg_grad_F(dec_w=current_dec_w, dec_batch=dec_batch)
            if self.gradient_tracking_flag:
                dec_batch_avg_grad += self.dec_c
            self.dec_z -= self.client_step_size * dec_batch_avg_grad
        if self.gradient_tracking_flag:
            dec_delta = (1. / (self.local_iteration_number * self.client_step_size)) * (self.dec_z - old_dec_z)
            self.dec_c += (dec_delta - self.U @ dec_delta)
        lr = self.server_step_size
        self.dec_z = self.U @ (old_dec_z * (1. - lr) + self.dec_z * lr)
        self.dec_w = self.dec_inverse_mirror_map(self.dec_z)
        return


class DecFastFedDA(DecOptMethod):
    def __init__(self,
                 dec_opt_problem: DecentralizedOptProblem,
                 mu: float,
                 L: float,
                 threshold: float,
                 local_iteration_number: int,
                 batch_size: int,
                 *args,
                 **kwargs
                 ):
        super().__init__(
            dec_opt_problem=dec_opt_problem,
        )
        assert local_iteration_number > 0 and batch_size > 0
        assert mu > 0. and L >= mu and threshold > 0.

        self.local_iteration_number = local_iteration_number
        self.batch_size = batch_size

        self.mu = mu
        self.L = mu
        self.threshold = threshold

        self.dec_w = jnp.repeat(dec_opt_problem.w_0[None, :], repeats=self.M, axis=0)
        self.norm_dec_z = jnp.zeros_like(self.dec_w)
        self.norm_dec_c = jnp.zeros_like(self.dec_w)

        self.cum_step_size = 0.  # A_t
        self.current_step = 0
        return

    def client_local_update(self):
        for k in range(self.local_iteration_number):
            alpha = (self.current_step + 1)
            new_cum_step_size = self.cum_step_size + alpha
            dec_batch = self.sample_oracle.sample(self.batch_size)
            dec_batch_avg_grad = self.dec_opt_problem.dec_batch_avg_grad_F(dec_w=self.dec_w, dec_batch=dec_batch)

            self.norm_dec_z = (self.cum_step_size / new_cum_step_size) * self.norm_dec_z - (alpha / new_cum_step_size) * dec_batch_avg_grad
            if k == self.local_iteration_number - 1:
                self.norm_dec_z = self.U @ self.norm_dec_z
                self.norm_dec_c = self.U @ self.norm_dec_c
            # Soft-thresholding
            tmp = self.norm_dec_z + self.mu * 0.5 * self.norm_dec_c
            tmp_abs = jnp.abs(tmp)
            tmp_sgn = jnp.sign(tmp)
            self.dec_w = (new_cum_step_size / (self.mu * 0.5 * self.cum_step_size + alpha * self.L)) * \
                         (tmp_abs - self.threshold) * (tmp_abs > self.threshold) * tmp_sgn

            new_alpha = (self.current_step + 2)
            self.norm_dec_c = (self.cum_step_size / new_cum_step_size) * self.norm_dec_c + (new_alpha / new_cum_step_size) * self.dec_w
            self.current_step += 1
            self.cum_step_size += alpha
        return


class DecConFedDA(DecOptMethod):
    def __init__(self,
                 dec_opt_problem: DecentralizedOptProblem,
                 global_round_number: int,
                 local_iteration_number: int,
                 mu: float,
                 L: float,
                 threshold: float,  # lambda
                 batch_size: int,
                 init_Q: float,
                 *args,
                 **kwargs
                 ):
        super().__init__(
            dec_opt_problem=dec_opt_problem,
        )
        assert local_iteration_number > 0 and global_round_number > 0 and batch_size > 0
        assert init_Q > 0 and threshold > 0

        self.local_iteration_number = local_iteration_number
        self.global_round_num = global_round_number
        self.current_step = 0
        self.current_global_round = 0
        self.batch_size = batch_size

        self.mu = mu
        self.L = L
        self.threshold = threshold

        self.ref_dec_w = jnp.repeat(dec_opt_problem.w_0[None, :], repeats=self.M, axis=0)
        self.dec_w = jnp.copy(self.ref_dec_w)
        self.norm_dec_z = jnp.zeros_like(self.dec_w)
        self.norm_dec_c = jnp.copy(self.dec_w)

        self.cum_step_size = 0.

        self.Q = init_Q
        return

    def newstep_reset(self, new_Q: float, new_local_iteration_number: int, new_global_round_number: int,
                      new_threshold: float, new_ref_dec_w: Array = None, *args, **kwargs):
        assert new_local_iteration_number > 0 and new_global_round_number > 0
        assert new_Q > 0 and new_threshold > 0
        if new_ref_dec_w is None:
            self.ref_dec_w = jnp.copy(self.dec_w)
        else:
            self.ref_dec_w = jnp.copy(new_ref_dec_w)
        self.dec_w = jnp.copy(self.ref_dec_w)
        self.norm_dec_z = jnp.zeros_like(self.dec_w)
        self.norm_dec_c = jnp.copy(self.dec_w)
        self.current_global_round = 0
        self.current_step = 0
        self.cum_step_size = 0

        self.Q = new_Q
        self.local_iteration_number = new_local_iteration_number
        self.global_round_num = new_global_round_number
        self.threshold = new_threshold
        return

    def client_local_update(self):
        if self.current_global_round >= self.global_round_num:
            raise RuntimeError('Maximum global rounds exceeded.')
        for k in range(self.local_iteration_number):
            alpha = (self.current_step + 1)
            gamma = self.L * alpha
            new_cum_step_size = self.cum_step_size + alpha
            dec_batch = self.sample_oracle.sample(self.batch_size)
            dec_batch_avg_grad = self.dec_opt_problem.dec_batch_avg_grad_F(dec_w=self.dec_w, dec_batch=dec_batch)

            self.norm_dec_z = (self.cum_step_size / new_cum_step_size) * self.norm_dec_z - (alpha / new_cum_step_size) * dec_batch_avg_grad
            if k == self.local_iteration_number - 1:
                self.norm_dec_z = self.U @ self.norm_dec_z
                self.norm_dec_c = self.U @ self.norm_dec_c
            # Soft-thresholding
            tmp = self.norm_dec_z + self.mu * 0.5 * self.norm_dec_c + (gamma / new_cum_step_size) * self.ref_dec_w
            tmp_abs = jnp.abs(tmp)
            tmp_sgn = jnp.sign(tmp)
            uncons_dec_w = (new_cum_step_size / (self.mu * 0.5 * self.cum_step_size + alpha * self.L)) * \
                           (tmp_abs - self.threshold) * (tmp_abs > self.threshold) * tmp_sgn
            self.dec_w = project_onto_l1_ball_center_w(uncons_dec_w, self.ref_dec_w, self.Q)

            new_alpha = (self.current_step + 2)
            self.norm_dec_c = (self.cum_step_size / new_cum_step_size) * self.norm_dec_c + (new_alpha / new_cum_step_size) * self.dec_w
            self.current_step += 1
            self.cum_step_size += alpha
        self.current_global_round += 1
        return


class DecFedMiD(DecOptMethod):
    def __init__(self,
                 dec_opt_problem: DecentralizedOptProblem,
                 client_step_size: float,
                 server_step_size: float,
                 local_iteration_number: int,
                 batch_size: int,
                 threshold: float,
                 *args,
                 **kwargs
                 ):
        super().__init__(
            dec_opt_problem=dec_opt_problem,
        )
        assert client_step_size > 0 and server_step_size > 0
        assert local_iteration_number > 0 and batch_size > 0
        assert threshold > 0.

        self.client_step_size = client_step_size
        self.server_step_size = server_step_size
        self.local_iteration_number = local_iteration_number
        self.batch_size = batch_size
        self.threshold = threshold

        self.dec_w = jnp.repeat(dec_opt_problem.w_0[None, :], repeats=self.M, axis=0)
        return

    def client_local_update(self):
        old_dec_w = jnp.copy(self.dec_w)
        for k in range(self.local_iteration_number):
            dec_batch = self.sample_oracle.sample(self.batch_size)
            dec_batch_avg_grad = self.dec_opt_problem.dec_batch_avg_grad_F(dec_w=self.dec_w, dec_batch=dec_batch)
            # Soft-thresholding
            tmp = self.dec_w - self.client_step_size * dec_batch_avg_grad
            tmp_abs = jnp.abs(tmp)
            tmp_sgn = jnp.sign(tmp)
            threshold = self.threshold * self.client_step_size
            self.dec_w = (tmp_abs - threshold) * (tmp_abs > threshold) * tmp_sgn
        lr = self.server_step_size
        delta = self.dec_w - old_dec_w
        tmp = self.U @ (old_dec_w + lr * delta)
        tmp_abs = jnp.abs(tmp)
        tmp_sgn = jnp.sign(tmp)
        threshold = self.threshold * self.client_step_size * self.local_iteration_number * lr
        self.dec_w = (tmp_abs - threshold) * (tmp_abs > threshold) * tmp_sgn
        return


class DecReFedDAGT(DecOptMethod):
    def __init__(self,
                 dec_opt_problem: DecentralizedOptProblem,
                 client_step_size: float,
                 server_step_size: float,
                 global_round_number: int,
                 local_iteration_number: int,
                 batch_size: int,
                 sparsity: int,
                 mirror_map_p: float,
                 init_Q: float,
                 *args,
                 **kwargs
                 ):
        super().__init__(
            dec_opt_problem=dec_opt_problem,
        )
        assert client_step_size > 0 and server_step_size > 0
        assert local_iteration_number > 0 and global_round_number > 0 and batch_size > 0
        assert mirror_map_p >= 2 and init_Q > 0

        self.client_step_size = client_step_size
        self.server_step_size = server_step_size
        self.local_iteration_number = local_iteration_number
        self.global_round_num = global_round_number
        self.current_global_round = 0
        self.batch_size = batch_size
        self.sparsity = sparsity

        self.mirror_map_p = mirror_map_p
        self.mirror_map_q = mirror_map_p / (mirror_map_p - 1.)

        # Numerical stable version.
        # Note that we do not add w_0 here.
        def proximal_operator_q(z: Array, Q) -> Array:
            z_p_norm = Lp_norm(z, self.mirror_map_p)
            normalized_z = replace_nan_with_zero(z / z_p_norm)
            inv_mir_map = (1. / (self.mirror_map_p - 1.)) * z_p_norm * \
                          (jnp.abs(normalized_z) ** (self.mirror_map_p - 1.)) * jnp.sign(z)
            lagrange_multiplier = jnp.maximum((z_p_norm / ((self.mirror_map_p - 1.) * jnp.sqrt(Q))) - 1., 0.)
            coeff = 1. / (1. + lagrange_multiplier)
            return coeff * inv_mir_map

        self.proximal_operator_q = jit(proximal_operator_q)
        self.dec_proximal_operator_q = jit(vmap(self.proximal_operator_q, in_axes=(0, None), out_axes=0))

        self.ref_dec_w = jnp.repeat(dec_opt_problem.w_0[None, :], repeats=self.M, axis=0)
        self.dec_w = jnp.copy(self.ref_dec_w)
        self.dec_z = jnp.zeros_like(self.dec_w)

        self.Q = init_Q

        self.dec_c = jnp.zeros_like(self.dec_w)
        return

    def newstep_reset(self, new_client_step_size: float, new_server_step_size: float, new_Q: float,
                      new_local_iteration_number: int, new_global_round_number: int, new_ref_dec_w: Array = None,
                      *args, **kwargs):
        assert new_client_step_size > 0 and new_server_step_size > 0
        assert new_local_iteration_number > 0 and new_global_round_number > 0
        assert new_Q > 0
        if new_ref_dec_w is None:
            new_ref_w = sparsify(jnp.average(self.dec_w, axis=0), sparsity=self.sparsity)
            self.ref_dec_w = jnp.repeat(new_ref_w[None, :], repeats=self.M, axis=0)
        else:
            new_ref_w = sparsify(jnp.average(new_ref_dec_w, axis=0), sparsity=self.sparsity)
            self.ref_dec_w = jnp.repeat(new_ref_w[None, :], repeats=self.M, axis=0)
        self.dec_w = jnp.copy(self.ref_dec_w)
        self.dec_z = jnp.zeros_like(self.dec_w)
        self.dec_c = jnp.zeros_like(self.dec_w)
        self.current_global_round = 0
        self.client_step_size = new_client_step_size
        self.server_step_size = new_server_step_size
        self.Q = new_Q
        self.local_iteration_number = new_local_iteration_number
        self.global_round_num = new_global_round_number
        return

    def client_local_update(self):
        if self.current_global_round >= self.global_round_num:
            raise RuntimeError('Maximum global rounds exceeded.')
        old_dec_z = jnp.copy(self.dec_z)
        for k in range(self.local_iteration_number):
            # Here, nan is exactly 0.
            self.dec_w = self.ref_dec_w + self.dec_proximal_operator_q(self.dec_z, self.Q)
            dec_batch = self.sample_oracle.sample(self.batch_size)
            dec_batch_avg_grad = self.dec_opt_problem.dec_batch_avg_grad_F(dec_w=self.dec_w, dec_batch=dec_batch)
            dec_batch_avg_grad += self.dec_c
            self.dec_z -= self.client_step_size * dec_batch_avg_grad
        dec_delta = (1. / (self.local_iteration_number * self.client_step_size)) * (self.dec_z - old_dec_z)
        self.dec_c += (dec_delta - self.U @ dec_delta)
        lr = self.server_step_size
        self.dec_z = self.U @ (old_dec_z * (1. - lr) + self.dec_z * lr)
        self.dec_w = self.ref_dec_w + self.dec_proximal_operator_q(self.dec_z, self.Q)
        self.current_global_round += 1
        return


method_register = {
    'Sparse_Central_SGD': SparseCentraSGD,
    'Central_SGD': CentraSGD,
    'SCAFFOLD': DecSCAFFOLD,
    'FedDA_GT': DecFedDAGT,
    'ReFedDA_GT': DecReFedDAGT,
    'Fast_FedDA': DecFastFedDA,
    'Con_FedDA': DecConFedDA,
    'FedMiD': DecFedMiD,
}

