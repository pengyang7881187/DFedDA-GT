import os
import jax
import yaml
import argparse
import numpy as np
import jax.numpy as jnp

from MyTyping import *

from utilities import add_scalar_dict, get_format_time, add_suffix_to_keys, \
    dict_add, dict_divide, dict_max, dict_min, dict_log
from DecOptProblem import DecentralizedOptProblem
from DecOptMethod import method_register
from DecOptTestFunction import F_register
from DecSample import FlowHomoGaussianSparseGLMSampleOracle, \
    FlowHomoGaussianSparseLinearBinaryClassificationSampleOracle
from DecGossipMatrix import gossip_register
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Hyperparameters of the experiment of decentralized optimization.')

parser.add_argument(
    '--now_time', type=str,
    help='Now time.'
)
parser.add_argument(
    '--gpu', default=1, type=int,
    help='GPU.'
)

# Basic arguments.
parser.add_argument(
    '--problem', default='linear_regression', type=str,
    choices=('linear_regression', 'logistic_regression'), help='Optimization problem type.'
)
parser.add_argument(
    '--method', default='Sparse_Central_SGD', type=str,
    choices=tuple(method_register.keys()), help='Optimization method.'
)
parser.add_argument(
    '--gradient_tracking', action='store_true',
    help='Whether to use gradient tracking in SCAFFOLD and FedDA-GT.'
)
parser.add_argument(
    '--seed', default=5, type=int,
    help='Random seed.'
)
# Decentralized optimization problem arguments.
parser.add_argument(
    '-d', '--num_parameters', default=1024, type=int,
    help='Number of parameters.'
)
parser.add_argument(
    '--param_scale', default=5., type=float,
    help='Scale of parameters.'
)
parser.add_argument(
    '--param_mode', default='random', type=str,
    choices=('random', 'all_one'), help='Mode of true parameters for devices.'
)
parser.add_argument(
    '-M', '--num_devices', default=16, type=int,
    help='Number of devices.'
)
parser.add_argument(
    '-s', '--sparsity', default=16, type=int,
    help='Sparsity, i.e. number of non-zero elements of true parameters.'
)
parser.add_argument(
    '--param_threshold', default=1., type=float,
    help='Minimum of abs of non-zero parameters.'
)
parser.add_argument(
    '-U', '--gossip_matrix_mode', default='chain', type=str,
    choices=tuple(gossip_register.keys()),
    help='Mode of gossip matrix U.'
)
parser.add_argument(
    '--mirror_map_p',
    default=12.,
    type=float,
    help='Mirror map p.'
)

# Data.
parser.add_argument(
    '--covariate_var', default=1., type=float,
    help='Variance of covariates.'
)
parser.add_argument(
    '--variate_var', default=1., type=float,
    help='Variance of variates.'
)

# Training.
parser.add_argument(
    '--random_init', action='store_true',
    help='Whether to use random initialization, use zero initialization by default.'
)
parser.add_argument(
    '--client_step_size', default=.1, type=float,
    help='Step size of each client.'
)
parser.add_argument(
    '--server_step_size', default=.1, type=float,
    help='Step size of the server or aggregation operation.'
)
parser.add_argument(
    '-B', '--batch_size', default=10, type=int,
    help='Batch size.'
)
parser.add_argument(
    '-K', '--num_local_iters', default=10, type=int,
    help='Number of local iterations at each client.'
)
parser.add_argument(
    '-R', '--num_global_rounds', default=100000, type=int,
    help='Number of global iterations.'
)
parser.add_argument(
    '--init_Q', default=100., type=float,
    help='Initial Q for ReFedDA-GT and ConFedDA.'
)
parser.add_argument(
    "--Q_decay_interval", type=int,
    default=2000,
    help="Number of rounds for Q decay."
)
parser.add_argument(
    '--ReMovFlag', action='store_true',
    help='Whether to use moving average in MReFedDA_GT.'
)
parser.add_argument(
    '--moving_avg_num', default=100, type=int,
    help='Moving average number.'
)
parser.add_argument(
    '--mu', default=1., type=float,
    help='mu for FastFedDA and ConFedDA.'
)
parser.add_argument(
    '--L', default=1., type=float,
    help='L for FastFedDA and ConFedDA.'
)
parser.add_argument(
    '--l1_reg', default=.1, type=float,
    help='lambda for FastFedDA, ConFedDA and FedMiD.'
)
parser.add_argument(
    "--lr_decay_no_decrease_tolerance", type=int,
    default=30,
    help="Number of evaluations to wait for decreasing of loss."
)
parser.add_argument(
    "--client_lr_decay_coeff", type=float,
    default=0.5,
    help="Client lr decay coefficient."
)
parser.add_argument(
    "--server_lr_decay_coeff", type=float,
    default=0.5,
    help="Server lr decay coefficient."
)

# Evaluation.
parser.add_argument(
    '--full_data_size', default=10, type=int,
    help='Full data size for evaluation, not needed in linear model.'
)
parser.add_argument(
    "--evaluation_interval", type=int, default=10,
    help="Evaluate with every evaluation_interval global iterations."
)
parser.add_argument(
    '--dec_evaluate', action='store_true',
    help='Whether to decentralized evaluate.'
)

args = parser.parse_args()

if args.gpu >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

problem = args.problem
method = args.method
gradient_tracking_flag = args.gradient_tracking
seed = args.seed
dim_param = args.num_parameters
sparsity = args.sparsity
M = args.num_devices
U_mode = args.gossip_matrix_mode
mirror_map_p = args.mirror_map_p
param_threshold = args.param_threshold
param_mode = args.param_mode
param_scale = args.param_scale

covariate_var = args.covariate_var
variate_var = args.variate_var

mirror_map_q = mirror_map_p / (mirror_map_p - 1.)
random_init = args.random_init
client_step_size = args.client_step_size
server_step_size = args.server_step_size
batch_size = args.batch_size
K = args.num_local_iters
R = args.num_global_rounds
lr_decay_no_decrease_tolerance = args.lr_decay_no_decrease_tolerance
client_lr_decay_coeff = args.client_lr_decay_coeff
server_lr_decay_coeff = args.server_lr_decay_coeff

init_Q = args.init_Q
Q_decay_interval = args.Q_decay_interval
moving_avg_flag = args.ReMovFlag
moving_avg_num = args.moving_avg_num

mu = args.mu
L = args.L
l1_reg = args.l1_reg

full_data_size = args.full_data_size
evaluation_interval = args.evaluation_interval
dec_evaluate_flag = args.dec_evaluate

F = F_register[f'{problem}_F']
U = gossip_register[U_mode](M=M)

important_parameter_dict = {
    'problem': problem,
    'method': method,
    'gradient_tracking': int(gradient_tracking_flag),
    'seed': seed,
    'dim_param': dim_param,
    'sparsity': sparsity,
    'M': M,
    'U_mode': U_mode,
    'param_threshold': param_threshold,
    'param_scale': param_scale,
    'param_mode': param_mode,
    'covariate_var': covariate_var,
    'variate_var': variate_var,
    'mirror_map_p': mirror_map_p,
    'random_init': int(random_init),
    'batch_size': batch_size,
    'full_data_size': full_data_size,
    'client_lr': client_step_size,
    'server_lr': server_step_size,
    'local_iter_num': K,
    'global_iter_round': R,
    'client_lr_decay_coeff': client_lr_decay_coeff,
    'server_lr_decay_coeff': server_lr_decay_coeff,
    'lr_decay_no_decrease_tolerance': lr_decay_no_decrease_tolerance,
    'moving_avg_flag': int(moving_avg_flag),
    'moving_avg_number': moving_avg_num,
    'init_Q': init_Q,
    'Q_decay_interval': Q_decay_interval,
    'mu': mu,
    'L': L,
    'l1_reg': l1_reg,
}

if args.now_time is None:
    now_time = get_format_time()
else:
    now_time = args.now_time
writer_dir = f'./log_flow_{problem}/{now_time}'
log_dir = f'{writer_dir}/{method}'
writer = SummaryWriter(log_dir=log_dir)
add_scalar_dict(input_dict=important_parameter_dict, writer=writer, global_step=0)
yaml_save_path = f'{log_dir}/param_dict.yml'
with open(yaml_save_path, 'w') as f:
    yaml.dump(important_parameter_dict, f)

jax_key = jax.random.PRNGKey(seed=seed)
if not random_init:
    # Zero initialization.
    w_0 = jnp.zeros((dim_param,), dtype=float)
else:
    # Random initialization.
    jax_key, subkey = jax.random.split(jax_key)
    w_0 = param_scale * jax.random.normal(subkey, shape=(dim_param,))

if problem == 'linear_regression':
    sample_oracle = FlowHomoGaussianSparseGLMSampleOracle(
        jax_key=jax_key,
        nonzero_param_threshold=param_threshold,
        full_data_default_size=full_data_size,
        sparsity=sparsity,
        dim_param=dim_param,
        M=M,
        x_nonzero_mean=jnp.zeros((sparsity,)),
        x_nonzero_cov=covariate_var * jnp.eye(sparsity),
        x_zero_mean=jnp.zeros((dim_param - sparsity - 1,)),
        x_zero_cov=covariate_var * jnp.eye(dim_param - sparsity - 1),
        nonzero_link_func=lambda x: x,
        zero_link_func=lambda x: x,
        error_var=variate_var,
        param_scale=param_scale,
        param_mode=param_mode,
        bias_flag=True,
    )
    dec_param = sample_oracle.flow_data_sampler.dec_param
    w_true = jnp.average(dec_param, axis=0)

    eval_expectation_flag = False

    # Assume x is mean 0 and have bias term.
    def theoretical_obj(w: Array) -> float:
        return float(0.5 * (variate_var +
                            covariate_var * jnp.average(
                                jnp.square(jnp.linalg.norm(dec_param[:, 1:] - w[None, 1:], axis=1))
                            )
                            + jnp.average(jnp.square(dec_param[:, 0] - w[0]))
                            )
                     )


    def theoretical_grad_norm(w: Array) -> float:
        coeff_vec = covariate_var * jnp.ones((dim_param,))
        coeff_vec = coeff_vec.at[0].set(1.)
        return float(jnp.linalg.norm(coeff_vec * jnp.average(dec_param - w[None, :], axis=0)))
else:
    sample_oracle = FlowHomoGaussianSparseLinearBinaryClassificationSampleOracle(
        jax_key=jax_key,
        nonzero_param_threshold=param_threshold,
        full_data_default_size=full_data_size,
        sparsity=sparsity,
        dim_param=dim_param,
        M=M,
        x_mean=jnp.zeros((dim_param,)),
        x_cov=covariate_var * jnp.eye(dim_param),
        func=jax.nn.sigmoid,
        param_scale=param_scale,
        param_mode=param_mode,
        bias_flag=False,
    )
    # TODO: For evaluation, run sparse central SGD to get an optimal solution first.
    param_dir = './log_flow_logistic_regression/yyyymmdd_hhmmss/Sparse_Central_SGD'
    try:
        w_true = jnp.load(f'{param_dir}/moving_100_avg_w.npy')
    except:
        w_true = jnp.zeros((dim_param,))
    eval_expectation_flag = True

dec_opt_problem = DecentralizedOptProblem(
    F=F,
    sample_oracle=sample_oracle,
    name=problem,
    dim_param=dim_param,
    U=U,
    w_0=jnp.copy(w_0),
)
dec_opt_problem.w_star = w_true
if problem == 'linear_regression':
    dec_opt_problem.obj_star = theoretical_obj(w_true)
    # Synchronize validation set.
    empirical_obj_star, empirical_grad_norm_star = dec_opt_problem.evaluate_obj_and_grad_norm(w_true, False)
else:
    dec_opt_problem.set_optimal_solution(w_star=w_true, expectation_flag=eval_expectation_flag)


current_Q = float(init_Q)
current_l1_reg = l1_reg
writer.add_scalar(tag='Q',
                  scalar_value=current_Q,
                  global_step=0)
writer.add_scalar(tag='l1_reg',
                  scalar_value=current_l1_reg,
                  global_step=0)

dec_optimizer = method_register[method](
    dec_opt_problem=dec_opt_problem,
    client_step_size=client_step_size,
    server_step_size=server_step_size,
    gradient_tracking_flag=gradient_tracking_flag,
    sparsity=sparsity,
    local_iteration_number=K,
    global_round_number=Q_decay_interval,
    batch_size=batch_size,
    mu=mu,
    L=L,
    init_Q=current_Q,
    threshold=current_l1_reg,
    mirror_map_p=mirror_map_p,
)
best_obj = np.inf
Q_best_obj = np.inf
no_decrease_cnt = 0
Q_decay_cnt = 0

moving_avg_dec_w = jnp.empty_like(dec_optimizer.dec_w)
moving_dec_w_window = [jnp.copy(dec_optimizer.dec_w), ]
avg_dec_w = jnp.copy(dec_optimizer.dec_w)

for r in range(R):
    if r % evaluation_interval == 0:
        print('=' * 100)
        print(f'{method} communication round {r}')
        if method in ('Sparse_Central_SGD', 'Central_SGD'):
            dec_evaluate_flag = False
        if dec_evaluate_flag:
            sum_dict: dict = None
            max_dict: dict = None
            min_dict: dict = None
            for m in range(M):
                device_m_result = dec_opt_problem.summary_result(
                    dec_optimizer.dec_w[m], sparsity=sparsity, mirror_map_q=mirror_map_q,
                    expectation_flag=eval_expectation_flag
                )
                if problem == 'linear_regression':
                    device_m_result['obj'] = theoretical_obj(dec_optimizer.dec_w[m])
                    device_m_result['obj_gap'] = device_m_result['obj'] - float(dec_opt_problem.obj_star)
                    device_m_result['grad_norm'] = theoretical_grad_norm(dec_optimizer.dec_w[m])
                sum_dict = dict_add(device_m_result, sum_dict)
                max_dict = dict_max(device_m_result, max_dict)
                min_dict = dict_min(device_m_result, min_dict)
            result = dict_divide(sum_dict, M)
            add_scalar_dict(dict_log(add_suffix_to_keys(max_dict, 'max')), writer, r)
            add_scalar_dict(dict_log(add_suffix_to_keys(min_dict, 'min')), writer, r)
        else:
            result = dec_opt_problem.summary_result(dec_optimizer.dec_w[0], sparsity=sparsity,
                                                    mirror_map_q=mirror_map_q, expectation_flag=eval_expectation_flag)
            if problem == 'linear_regression':
                result['obj'] = theoretical_obj(dec_optimizer.dec_w[0])
                result['obj_gap'] = result['obj'] - float(dec_opt_problem.obj_star)
                result['grad_norm'] = theoretical_grad_norm(dec_optimizer.dec_w[0])
        result = dict_log(result)
        add_scalar_dict(result, writer, r)
        # nan or inf detect and lr decay.
        if not np.isfinite(result['obj']):
            break
        elif result['obj'] < Q_best_obj:
            Q_best_obj = result['obj']
            no_decrease_cnt = 0
            if Q_best_obj < best_obj:
                best_obj = Q_best_obj
                jnp.save(f'{log_dir}/best_dec_w.npy', dec_optimizer.dec_w)
        else:
            no_decrease_cnt += 1
            if no_decrease_cnt >= lr_decay_no_decrease_tolerance:
                client_step_size *= client_lr_decay_coeff
                dec_optimizer.client_step_size *= client_lr_decay_coeff
                writer.add_scalar(tag='client_step_size',
                                  scalar_value=dec_optimizer.client_step_size,
                                  global_step=r)
                if hasattr(dec_optimizer, 'server_step_size'):
                    server_step_size *= server_lr_decay_coeff
                    dec_optimizer.server_step_size *= server_lr_decay_coeff
                    writer.add_scalar(tag='server_step_size',
                                      scalar_value=dec_optimizer.server_step_size,
                                      global_step=r)
                if client_step_size < 1e-12:
                    break
                no_decrease_cnt = 0
    dec_optimizer.client_local_update()
    # Save last iterate parameter.
    jnp.save(f'{log_dir}/last_dec_w.npy', dec_optimizer.dec_w)
    # Update and save avg w and moving avg w.
    avg_dec_w = (((r + 1.) * avg_dec_w) + dec_optimizer.dec_w) / (r + 2.)
    jnp.save(f'{log_dir}/avg_dec_w.npy', avg_dec_w)

    moving_dec_w_window.append(jnp.copy(dec_optimizer.dec_w))
    if r <= moving_avg_num - 2:
        moving_avg_dec_w = jnp.copy(avg_dec_w)
    else:
        moving_avg_dec_w = moving_avg_dec_w + ((dec_optimizer.dec_w - moving_dec_w_window[0]) / moving_avg_num)
        moving_dec_w_window.pop(0)
    jnp.save(f'{log_dir}/moving_{moving_avg_num}_avg_dec_w', moving_avg_dec_w)
    Q_decay_cnt += 1
    if method in ('ReFedDA_GT', 'Con_FedDA'):
        if Q_decay_cnt == Q_decay_interval:
            Q_best_obj = np.inf
            Q_decay_cnt = 0
            current_Q /= 2.
            current_l1_reg /= 2.

            writer.add_scalar(tag='Q',
                              scalar_value=current_Q,
                              global_step=r)
            writer.add_scalar(tag='l1_reg',
                              scalar_value=current_l1_reg,
                              global_step=r)
            new_ref_dec_w = jnp.copy(moving_avg_dec_w)
            if not args.ReMovFlag:
                new_ref_dec_w = None
            dec_optimizer.newstep_reset(new_client_step_size=client_step_size,
                                        new_server_step_size=server_step_size,
                                        new_Q=current_Q,
                                        new_threshold=current_l1_reg,
                                        new_local_iteration_number=K,
                                        new_global_round_number=Q_decay_interval,
                                        new_ref_dec_w=new_ref_dec_w,
                                        )
writer.close()
