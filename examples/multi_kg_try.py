"""
Pick 4 functions. See how the KG evolves and try out sampling rules.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np
import os, sys
import time

from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cppTensorProductDomain
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import PosteriorMeanMCMC
from moe.optimal_learning.python.cpp_wrappers.log_likelihood_mcmc import \
    GaussianProcessLogLikelihoodMCMC as cppGaussianProcessLogLikelihoodMCMC
from moe.optimal_learning.python.cpp_wrappers.optimization import \
    GradientDescentParameters as cppGradientDescentParameters
from moe.optimal_learning.python.cpp_wrappers.optimization import \
    GradientDescentOptimizer as cppGradientDescentOptimizer
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient import posterior_mean_optimization, PosteriorMean

from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.repeated_domain import RepeatedDomain
from moe.optimal_learning.python.default_priors import DefaultPrior

from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.optimization import \
    GradientDescentParameters as pyGradientDescentParameters
from moe.optimal_learning.python.python_version.optimization import \
    GradientDescentOptimizer as pyGradientDescentOptimizer
from moe.optimal_learning.python.python_version.optimization import multistart_optimize as multistart_optimize

from examples import bayesian_optimization
from examples import synthetic_functions

# arguments for calling this script:
# python main.py [obj_func_name] [method_name] [num_to_sample] [job_id]
# example: python main.py Branin KG 4 1
# you can define your own obj_function and then just change the objective_func object below, and run this script.

# argv = sys.argv[1:]
# obj_func_name = str(argv[0])
method = "KG"
num_to_sample = 1
# job_id = int(argv[3])

# constants
# num_func_eval = 12
# num_iteration = int(old_div(num_func_eval, num_to_sample)) + 1

obj_func_dict = {'Branin': synthetic_functions.Branin(),
                 'Rosenbrock': synthetic_functions.Rosenbrock(),
                 'Hartmann3': synthetic_functions.Hartmann3(),
                 'Levy4': synthetic_functions.Levy4(),
                 'Hartmann6': synthetic_functions.Hartmann6(),
                 'Ackley': synthetic_functions.Ackley()}

objective_func_list = [obj_func_dict["Branin"], obj_func_dict["Rosenbrock"], obj_func_dict["Hartmann3"],
                       obj_func_dict["Hartmann6"]]
# objective_func = obj_func_dict[obj_func_name]
dim = [int(objective_func._dim) for objective_func in objective_func_list]
num_initial_points = [int(objective_func._num_init_pts) for objective_func in objective_func_list]

num_fidelity = [0, 0, 0, 0]
inner_search_domain = [0, 0, 0, 0]
cpp_search_domain = [0, 0, 0, 0]
cpp_inner_search_domain = [0, 0, 0, 0]
for i in range(4):
    objective_func = objective_func_list[i]
    num_fidelity[i] = objective_func._num_fidelity
    inner_search_domain[i] = pythonTensorProductDomain([ClosedInterval(objective_func._search_domain[i, 0], objective_func._search_domain[i, 1])
                                   for i in range(objective_func._search_domain.shape[0] - num_fidelity[i])])
    cpp_search_domain[i] = cppTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in objective_func._search_domain])
    cpp_inner_search_domain = cppTensorProductDomain([ClosedInterval(objective_func._search_domain[i, 0], objective_func._search_domain[i, 1])
                                for i in range(objective_func._search_domain.shape[0] - num_fidelity[i])])

init_pts = [0, 0, 0, 0]
derivatives = [0, 0, 0, 0]
observations = [0, 0, 0, 0]
init_pts_value = [0, 0, 0, 0]
true_value_init = [0, 0, 0, 0]
init_data = [0, 0, 0, 0]
for i in range(4):
    objective_func = objective_func_list[i]
    # get the initial data
    init_pts[i] = np.zeros((objective_func._num_init_pts, objective_func._dim))
    init_pts[i][:,
    :objective_func._dim - objective_func._num_fidelity] = inner_search_domain[
        i].generate_uniform_random_points_in_domain(
        objective_func._num_init_pts)
    for pt in init_pts[i]:
        pt[objective_func._dim - objective_func._num_fidelity:] = np.ones(objective_func._num_fidelity)

    # observe
    derivatives[i] = objective_func._observations
    observations[i] = [0] + [j + 1 for j in derivatives[i]]
    init_pts_value[i] = np.array([objective_func.evaluate(pt) for pt in init_pts[i]])  # [:, observations]
    true_value_init[i] = np.array([objective_func.evaluate_true(pt) for pt in init_pts[i]])  # [:, observations]

    init_data[i] = HistoricalData(dim=objective_func._dim, num_derivatives=0)
    init_data[i].append_sample_points([SamplePoint(pt, [init_pts_value[i][num, j] for j in observations[i]],
                                                   objective_func._sample_var) for num, pt in enumerate(init_pts[i])])

prior = [0, 0, 0, 0]
cpp_gp_loglikelihood = [0, 0, 0, 0]
for i in range(4):
    # initialize the model
    prior[i] = DefaultPrior(1 + dim[i] + len(observations[i]), len(observations[i]))

    # noisy = False means the underlying function being optimized is noise-free
    cpp_gp_loglikelihood[i] = cppGaussianProcessLogLikelihoodMCMC(historical_data=init_data[i],
                                                                  derivatives=derivatives[i],
                                                                  prior=prior[i],
                                                                  chain_length=1000,
                                                                  burnin_steps=2000,
                                                                  n_hypers=2 ** 4,
                                                                  noisy=False)
    cpp_gp_loglikelihood[i].train()

py_sgd_params_ps = pyGradientDescentParameters(max_num_steps=1000,
                                               max_num_restarts=3,
                                               num_steps_averaged=15,
                                               gamma=0.7,
                                               pre_mult=1.0,
                                               max_relative_change=0.02,
                                               tolerance=1.0e-10)

cpp_sgd_params_ps = cppGradientDescentParameters(num_multistarts=1,
                                                 max_num_steps=6,
                                                 max_num_restarts=1,
                                                 num_steps_averaged=3,
                                                 gamma=0.0,
                                                 pre_mult=1.0,
                                                 max_relative_change=0.1,
                                                 tolerance=1.0e-10)

cpp_sgd_params_kg = cppGradientDescentParameters(num_multistarts=200,
                                                 max_num_steps=50,
                                                 max_num_restarts=2,
                                                 num_steps_averaged=4,
                                                 gamma=0.7,
                                                 pre_mult=1.0,
                                                 max_relative_change=0.5,
                                                 tolerance=1.0e-10)

eval_pts = [0, 0, 0, 0]
test = [0, 0, 0, 0]
ps = [0, 0, 0, 0]
py_repeated_search_domain = [0, 0, 0, 0]
ps_mean_opt = [0, 0, 0, 0]
report_point = [0, 0, 0, 0]
for i in range(4):
    objective_func = objective_func_list[i]
    # minimum of the mean surface
    eval_pts[i] = inner_search_domain[i].generate_uniform_random_points_in_domain(int(1e3))
    eval_pts[i] = np.reshape(
        np.append(eval_pts[i], (cpp_gp_loglikelihood[i].get_historical_data_copy()).points_sampled[:,
                               :(cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity)]),
        (eval_pts[i].shape[0] + cpp_gp_loglikelihood[i]._num_sampled,
         cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity))

    test[i] = np.zeros(eval_pts[i].shape[0])
    ps[i] = PosteriorMeanMCMC(cpp_gp_loglikelihood[i].models, num_fidelity[i])
    for j, pt in enumerate(eval_pts[i]):
        ps[i].set_current_point(pt.reshape((1, cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity)))
        test[i] = -ps[i].compute_objective_function()
    report_point[i] = eval_pts[i][np.argmin(test[i])].reshape(
        (1, cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity))

    py_repeated_search_domain[i] = RepeatedDomain(num_repeats=1, domain=inner_search_domain[i])
    ps_mean_opt[i] = pyGradientDescentOptimizer(py_repeated_search_domain[i], ps[i], py_sgd_params_ps)
    report_point[i] = multistart_optimize(ps_mean_opt[i], report_point[i], num_multistarts=1)[0]
    report_point[i] = report_point[i].ravel()
    report_point[i] = np.concatenate((report_point[i], np.ones(objective_func._num_fidelity)))

current_best = [0, 0, 0, 0]
best_point = report_point
for i in range(4):
    current_best[i] = true_value_init[i][np.argmin(true_value_init[i][:, 0])][0]
    print("obj ", i, " best so far in the initial data {0}".format(current_best[i]))
    print("obj ", i, "report point value", objective_func_list[i].evaluate_true(report_point[i])[0])
capital_so_far = 0.

next_points = [0, 0, 0, 0]
voi = [0, 0, 0, 0]
for i in range(4):
    objective_func = objective_func_list[i]
    # KG
    time1 = time.time()
    discrete_pts_list = []

    discrete, _ = bayesian_optimization.gen_sample_from_qei_mcmc(cpp_gp_loglikelihood[i]._gaussian_process_mcmc,
                                                                 cpp_search_domain[i],
                                                                 cpp_sgd_params_kg, 10, num_mc=2 ** 10)
    for j, cpp_gp in enumerate(cpp_gp_loglikelihood[i].models):
        discrete_pts_optima = np.array(discrete)

        eval_pts = inner_search_domain[i].generate_uniform_random_points_in_domain(int(1e3))
        eval_pts = np.reshape(np.append(eval_pts,
                                        (cpp_gp.get_historical_data_copy()).points_sampled[:,
                                        :(cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity)]),
                              (eval_pts.shape[0] + cpp_gp.num_sampled, cpp_gp.dim - objective_func._num_fidelity))

        test = np.zeros(eval_pts.shape[0])
        ps_evaluator = PosteriorMean(cpp_gp, num_fidelity[i])
        for k, pt in enumerate(eval_pts):
            ps_evaluator.set_current_point(pt.reshape((1, cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity)))
            test[k] = -ps_evaluator.compute_objective_function()

        initial_point = eval_pts[np.argmin(test)]

        ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_inner_search_domain[i], ps_evaluator, cpp_sgd_params_ps)
        report_point = posterior_mean_optimization(ps_sgd_optimizer, initial_guess=initial_point, max_num_threads=4)

        ps_evaluator.set_current_point(
            report_point.reshape((1, cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity)))
        if -ps_evaluator.compute_objective_function() > np.min(test):
            report_point = initial_point

        discrete_pts_optima = np.reshape(np.append(discrete_pts_optima, report_point),
                                         (discrete_pts_optima.shape[0] + 1,
                                          cpp_gp.dim - objective_func._num_fidelity))
        discrete_pts_list.append(discrete_pts_optima)

    ps_evaluator = PosteriorMean(cpp_gp_loglikelihood[i].models[0], num_fidelity[i])
    ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_inner_search_domain[i], ps_evaluator, cpp_sgd_params_ps)
    # KG method
    next_points[i], voi[i] = bayesian_optimization.gen_sample_from_qkg_mcmc(
        cpp_gp_loglikelihood[i]._gaussian_process_mcmc,
        cpp_gp_loglikelihood[i].models,
        ps_sgd_optimizer, cpp_search_domain[i],
        num_fidelity[i], discrete_pts_list,
        cpp_sgd_params_kg, num_to_sample,
        num_mc=2 ** 7)
    print(method + " takes " + str((time.time() - time1)) + " seconds for objective", i)
    print(method + " suggests points: ", next_points[i], " with voi: ", voi[i])

while True:
    print(method + ", multiples, {0}th iteration".format(capital_so_far))

    print("Suggested points: ", next_points)
    print("Corresponding voi: ", voi)
    print("Current best: ", current_best)

    i = int(input("pick the next sample i = {1, 2, 3, 4} (or -1 to quit): "))
    if i == -1:
        break

    objective_func = objective_func_list[i]

    time1 = time.time()

    sampled_points = [SamplePoint(pt, objective_func.evaluate(pt)[observations[i]], objective_func._sample_var) for pt in
                      next_points[i]]

    print("evaluating takes " + str((time.time() - time1)) + " seconds")

    capital_so_far += len(sampled_points)
    print("evaluating takes capital " + str(capital_so_far) + " so far")

    # retrain the model
    time1 = time.time()

    cpp_gp_loglikelihood[i].add_sampled_points(sampled_points)
    cpp_gp_loglikelihood[i].train()

    print("retraining the model takes " + str((time.time() - time1)) + " seconds")
    time1 = time.time()

    # report the point
    eval_pts = inner_search_domain[i].generate_uniform_random_points_in_domain(int(1e4))
    eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood[i].get_historical_data_copy()).points_sampled[:,
                                              :(cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity)]),
                          (eval_pts.shape[0] + cpp_gp_loglikelihood[i]._num_sampled,
                           cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity))

    ps = PosteriorMeanMCMC(cpp_gp_loglikelihood[i].models, num_fidelity[i])
    test = np.zeros(eval_pts.shape[0])
    for j, pt in enumerate(eval_pts):
        ps.set_current_point(pt.reshape((1, cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity)))
        test[j] = -ps.compute_objective_function()
    initial_point = eval_pts[np.argmin(test)].reshape((1, cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity))

    py_repeated_search_domain = RepeatedDomain(num_repeats=1, domain=inner_search_domain[i])
    ps_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, ps, py_sgd_params_ps)
    report_point = multistart_optimize(ps_mean_opt, initial_point, num_multistarts=1)[0]

    ps.set_current_point(report_point.reshape((1, cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity)))
    if -ps.compute_objective_function() > np.min(test):
        report_point = initial_point

    report_point = report_point.ravel()
    report_point = np.concatenate((report_point, np.ones(objective_func._num_fidelity)))

    print()
    print("Optimization finished successfully!")
    print("The recommended point: ", end=' ')
    print(report_point)
    print("recommending the point takes " + str((time.time() - time1)) + " seconds")
    best_point[i] = report_point
    current_best[i] = objective_func.evaluate_true(report_point)[0]
    print(method + ", VOI {0}, best so far {1}".format(voi, current_best[i]))

    time1 = time.time()
    # KG
    discrete_pts_list = []

    discrete, _ = bayesian_optimization.gen_sample_from_qei_mcmc(cpp_gp_loglikelihood[i]._gaussian_process_mcmc,
                                                                 cpp_search_domain[i],
                                                                 cpp_sgd_params_kg, 10, num_mc=2 ** 10)
    for j, cpp_gp in enumerate(cpp_gp_loglikelihood[i].models):
        discrete_pts_optima = np.array(discrete)

        eval_pts = inner_search_domain[i].generate_uniform_random_points_in_domain(int(1e3))
        eval_pts = np.reshape(np.append(eval_pts,
                                        (cpp_gp.get_historical_data_copy()).points_sampled[:,
                                        :(cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity)]),
                              (eval_pts.shape[0] + cpp_gp.num_sampled, cpp_gp.dim - objective_func._num_fidelity))

        test = np.zeros(eval_pts.shape[0])
        ps_evaluator = PosteriorMean(cpp_gp, num_fidelity[i])
        for k, pt in enumerate(eval_pts):
            ps_evaluator.set_current_point(
                pt.reshape((1, cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity)))
            test[k] = -ps_evaluator.compute_objective_function()

        initial_point = eval_pts[np.argmin(test)]

        ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_inner_search_domain[i], ps_evaluator, cpp_sgd_params_ps)
        report_point = posterior_mean_optimization(ps_sgd_optimizer, initial_guess=initial_point, max_num_threads=4)

        ps_evaluator.set_current_point(
            report_point.reshape((1, cpp_gp_loglikelihood[i].dim - objective_func._num_fidelity)))
        if -ps_evaluator.compute_objective_function() > np.min(test):
            report_point = initial_point

        discrete_pts_optima = np.reshape(np.append(discrete_pts_optima, report_point),
                                         (discrete_pts_optima.shape[0] + 1,
                                          cpp_gp.dim - objective_func._num_fidelity))
        discrete_pts_list.append(discrete_pts_optima)

    ps_evaluator = PosteriorMean(cpp_gp_loglikelihood[i].models[0], num_fidelity[i])
    ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_inner_search_domain[i], ps_evaluator, cpp_sgd_params_ps)
    # KG method
    next_points[i], voi[i] = bayesian_optimization.gen_sample_from_qkg_mcmc(
        cpp_gp_loglikelihood[i]._gaussian_process_mcmc,
        cpp_gp_loglikelihood[i].models,
        ps_sgd_optimizer, cpp_search_domain[i],
        num_fidelity[i], discrete_pts_list,
        cpp_sgd_params_kg, num_to_sample,
        num_mc=2 ** 7)

    print(method + " takes " + str((time.time() - time1)) + " seconds for objective", i)
    print(method + " suggests points: ", next_points[i], " with voi: ", voi[i])
