# -*- coding: utf-8 -*-
"""
Code snippet on how to use 

@author: tristan
"""
import numpy as np
from intCentering import IntCentering
import cma
import cocoex


def mueff(lam):
    wi = cma.recombination_weights.RecombinationWeights(lam)
    return 1 / sum(wi.positive_weights**2)

# Define the lower bound value
def sigma_lb(N, lam):
    return min(mueff(lam)/N, 0.2)

# Define bbob mixint problem
suite = cocoex.Suite("bbob-mixint", "", "")
suite_filter_options = (#"function_indices: 3-5"  # without filtering, a suite has instance_indices 1-15
                        "dimensions: 5,10,20,40,80 "   # skip dimension 40
                        "instance_indices: 1-15 "  # relative to suite instances
                       )
problem = suite.get_problem_by_function_dimension_instance(1, 40, 1)

# Activate lower bound and integer centering
integer_centering = True
lower_bound = True

int_idxs = list(range(problem.number_of_integer_variables))
bounds = [[],[]]
bounds[0] = [problem.lower_bounds[idx] - 0.5 if idx in int_idxs else -np.inf for idx in range(problem.dimension)]
bounds[1] = [problem.upper_bounds[idx] + 0.5 if idx in int_idxs else  np.inf for idx in range(problem.dimension)]

# callback to stop optimization after problem is solved
target_callback = lambda es : problem.final_target_hit

budget_multiplier = 20000
evalsleft = lambda: int(problem.dimension * budget_multiplier + 1 -
                        max((problem.evaluations, problem.evaluations_constraints)))

restarts = 0
max_restarts = 9

popsize = int(4 + 3*np.log(problem.dimension))

# Main optimization loop
while True:
    opts = {'maxfevals':evalsleft(), 
            'tolfunhist':0, 'tolflatfitness': 5,  # Stopping conditions
            'termination_callback':target_callback, 
            'CMA_stds': ( problem.upper_bounds - problem.lower_bounds ) / 5, #Initial coordinate wise std
            'popsize':popsize,
            }
    
    if lower_bound:
        sigma_min = sigma_lb(problem.dimension, popsize) 
        opts["minstd"] = [sigma_min if idx in int_idxs else 0 for idx in range(problem.dimension)]
        
    boundary_handler = cma.constraints_handler.BoundPenalty(bounds=bounds)
    es = cma.CMAEvolutionStrategy(problem.initial_solution_proposal, 1, opts)
    
    if integer_centering:
        int_centering = IntCentering(int_idxs, es)
    
    while not es.stop():  # iteration loop
        X = es.ask()
        fit = np.array([problem(x) for x in X])
        fit += boundary_handler.update(fit, es)(X, es.sent_solutions, es.gp) #Boundary handling
        if integer_centering:
            int_centering([X[i] for i in np.argsort(fit)[:es.sp.weights.mu]], es.mean)
        es.tell(X, fit)
        es.disp()

    if restarts > max_restarts or 'ftarget' in es.stop() \
            or 'maxfevals' in es.stop(check=False) or 'callback' in es.stop(check=False):
        break
    
    restarts += 1
    popsize *= 2

print("best function value :%f" %(es.best.f))
print("best solution :")
print(es.best.x)
