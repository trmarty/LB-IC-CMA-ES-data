# -*- coding: utf-8 -*-
import numpy as np

class IntCentering:
    """round values of int-variables but not always for the mean"""
    def __init__(self, int_idxs, es=None, method=2):
        self.int_idxs = int_idxs
        self.int_mask = None
        self.es = es
        self.mahalanobis0 = []  # only for the record
        self.mahalanobis1 = []  # after modification
        self.last_changes = []
        self.last_changes_iteration = []
        self.center = getattr(self, 'method' + str(method))
    def __call__(self, solution_list, mean):
        """round values of int-variables in `solution_list` without bias to the mean"""
        return self.center(solution_list, mean)
    def method1(self, solution_list, mean):
        """round values of int-variables in `solution_list` and reduce bias to the mean"""
        m_int = np.round(mean)
        mutated_down = np.mean([np.round(x) < m_int for x in solution_list], axis=0)
        mutated_up = np.mean([np.round(x) > m_int for x in solution_list], axis=0)
        maxmut_ratio = np.where(mutated_down > mutated_up, mutated_down, mutated_up)
        # assert all(maxmut_ratio == np.max([mutated_down, mutated_up], axis=0))  # 3x slower
        self.last_changes = []
        for x in solution_list:
            for i in self.int_idxs:
                if (np.round(x[i]) != m_int[i] or #True or #(False and (
                    maxmut_ratio[i] > 0.5 or  # reduce bias
                    np.random.rand() < maxmut_ratio[i] / (1 - mutated_up[i] - mutated_down[i])): #)):
                    # self.mahalanobis0.append(self.es.mahalanobis_norm(x - mean))
                    x[i] = np.round(x[i])
                    # self.mahalanobis1.append(self.es.mahalanobis_norm(x - mean))
                    # n0, n1 = self.mahalanobis0[-1], self.mahalanobis1[-1]
                    # self.last_changes.append([i, n0, n1])
                    # self.last_changes_iteration.append(es.countiter)
        return solution_list  # declarative, elements of solution_list have changed in place
    def method2(self, solution_list, mean):
        """round values of int-variables in `solution_list` without bias to the mean"""
        mean0 = np.mean(solution_list, axis=0)  # for the record only
        if self.int_mask is None:
            self.int_mask = np.asarray([i in self.int_idxs
                for i in range(len(mean))])
        m_int = np.round(mean)
        offbiases = np.zeros(len(solution_list[0]))
        '''bias created from setting off-mean solutions'''
        mneg = np.zeros(len(solution_list[0]))
        mpos = np.zeros(len(solution_list[0]))
        for x in solution_list:
            x_int = np.round(x)
            ism = x_int == m_int
            mpos += ism * ((x_int - x) > 0) * (x_int - x)
            mneg += ism * ((x_int - x) < 0) * (x_int - x)
            offbiases += ~ism * (x_int - x)
            x[~ism * self.int_mask] = x_int[~ism * self.int_mask]  # round off-mean values
        # compare offbiases with mpos or mneg
        alphas = []
        for i, (b, p, n) in enumerate(zip(offbiases, mpos, mneg)):
            if not self.int_mask[i]:
                alphas += [0.]
            elif b * p < 0:
                alphas += [-b / p if -b < p else 1.0]
            elif b * n < 0:
                alphas += [-b / n if b < -n else 1.0]
            else:
                alphas += [0.]
        alphas = np.asarray(alphas)
        for x in solution_list:
            x += alphas * (m_int - x) * (np.round(x) == m_int) * (offbiases * (m_int - x) < 0)
        if 11 < 3:  # print remaining biases
            mean1 = np.mean(solution_list, axis=0)
            biases = [(i, mean1[i] - mean0[i]) for i in range(len(mean0))
                    if (mean1[i] - mean0[i])**2 > 1e-22]
            if biases:
                print(self.es.countiter, biases)
        return solution_list