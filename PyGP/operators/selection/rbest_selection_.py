import numpy

from PyGP import Program
import random
import numpy as np

def dominate_slt_(fit_list: np.array, size):
    return fit_list.argsort()[0:size]

def selection(fit_list: np.array, slt_num):
    # random.seed(0)
    bfit_collect = list(dominate_slt_(fit_list, len(fit_list)))
    fitness = []
    for i in range(slt_num):
        if i > slt_num:
            rand_val = random.randint(0, len(bfit_collect) - 1)
            # rand_val = rand_val_0 if fit_list[rand_val_0] < fit_list[rand_val_1] else rand_val_1
            count = 0
            while (np.isnan(fit_list[bfit_collect[rand_val]]) or np.isinf(fit_list[bfit_collect[rand_val]])) and count < 5:
                bfit_collect.pop(rand_val)
                rand_val = random.randint(0, len(bfit_collect) - 1)
                count += 1
            fitness.append(bfit_collect[rand_val])
            bfit_collect.pop(rand_val)
        elif i == 0:
            fitness.append(bfit_collect[0])
            bfit_collect.pop(0)
        else:
            rand_vals = np.random.choice(range(len(bfit_collect) - 1), 2, replace=False)
            # rand_val = random.randint(0, len(bfit_collect) - 1)
            # rand_val_1 = random.randint(0, len(bfit_collect) - 1)
            # rand_val_2 = random.randint(0, len(bfit_collect) - 1)
            # rand_val_3 = random.randint(0, len(bfit_collect) - 1)
            # rand_val_4 = random.randint(0, len(bfit_collect) - 1)
            slt = min(rand_vals)
            fitness.append(bfit_collect[slt])
            bfit_collect.pop(slt)
    return fitness

class RbestSelector:
    def __call__(self, fit_list, slt_num):
        return selection(fit_list, slt_num)