import numpy

from PyGP import Program
import random
import numpy as np

def dominate_slt_(fit_list: np.array, size):
    b = list(range(int(size / 2), size)) + list(range(int(size / 2)))
    return np.lexsort((b, fit_list))
    # return fit_list.argsort()

# def selection(fit_list: np.array, prlt_list, slt_num):
#     bfit_collect = list(dominate_slt_(fit_list, len(fit_list)))
#     fitness = []
#     for i in range(slt_num):
#         if np.random.uniform(0, 1) < 0.5:
#             fitness.append(i if fit_list[i] < fit_list[i + slt_num] else i + slt_num)
#         else:
#             fitness.append(i + slt_num * np.random.randint(0, 2))
#     return fitness

def selection(fit_list: np.array, prlt_list, slt_num):
    # random.seed(0)
    bfit_collect = list(dominate_slt_(fit_list, len(fit_list)))
    fitness = []
    # print(fit_list[0:100], fit_list[200:300])
    # print(fit_list[bfit_collect])
    for i in range(slt_num):
        if i > slt_num * 11 / 10:
            rand_val = random.randint(0, len(bfit_collect) - 1)
            # rand_val = rand_val_0 if fit_list[rand_val_0] < fit_list[rand_val_1] else rand_val_1
            count = 0
            # while (np.isnan(fit_list[bfit_collect[rand_val]]) or np.isinf(fit_list[bfit_collect[rand_val]])) and count < 5:
            #     bfit_collect.pop(rand_val)
            #     rand_val = random.randint(0, len(bfit_collect) - 1)
            #     count += 1
            fitness.append(bfit_collect[rand_val])
            bfit_collect.pop(rand_val)
        elif i == 0:
            init_id = 0
            # print('best_id: ', bfit_collect[init_id], 'origin_ft: ', fit_list[bfit_collect[init_id]] if bfit_collect[init_id] < slt_num else fit_list[bfit_collect[init_id] - slt_num])
            succeed = False

            fitness.append(bfit_collect[init_id])
            bfit_collect.pop(init_id)

            # while fit_list[bfit_collect[init_id]] / (1+prlt_list[bfit_collect[init_id]]) < fit_list[0] / (1+prlt_list[0]) or np.isnan(prlt_list[0]):
            #     if not np.isnan(prlt_list[bfit_collect[init_id]]) and fit_list[bfit_collect[init_id]] < fit_list[0] and (1+prlt_list[bfit_collect[init_id]]) > (1+prlt_list[0]):
            #         fitness.append(bfit_collect[init_id])
            #         bfit_collect.pop(init_id)
            #         succeed=True
            #         break
            #     elif not np.isnan(prlt_list[bfit_collect[init_id]]) and np.isnan(prlt_list[0]):
            #         fitness.append(bfit_collect[init_id])
            #         bfit_collect.pop(init_id)
            #         succeed=True
            #         break
            #     init_id += 1
            # if not succeed:
            #     fitness.append(0)
        elif i > slt_num * 1 / 10:
            time = 3
            # rand_vals = [i, i + slt_num]
            rand_vals = np.random.choice(range(len(bfit_collect) - 1), time, replace=False)
            # rand_val = random.randint(0, len(bfit_collect) - 1)
            # rand_val_1 = random.randint(0, len(bfit_collect) - 1)
            # rand_val_2 = random.randint(0, len(bfit_collect) - 1)
            # rand_val_3 = random.randint(0, len(bfit_collect) - 1)
            # rand_val_4 = random.randint(0, len(bfit_collect) - 1)

            slt = min(rand_vals)
            fitness.append(bfit_collect[slt])
            bfit_collect.pop(slt)

            # rand_vals = np.sort(rand_vals)
            # min_val = 100000
            # min_slt = -1
            # for j in range(time):
            #     slt = rand_vals[j]
            #     if fit_list[bfit_collect[slt]] < min_val:
            #         min_val = fit_list[bfit_collect[slt]]
            #         min_slt = slt
            #         # fitness.append(bfit_collect[slt])
            #         # bfit_collect.pop(slt)
            #         # succeed = True
            #         # break
            # if min_slt != -1:
            #     fitness.append(bfit_collect[min_slt])
            #     bfit_collect.pop(min_slt)
            # else:
            #     fitness.append(i)
        else:
            time = 3
            # rand_vals = [i, i + slt_num]
            rand_vals = np.random.choice(range(len(bfit_collect) - 1), time, replace=False)
            # rand_val = random.randint(0, len(bfit_collect) - 1)
            # rand_val_1 = random.randint(0, len(bfit_collect) - 1)
            # rand_val_2 = random.randint(0, len(bfit_collect) - 1)
            # rand_val_3 = random.randint(0, len(bfit_collect) - 1)
            # rand_val_4 = random.randint(0, len(bfit_collect) - 1)

            slt = min(rand_vals)
            fitness.append(bfit_collect[slt])
            bfit_collect.pop(slt)

            # rand_vals = np.sort(rand_vals)
            # min_val = 100000
            # min_slt = -1
            # for j in range(time):
            #     slt = rand_vals[j]
            #     if fit_list[bfit_collect[slt]] < min_val:
            #         min_val = fit_list[bfit_collect[slt]]
            #         min_slt = slt
            #         # fitness.append(bfit_collect[slt])
            #         # bfit_collect.pop(slt)
            #         # succeed = True
            #         # break
            # if min_slt != -1:
            #     fitness.append(bfit_collect[min_slt])
            #     bfit_collect.pop(min_slt)
            # else:
            #     fitness.append(i)
    return fitness

class RbestSelector:
    def __call__(self, fit_list, prlt_list, slt_num):
        return selection(fit_list, prlt_list, slt_num)