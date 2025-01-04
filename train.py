import PyGP
from PyGP import Population

from PyGP.utils import time_record
import time
import random
import numpy as np
from tqdm import tqdm
import textwrap


def train(run_id, train_dset, test_dset, set_seed=1111, data_id=None, d_name=None, s_name=None):

    print('')
    print(f"|{'/'*160:^160}|")
    print(f"|{'Currently training datasets: ' + str(train_dset):/^160}|")
    print(f"|{'/'*160:^160}|")
    R2_list, time_list, rmse_list, prog_size_list, test_r2_list = [], [], [], [], []
    (n_terms, range_, data_size, data, fitness, data_candidate, fitness_candidate) = PyGP.read_data_Feyn(
        train_dset, test_dset,False, False)
    

    pop = Population(pop_size=PyGP.POP_SIZE, cross_rate=0.9, mut_rate=0.9,
                     function_set=['mul', 'add', 'sub', 'div', 'sin', 'cos', 'log', 'exp', 'sqrt', 'fabs'], seed=set_seed)
    data_f_num = 0
    for i in range(n_terms):
        data_f_num += int((range_[i][1] - range_[i][0]) / 100)
    if data_f_num < data_size and data_f_num > 0 and n_terms > 10000000000:
        data_f_num = int(data_size / 10) if int(data_size / 10) > data_f_num else data_f_num
        (data_f, fitness_f) = PyGP.data_filter(data, fitness, range_, data_f_num)
        pop.initDataset(data_f, fitness_f, range_)
    else:
        pop.initDataset(data, fitness, range_)

    start = time.time()
    pop.initialization(initial_method='half-and-half', init_depth=[2, 8])
    PyGP.LIBRARY_SUPPLEMENT_INTERVAL = 1
    PyGP.LIBRARY_SUPPLEMENT_NUM = 2
    pop.bpselect_depth_ = 1

    
    func_set = [
            ['mul', 'add', 'div'],
            ['mul', 'add', 'sub', 'div'],
            ['mul', 'add', 'sub', 'div', 'log'],
            ['mul', 'add', 'sub', 'div', 'exp'],
            ['mul', 'add', 'sub', 'div', 'sqrt'],
            ['mul', 'add', 'sub', 'div', 'sin', 'cos'],
            ['mul', 'add', 'sub', 'div', 'log', 'exp'],
            ['mul', 'add', 'sub', 'div', 'sqrt'],
            ['mul', 'add', 'sub', 'div', 'sin', 'cos', 'log', 'exp', 'sqrt'],
            ]

    total_init_epoch = 20
    for i in tqdm(range(total_init_epoch), desc="Initialize the Semantic Library"):
        pop.progs = []
        for j in range(pop.pop_size):
            
            new_fset = func_set[np.random.randint(len(func_set))]
            funcs_name = new_fset
            funcs = PyGP.FunctionSet(type(PyGP.TreeNode))
            funcs.init(funcs_name)
            if isinstance(pop.init_depth, int):
                depth = pop.init_depth  # random.randint(2, init_depth + 1)
            else:
                depth = random.randint(5, 6)
            pop.progs.append(PyGP.Program(pop.pop_id, j, init_depth=depth, funcs=funcs))

        if PyGP.SEMANTIC_SIGN:
            pmask = range(pop.pop_size)
            pop.backpSelect(PyGP.LIBRARY_SUPPLEMENT_NUM, pmask)
        
        pop.execution()
        # print('=======================semantics size: ', pop.semantics.get_lib_size(), '==========================')
    pop.bpselect_depth_ = None
    pop.progs = []
    # pop.method = 'half-and-half'



    for j in range(pop.pop_size):
        
        new_fset = func_set[-1]#['mul', 'div', 'sin', 'exp', 'add']
        funcs_name = new_fset
        funcs = PyGP.FunctionSet(type(PyGP.TreeNode))
        funcs.init(funcs_name)
        if isinstance(pop.init_depth, int):
            depth = pop.init_depth  # random.randint(2, init_depth + 1)
        else:
            depth = random.randint(2, 3)
        p = PyGP.Program(pop.pop_id, j, init_depth=depth, funcs=funcs)
        repeat_b = False
        while repeat_b:
            repeat_b = False
            for k in range(0, j):
                if str(p) == str(pop.progs[k]):
                    repeat_b = True
                    p = PyGP.Program(pop.pop_id, j, init_depth=depth, funcs=funcs)
                    break
            if not repeat_b:
                break
        pop.progs.append(p)

    PyGP.LIBRARY_SUPPLEMENT_INTERVAL = 2
    PyGP.LIBRARY_SUPPLEMENT_NUM = 5
    pop.genetic_register('crossover', PyGP.SMT_Weight_Crossover_LV2, pop.pop_size)
    pop.register('crossover__', PyGP.SMT_Weight_Crossover_LV2, pop.pop_size)
    pop.register('const_optimize', PyGP.ConstOptimization, pop.pop_size)
    
    pop.slt_time = 0

    if PyGP.SEMANTIC_SIGN:
        
        pmask = np.argsort(pop.child_fitness)
        pmask = pmask[:20]
        pop.backpSelect(PyGP.LIBRARY_SUPPLEMENT_NUM, pmask)
    pop.execution(0)

    end = time.time()
    iteration = 100
    (res, R2, _, _) = pop.verify(data_candidate, fitness_candidate, inverse_transform=False)


    pop.selection()

    depth_limit = pop.depth_limit
    oper_limit = 5 * 10 ** 8
    keep_time = end - start
    train_res = 0
    eld_fit = pop.child_fitness[0]

    print('')
    print(f"|{'-'*10:<10}|{'-'*30:<30}|{'-'*30:<30}|{'-'*40:<40}|{'-'*40:<40}|{'-'*15:<15}|")
    print(f'|{"Iteration":<10}|{"R2":<30}|{"RMSE":<30}|{"Program Length(before simplified)":<40}|{"Average Length(before simplified)":<40}|{"Iter Time":<15}|')
    print(f"|{'-'*10:<10}|{'-'*30:<30}|{'-'*30:<30}|{'-'*40:<40}|{'-'*40:<40}|{'-'*15:<15}|")
    for i in range(iteration):
        start = time.time()
        mutate = False
        if oper_limit <= 0 or (pop.child_R2[0] == 1.):#pop.child_fitness[0] < 1e-12 and 
            break
        nan_arg = None
        nan_arg_cur = None
        time_s = []
        r_mut_rate = 1

    
        if i % 20 != 0 or not PyGP.OPT:
            if np.random.uniform(0, 1) < r_mut_rate:  # random.uniform(0, 1) < r_mut_rate:
                time_record(time_s, pop.crossover, pop.funcs, pop.depth_limit, fitness)
            else:
                mut_rate = 1  # pop.mut_rate + float((1. - pop.mut_rate) * i) / float(iteration * 2)
                time_record(time_s, pop.mutation, mut_rate, pop.funcs)
                mutate = True
        else:
            time_record(time_s, pop.const_optimize, data, fitness, 100)
        
        if i == (iteration - 1):
            time_record(time_s, pop.const_optimize, data, fitness, 500)
        if PyGP.SEMANTIC_SIGN:
            
            pmask = np.argsort(pop.child_fitness)
            pmask = pmask[:int(pop.pop_size / 10)]
            pop.backpSelect(PyGP.LIBRARY_SUPPLEMENT_NUM, pmask)

        time_record(time_s, pop.execution)
        time_record(time_s, pop.selection, nan_arg, nan_arg_cur)

        if pop.child_fitness[0] < eld_fit:  # i % 10 == 0:
            eld_fit = pop.child_fitness[0]
            (res, R2, _, r_cpu) = pop.verify(data_candidate, fitness_candidate, inverse_transform=False)
            test_r2_list.append((i, R2[0]))
            
        end = time.time()
        keep_time += end - start

        
        R2_list.append(pop.child_R2[0])
        time_list.append(keep_time)
        rmse_list.append(pop.child_fitness[0])
        prog_size_list.append(pop.progs[0].size)


        print(f"|{i:<10}|{pop.child_R2[0]:<30}|{pop.child_fitness[0]:<30}|{pop.progs[0].size:<40}|{pop.getAverSize():<40}|{str(round(keep_time, 2)) + ' s':<15}|")

        oper_limit -= pop.getAverSize() * pop.pop_size
        train_res = pop.child_fitness[0]

    (train_res, train_R2, _, train_res_cpu) = pop.verify(data, fitness, inverse_transform=False)

    (res, R2, _, r_cpu) = pop.verify(data_candidate, fitness_candidate, inverse_transform=False)

    f_s = np.sort(fitness_candidate)

    pop.pool.close()
    pop.pool.join()

    with open(s_name + ".csv", 'a+', newline='') as f:
        import csv
        csv_writer = csv.writer(f)
        csv_writer.writerow([data_id, d_name])
        csv_writer.writerow(R2_list)
        csv_writer.writerow(rmse_list)
        csv_writer.writerow(prog_size_list)
        csv_writer.writerow(time_list)
        csv_writer.writerow([r[0] for r in test_r2_list])
        csv_writer.writerow([r[1] for r in test_r2_list])
        csv_writer.writerow([])
    print('')
    print(f"|{'-'*30:<30}|{'-'*30:<30}|{'-'*40:<40}|")
    print(f"|{'Test R2':<30}|{'Test RMSE':<30}|{'Final Training Time':<40}|")
    print(f"|{'-'*30:<30}|{'-'*30:<30}|{'-'*40:<40}|")
    print(f"|{R2[0]:<30}|{res[0]:<30}|{str(round(keep_time, 2)) + ' s':<40}|")
    print(f"|{'-'*30:<30}|{'-'*30:<30}|{'-'*40:<40}|")
    print('')
    print(f"|{'-'*170:<170}|")
    print(f"|{'Best-train expression(before simplify)':<170}|")
    print(f"|{'-'*170:<170}|")
    print(textwrap.fill(pop.progs[0].print_exp(), 170))
    print(f"|{'-'*170:<170}|")
    
    return (res[0], (train_res[0], train_R2[0], train_res_cpu[0]), R2[0], pop.progs[0].print_exp(), keep_time, pop.progs[0].size)


import argparse
import importlib

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False)
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-run_id', dest='run_id', type=int,
                        help='the index of run')
    parser.add_argument('-job_id', dest='job_id', type=int, default=None,
                        help='the index of dataset')
    parser.add_argument('-file_name', dest='save_file', type=str, default="res.txt",
                        help='the index of dataset')
    parser.add_argument('-train_set', dest='train_set', type=str,
                        help='the index of dataset')
    parser.add_argument('-test_set', dest='test_set', type=str,
                        help='the index of dataset')
    parser.add_argument('-dataset_name', dest='d_name', type=str,
                        help='the index of dataset')
    parser.add_argument('-seed', dest='seed', type=int, default=1111,
                        help='the index of dataset')
    args = parser.parse_args()
    test_min = 100000000

    
    if args.save_file == './result/PMLB__fri_noopt.txt':
        PyGP.OPT = False
    if args.save_file == './result/PMLB__fri_noslt.txt':
        PyGP.SEMANTIC_CDD = 1
    if args.save_file == './result/PMLB__fri_nocrg.txt':
        PyGP.CRO_STG = 1
    if args.save_file == './result/PMLB_res_test_fri.txt':
        PyGP.NEW_BVAL = True

    res = train(args.run_id, args.train_set, args.test_set, args.seed, args.job_id, args.d_name, args.save_file)
    if res is None:

        with open(args.save_file, 'a+') as f:
            f.write("{0}\n".format("Sth wrong....."))
        exit(-1)
    else:
        (test, train_res, R2, exp, keep_time, prog_size) = res
    if (test < test_min):
        test_min = test
    
    with open(args.save_file, 'a+') as f:
        f.write(
            "{0} {1} {2} {3} {4} {5} {6}\n".format(R2, test, keep_time, train_res[0], train_res[1], train_res[2],
                                                   prog_size))
        f.write("{0}\n".format(exp))