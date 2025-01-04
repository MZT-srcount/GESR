import itertools

import PyGP
from PyGP import Program
import pycuda.autoinit as autoinit
import random


def _input_collect(childs, cash_open, output, cnode):
    def set_input(x):
        input_id = x.getCashId() if x.getCashState() == 1 and cash_open else output[x.node_id]
        if x.getCashState() == 2 and cash_open:
            x.setCashState(1)
        return input_id

    return list(map(lambda x: set_input(x), childs))


def gradient_exp_generate(prog: Program, output, cash_open=False):
    stack = [[prog.root, 1]]
    const_num = 0
    expunit_collects = []
    c_sign = {}
    cnode_val = []
    cnode_tr = []
    while stack:
        cnode = stack.pop()

        # 第一次访问且非叶子节点；如果该节点已经被缓存，则不再会访问子节点
        if cnode[1] == 1 and cnode[0].dtype == "Func":
            childs = cnode[0].getChilds()
            stack.append([cnode[0], cnode[1] + 1])
            stack.extend(list(map(lambda x: [x, 1], childs)))

        # 第二次访问/叶子节点
        elif cnode[0].dtype == "Func":
            # assert (not (cnode[0] == prog.root and prog.root.dtype != "Func"))  # root can not be an input or const value

            # 自定义节点处理，暂不考虑
            childs = cnode[0].getChilds()
            for i in range(len(childs)):
                if c_sign.get(childs[i].node_id):
                    if not c_sign.get(cnode[0].node_id):
                        c_sign[cnode[0].node_id] = []
                    for k in c_sign[childs[i].node_id]:
                        # 输入
                        expunit = [cnode[0].nodeval.id]
                        childs = cnode[0].getChilds()
                        expunit.append(len(childs))  # input size
                        expunit.extend(_input_collect(childs, cash_open, output, cnode[0]))
                        assert (len(expunit) - 2 == len(childs))
                        # 操作符位置
                        expunit.append(i)
                        # 输出
                        expunit.append(k)

                        expunit_collects[k].append(expunit)
                        c_sign[cnode[0].node_id].append(k)


        elif cnode[0].dtype == "Const":
            c_sign[cnode[0].node_id] = [const_num]
            expunit_collects.append([])
            cnode_val.append(cnode[0].nodeval)
            cnode_tr.append(cnode[0])
            const_num += 1

    return (expunit_collects, cnode_val, cnode_tr)


import pycuda.driver as cuda
from src.cuda_backend import mod
import numpy as np
import scipy
# import jax.numpy as jnp
from ..base_oper import BaseOperator
import time

def is_full_rank(matrix):
    # print(matrix.shape[0], np.linalg.matrix_rank(matrix))
                    
    return np.linalg.matrix_rank(matrix) == matrix.shape[0]

class ConstOptimization(BaseOperator):
    def __init__(self, pop_size):
        self.pop_size = pop_size

    def run(self, prog, dataset, output, cash_open=False, iter_time=200):
        return self.gauss_newton(prog, dataset, output, iter_time)

    def gauss_newton(self, progs, dataset, output, iter_time):
        executor = PyGP.Execution()
        dataset_len = len(dataset[0])
        progs_size = len(progs)
        res = executor(progs, dataset)

        (Y, cnode_vals, cnode_trs, prog_opt) = self.gradient_exec(progs, dataset, executor)
        if Y is None:
            return progs
        prefix_sum = 0
        prefix_array = [0]
        res = [np.array(res[dataset_len * i: dataset_len * (i + 1)]) for i in prog_opt]

        for i in range(len(cnode_vals)):
            prefix_sum += len(cnode_vals[i])
            prefix_array.append(prefix_sum)

        stream = cuda.Stream()
        execution_GPU = mod.get_function("execution_GPU")
        grad_exec = mod.get_function("gradient")
        cvals = np.array(list(itertools.chain(*cnode_vals)), dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
        # print('np.shape(cvals)', np.shape(cvals))
        
        lambda_1 = 0.01
        
        # origin_error = [np.dot((output - res[i]), (output - res[i])) + lambda_1 / 2 * np.dot(cvals[prefix_array[i]: prefix_array[i + 1]], cvals[prefix_array[i]: prefix_array[i + 1]]) for i in range(len(res))]
        # origin_error = [np.sqrt(np.dot((output - res[i]), (output - res[i]))) for i in range(len(res))]
        origin_error = [np.sqrt(np.dot((output - res[i]), (output - res[i]))) + lambda_1 * np.sum(np.fabs(cvals[prefix_array[i]: prefix_array[i + 1]])) for i in range(len(res))]
        
        last_error = best_error = origin_error.copy()

        Y = [np.transpose(np.array(Y[dataset_len * prefix_array[i]:dataset_len * prefix_array[i + 1]] 
        ).reshape(
            (len(cnode_vals[i]), dataset_len))) for i in range(len(cnode_vals))]
            
        YTY = [np.dot(np.transpose(Y[i]), Y[i]) for i in range(len(Y))]

        u0 = [np.max(YTY[i]) * 10 ** -3 for i in range(len(YTY))]
        u0 = [random.uniform(0, 1) if u0[i] == 0 else u0[i] for i in range(len(YTY))]
        start_ = time.time()
        pinv_time = 0
        t_0 , t_1 = 0, 0
        for z in range(iter_time):
            error = [res[i] - output for i in range(len(res))]
            # res_YT = [np.dot(np.transpose(Y[i]), error[i]).astype(np.float64) + np.fabs(cvals[i]) * lambda_1 for i in range(len(Y))]
            # res_YT = [np.dot(np.transpose(Y[i]) * cvals[i] * lambda_1 / len(cnode_vals[i]), error[i]).astype(np.float64) for i in range(len(Y))]
            # res_YT = [np.dot(np.transpose(Y[i]), error[i]).astype(np.float64) for i in range(len(Y))]
            res_YT = [np.dot(np.transpose(Y[i]), error[i]).astype(np.float64) + lambda_1 for i in range(len(Y))]
            delta_ks = []
            st_0 = time.time()
            for i in range(len(Y)):
                # det_1 = np.linalg.det(YTY[i])
                # det_2 = np.linalg.det(YTY[i])
                # det_3 = np.linalg.det(res_YT[i])
                # if not (np.isnan(YTY[i]).any() or np.isnan(YTY[i]).any() or np.isnan(
                #         res_YT[i]).any() or np.isinf(res_YT[i]).any() or np.isinf(u0[i]).any() or np.isnan(u0[i]).any()):
                # matrix_tmp = (YTY[i] + u0[i] * np.random.uniform(0.9999, 1.0001, size=YTY[i].shape) ).astype(np.float64)
                matrix_tmp = (YTY[i] + u0[i] * np.ones(shape=YTY[i].shape)).astype(np.float64)
                
                # matrix_tmp = np.where(matrix_tmp == 0., np.random.uniform(0.000001, 0.001), matrix_tmp)
                
                if not (np.isnan(YTY[i]).any() or np.isnan(YTY[i]).any() or np.isnan(
                        res_YT[i]).any() or np.isinf(res_YT[i]).any() or np.isinf(u0[i]).any() or np.isnan(u0[i]).any()):
                    matrix_tmp_max = np.max(matrix_tmp)
                    matrix_tmp /= matrix_tmp_max
                    if is_full_rank(matrix_tmp):
                    #     # delta = np.linalg.solve((YTY[i] + u0[i] * np.ones(shape=YTY[i].shape)).astype(np.float64),
                    #     #                            res_YT[i])  # 0.1 * np.dot(JX_s, vec * tgdrvt_origin)
                        delta = np.linalg.inv(matrix_tmp)
                    else:
                        # print(matrix_tmp)
                        start = time.time()
                        delta = np.linalg.pinv(matrix_tmp)#np.linalg.pinv((YTY[i] + u0[i] * np.ones(shape=YTY[i].shape)).astype(np.float64)) @ res_YT[i]
                        pinv_time += time.time() - start
                    res_YT_max = np.max(np.fabs(res_YT[i]))
                    if res_YT_max == 0.:
                        delta_ks.append(np.zeros(len(cnode_vals[i]), dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64))
                        continue
                    res_YT[i] /= res_YT_max
                    delta = np.matmul(delta / matrix_tmp_max, res_YT[i]) * res_YT_max
                    if np.isnan(delta).any() or np.isinf(delta).any():
                        delta = np.zeros(len(cnode_vals[i]), dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
                        
                    delta_ks.append(delta)
                
                else:
                    delta_ks.append(np.zeros(len(cnode_vals[i]), dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64))

            t_0 += time.time() - st_0
            cvals_ = np.array([cvals[prefix_array[i] + j] - delta_ks[i][j] if not
            (np.isnan(delta_ks[i]).any() or np.isnan(delta_ks[i]).any() or np.isinf(delta_ks[i]).any() or np.isinf(delta_ks[i]).any()) else cvals[prefix_array[i] + j]
                              for i in range(len(cnode_vals)) for j in range(len(cnode_vals[i]))], dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64)

            st_1 = time.time()
            const_vals_gpu = executor.exec_para['cvals']
            executor.cuda_manager.host2device(cvals_, const_vals_gpu)
            execution_GPU(executor.exec_para['exps'], executor.exec_para['exp_posi'],
                          executor.exec_para['input_gpu'], np.int64(executor.exec_para['input_pitch']),
                          executor.exec_para['info'], const_vals_gpu,
                          block=(int(32), 1, 1),
                          grid=(int((progs_size) * PyGP.BATCH_NUM), 1, 1),
                          stream=stream)

            res = np.empty(dataset_len * progs_size).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64)

            executor.cuda_manager.memcopy_2D(res,
                                             dataset_len * PyGP.DATA_TYPE,
                                             executor.exec_para['input_gpu'], executor.exec_para['input_pitch'],
                                             dataset_len * PyGP.DATA_TYPE, progs_size,
                                             stream, 0,
                                             src_y_offset=int(progs[0].n_terms + 1))

            autoinit.context.synchronize()
            res = [np.array(res[dataset_len * i: dataset_len * (i + 1)]) for i in prog_opt]

            grad_exec(self.gradient_para['exps'], self.gradient_para['exps_posi'], self.gradient_para['Y_gpu'],
                      np.int64(self.gradient_para['Y_pitch']), executor.exec_para['input_gpu'],
                      np.int64(executor.exec_para['input_pitch']), executor.exec_para['info'], const_vals_gpu,
                      np.int32(dataset_len * prefix_sum),
                      block=(32, 1, 1), grid=(prefix_sum, 1, 1), stream=stream
                      )
            autoinit.context.synchronize()

            Y_ = np.empty(dataset_len * prefix_sum, dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
            executor.cuda_manager.memcopy_2D(Y_, dataset_len * PyGP.DATA_TYPE, self.gradient_para['Y_gpu'],
                                             self.gradient_para['Y_pitch'], dataset_len * PyGP.DATA_TYPE,
                                             prefix_sum, stream, 0, 0)
            autoinit.context.synchronize()
            Y_ = [np.transpose(np.array(Y_[dataset_len * prefix_array[i]:dataset_len * prefix_array[i + 1]] 
        ).reshape((len(cnode_vals[i]), dataset_len))) for i in range(len(cnode_vals))]
            # cur_error = [np.dot((output - res[i]), (output - res[i])) + lambda_1 / 2 * np.dot(cvals_[prefix_array[i]: prefix_array[i + 1]], cvals_[prefix_array[i]: prefix_array[i + 1]]) for i in range(len(res))]
            # cur_error = [np.sqrt(np.dot((output - res[i]), (output - res[i]))) for i in range(len(res))]
            cur_error = [np.sqrt(np.dot((output - res[i]), (output - res[i]))) + lambda_1 * np.sum(np.fabs(cvals_[prefix_array[i]: prefix_array[i + 1]])) for i in range(len(res))]
        
            
            t_1 += time.time() - st_1
            for i in range(len(cur_error)):
                if cur_error[i] > last_error[i]:  # ! [ ]
                    u0[i] *= 2
                else:
                    u0[i] /= 3
                    if cur_error[i] < best_error[i]:
                        best_error[i] = cur_error[i].copy()
                        Y[i] = Y_[i]
                        YTY[i] = np.dot(np.transpose(Y[i]), Y[i])
                        for j in range(len(cnode_trs[i])):
                            cvals[prefix_array[i] + j] = cvals_[prefix_array[i] + j]

            last_error = cur_error.copy()
        # print('cstop_time:   ', pinv_time, t_0, t_1, time.time() - start_)
        for i in range(len(best_error)):
            if best_error[i] < origin_error[i]:
                # print('succeed...', i)
                for j in range(len(cnode_trs[i])):
                    cnode_trs[i][j].nodeval = cvals[prefix_array[i] + j]
        return progs

    def gradient_exec(self, progs, dataset, executor):
        # n_terms = progs[0].n_terms
        dataset_len = len(dataset[0])
        progs_size = len(progs)
        cuda_manager = executor.cuda_manager
        cnode_num = 0
        init_posi = 0
        cnode_iposi = [0]
        exp_iposi = [0]
        cnode_vals = []
        cnode_trs = []
        gexps = []
        prog_opt = []
        for i in range(progs_size):
            (gexps_, cnode_vals_, cnode_trs_) = gradient_exp_generate(progs[i], executor.expms[i], False)
            if len(gexps_) > 0:
                prog_opt.append(i)
                # gexps_ = list(map(lambda x: x.reverse(), gexps_))
                for exp in gexps_:
                    exp.reverse()
                    exp = list(itertools.chain(*exp))
                    exp.append(-1)
                    init_posi += len(exp)
                    exp_iposi.append(init_posi)
                    gexps.extend(exp)
                assert (len(gexps_) == len(cnode_trs_))
                cnode_iposi.append(len(cnode_trs_))
                cnode_vals.append(cnode_vals_)
                cnode_trs.append(cnode_trs_)
                cnode_num += len(cnode_trs_)
        
        if len(cnode_vals) == 0:
            return (None, None, None, None)
        (Y_gpu, Y_pitch) = cuda.mem_alloc_pitch(dataset_len * PyGP.DATA_TYPE, cnode_num,
                                                PyGP.DATA_TYPE)  # 雅可比矩阵计算

        grad_exec = mod.get_function("gradient")

        exps_gpu = cuda.mem_alloc(len(gexps) * 4)
        cuda_manager.host2device(np.array(gexps, dtype=np.int32), exps_gpu)

        exp_iposi_gpu = cuda.mem_alloc(len(exp_iposi) * 4)
        cuda_manager.host2device(np.array(exp_iposi, dtype=np.int32), exp_iposi_gpu)

        stream = cuda.Stream()
        """
        __global__ void gradient(int* program, int* progs_initposi, double* results, double* dataset, int pitch, int* info, double* const_vals, int exp_num){
        """
        grad_exec(exps_gpu, exp_iposi_gpu, Y_gpu, np.int64(Y_pitch), executor.exec_para['input_gpu'],
                  np.int64(executor.exec_para['input_pitch']), executor.exec_para['info'], executor.exec_para['cvals'],
                  np.int32(dataset_len * cnode_num),
                  block=(32, 1, 1), grid=(cnode_num, 1, 1), stream=stream
                  )
        autoinit.context.synchronize()
        Y = np.empty(dataset_len * cnode_num, dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
        cuda_manager.memcopy_2D(Y, dataset_len * PyGP.DATA_TYPE, Y_gpu, Y_pitch, dataset_len * PyGP.DATA_TYPE,
                                cnode_num, stream, 0, 0)

        self.gradient_para = {
            "Y_gpu": Y_gpu,
            "Y_pitch": Y_pitch,
            "exps": exps_gpu,
            "exps_posi": exp_iposi_gpu
        }

        # del exps_gpu
        # del exp_iposi_gpu
        # del Y_gpu
        
        return (Y, cnode_vals, cnode_trs, prog_opt)

