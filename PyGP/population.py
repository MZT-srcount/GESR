import random
import pycuda.driver as cuda
import pycuda.autoinit as autoinit
import PyGP
from src.cuda_backend import host_to_gpu, MemManager, mod, Info, EvalParaInfo
import numpy as np
import time
from PyGP import cashUpdate, cashGenerate
from PyGP import Program, DataCollects, SManager, Func, FunctionSet
from .operators import FuncCall
import multiprocessing
import dill
import array, ctypes
from multiprocessing.sharedctypes import RawArray as SArray
import math

multiprocessing.set_start_method('spawn', force=True)

class Population(PyGP.Base):
    def __init__(
            self,
            pop_size,
            cross_rate=0.1,
            mut_rate=0.2,
            function_set=None,
            rand_state=None,
            const_range=None,
            subpop_size=None,
            input_gmem_alloc=8 * (1024 ** 3),
            cash_gmem_alloc=2 * (1024 ** 3),
            type=type(PyGP.TreeNode),
            seed=None,
    ):
        if function_set is None:
            funcs = ["add", "sub", "mul", "div"]
        else:
            funcs = function_set

        self.semsave_cpu = []
        self.semsave_sign = {}
        self.funcs_name = funcs
        self.funcs = FunctionSet(type)
        self.funcs.init(self.funcs_name)
        self._crossover = PyGP.SMT_Weight_Crossover_LV2
        self._mutation = PyGP.RtnMutation()
        self.backfuncs = []  # 反向传播函数
        self.semantic_data = {}  # 语义信息保存，节点gpu位置：语义
        self.last_best = None
        self.iter_id = 0
        if seed is not None:
            self.seed=seed
        else:
            self.seed=time.time()
            assert (0==1)
        # self.seeds_tmp = [np.random.RandState() for i in range(PyGP.PARALLEL)]
        self.seedseq = np.random.SeedSequence(np.random.randint(2**31 - 1))
        
        # self.seeds_tmp = [np.random.RandomState(np.random.randint(2 ** 31)) for i in range(PyGP.PARALLEL)]
        # print(type(self.seeds_tmp), self.seeds_tmp)
        # assert(0 == 1)


        self.cross_rate = cross_rate
        self.mut_rate = mut_rate
        if rand_state is not None:
            self.rand_state = rand_state
        else:
            # random.seed(0)
            self.rand_state = np.random

        if const_range is not None:
            self.const_range = const_range
        else:
            self.const_range = None

        if subpop_size is None:
            if pop_size > 4:
                self.subpop_size = 4
            else:
                self.subpop_size = pop_size
        else:
            self.subpop_size = subpop_size
        self.input_gmem_alloc = input_gmem_alloc
        self.cuda_mem_manager = MemManager(input_gmem_alloc, cash_gmem_alloc)

        self.device = 0
        self.pop_size = pop_size
        self.t_dataset = None
        super()._init(self.seed)
        self.pop_id = self.POP_SIZE
        self.POP_SIZE += 1

    def func_register(self, func_name, func, arity):
        self.funcs.register(func_name, func, arity)

    def setDevice(self, device):
        if device >= cuda.Device.count():
            raise ValueError("can not find such device")
        self.device = device

    def initialization(
            self,
            initial_method,
            init_depth,
            cash_size=PyGP.CASHNUM_PERBLOCK,
    ):
        self.bpselect_depth_ = None
        self.progs = []
        self.pprogs = []
        self.init_depth = init_depth
        self.cashInit()
        self.isImproved = False
        self.Improved_100p = 0
        self.Improved_100p_50 = 0
        self.depth_limit = PyGP.DEPTH_LIMIT

        self.fit_ablt = [0, 1]

        self.slt_time = 0
        self.iter_time = 0
        self.cov = False
        self.cash_size = int(self.CASH_MANAGER.getCashSize(1) / self.pop_size)

        super().property_add({
            'rd_st': self.rand_state,
            'const_range': self.const_range,
            'init_depth': self.init_depth,
            'method': initial_method,
            'cash_size': self.cash_size,
            'n_terms': self.n_terms,
            'data_rg': self.data_rg,
            'funcs': self.funcs
        })

        for i in range(self.pop_size):
            if isinstance(init_depth, int):
                depth = init_depth#random.randint(2, init_depth + 1)
            else:
                depth = np.random.randint(self.init_depth[0], self.init_depth[1] + 1)
            self.progs.append(Program(self.pop_id, i, init_depth=depth))


        self.sharedmemory = [SArray(ctypes.c_double, PyGP.MAX_SIZE * 6) for i in range(self.pop_size)]
        self.pool = multiprocessing.Pool(processes=PyGP.PARALLEL, initializer= PyGP.init_shared, initargs=(self.sharedmemory,  dill.dumps(self.getBasecore()), self.seed))
        self.semantics = self.PROC_MANAGER.PopSemantic(data_rg=self.data_rg, base_dict=dill.dumps(self.getBasecore()), seed=self.seed)
        # self.semantics = SManager(proc_manager=self.PROC_MANAGER, data_rg=self.data_rg, base_dict=dill.dumps(self.getBasecore()), seed=self.seed)

    def backpSelect(self, select_num, pmask):  # select nodes for semantic backpropagation
        self.semantics.reset() #refresh
        # s_idx = np.argsort(self.child_fitness)
        # s_idx = s_idx[:int(self.pop_size / 10)]
        s_idx = pmask
        if self.slt_time % PyGP.LIBRARY_SUPPLEMENT_INTERVAL == 0:
            self.semantics.set_library(PyGP.backpSelect(self, select_num, self.bpselect_depth_, range(len(self.progs)), s_idx))
        else:
            self.semantics.set_library(PyGP.backpSelect(self, 0, self.bpselect_depth_, range(len(self.progs)), s_idx))
        self.slt_time += 1

    def genetic_register(self, gname, func, *args):
        if gname == "crossover":
            self._crossover = func(*args)
        elif gname == "mutation":
            self._mutation = func(*args)

    def return_callback(self, res):
        for j in res:
            if j != -1:
                if(j == 0):
                    print("HEREEEEEEE")
                self.progs[j] = PyGP.unzip(self.sharedmemory[j])

    def register(self, gname, func, *args):
        if not callable(func):
            raise ValueError("gname %s is not callable." % gname)
        oper_new = FuncCall(gname, func, self, *args)
        setattr(self, gname, oper_new)

    def crossover(self, *args):
        self.cov = True
        seqs = self._crossover.preprocess()
        split_seqs = np.array_split(seqs, PyGP.PARALLEL)
        # if self.iter_id % 50 == 0:
        #     self.pool.close()
        #     self.pool = multiprocessing.Pool(processes=PyGP.PARALLEL)

        # sm = shared_memory.ShareableList([list(x.values())[1:] for x in self.progs[x].zip()])
        mp_progs = []
        base_dict = dill.dumps(self.getBasecore())

        self.parallel_prepare()

        # self.seeds_tmp = self.seedseq.spawn(PyGP.PARALLEL)
        # print(self.seeds_tmp)
        self.seeds_tmp = self.seedseq.spawn(PyGP.PARALLEL)#np.random.randint(2 ** 31, size=PyGP.PARALLEL)

        for i in range(1, PyGP.PARALLEL, 1):
            shared_list = []
            for j in split_seqs[i]:
                progs_zip = self.progs[j].zip()
                self.sharedmemory[j][:len(progs_zip)] = progs_zip
                shared_list.append(j)
            mp_progs.append(self.pool.apply_async(self._crossover, args=(shared_list, self.seeds_tmp[i], self.semantics, *args), callback=self.return_callback))
        new_progs = self._crossover([self.progs[i] for i in split_seqs[0]], self.seeds_tmp[0], self.semantics, *args)
        # print('new_progs', new_progs)
        for i, x in enumerate(split_seqs[0]):
            self.progs[x] = new_progs[i]
        # progs = []
        for i in range(0, PyGP.PARALLEL - 1):
            mp_progs[i].wait()
        # for i in range(1, len(split_seqs), 1):
        #     self.progs.extend([PyGP.unzip(self.sharedmemory[j]) for j in split_seqs[i]])
        for i in range(self.pop_size):
            self.progs[i].setId(i)

        self.parallel_restore()

        # self.progs = self._crossover(PyGP.Base.CASH_MANAGER, PyGP.Base.ID_MANAGER, self.progs, *args)
        # for i in range(self.pop_size):
        #     self.progs[i].setId(i)

    def mutation(self, *args):
        self._mutation(self.progs, self.semantics, *args)
        # self.progs = [self.progs[i].copy(i) for i in range(len(self.progs))]

    def evaluation(self):
        raise NotImplementedError

    def selection(self, nan_child=None, nan_cur=None):
        if not self.pprogs:
            self.pprogs = [self.progs[i].copy(i) for i in range(len(self.progs))]
            self.cur_fitness = self.child_fitness.copy()
            self.cur_R2 = self.child_R2.copy()
            self.cur_prlt = self.child_prlt.copy()
            return
        times = []
        # if self.pprogs is None:
        #     self.pprogs = self.progs
        #     self.cur_fitness = self.child_fitness.copy()
        #     self.progs = []
        #     return
        start = time.time()
        sltor = PyGP.RbestSelector()
        if nan_child is None:
            nan_child = np.isnan(self.child_fitness)==False
        if nan_cur is None:
            nan_cur = np.isnan(self.cur_fitness)==False
        # print(nan_cur, nan_child)
        fit_list = list(self.cur_fitness) + list(self.child_fitness)
        R2_list = list(self.cur_R2) + list(self.child_R2)
        prlt_list = list(self.cur_prlt) + list(self.child_prlt)
        # prog_list = self.pprogs[np.where(nan_cur)] + self.progs[np.where(nan_child)]
        prog_list = self.pprogs + self.progs
        # fit_list = list(self.cur_fitness[np.where(nan_cur)]) + list(self.child_fitness[np.where(nan_child)])
        # prog_list = [self.pprogs[i] for i in np.where(nan_cur)[0]] + [self.progs[i] for i in np.where(nan_child)[0]]
        fit_slt = sltor(np.array(fit_list), np.array(prlt_list), self.pop_size)
        self.rk_posi = fit_slt
        self.semantics.select(fit_slt)
        end = time.time()
        times.append(end - start)
        # self.pprogs = self.progs
        start = time.time()
        self.progs = [prog_list[x].copy(i) for i, x in enumerate(fit_slt)]
        end = time.time()
        times.append(end - start)
        
        self.child_R2 = np.array(list(map(lambda x: R2_list[x], fit_slt)))
        self.child_fitness = np.array(list(map(lambda x: fit_list[x], fit_slt)))
        self.child_prlt = np.array(list(map(lambda x: prlt_list[x], fit_slt)))

        start = time.time()
        fit_tmp = self.fit_ablt[0] / self.fit_ablt[1]

        if self.last_best is None or (self.last_best > self.child_fitness[0] and self.cov):
            self.isImproved = True
        else:
            self.isImproved = False

        if self.last_best is None or ((self.last_best - self.child_fitness[0]) > 0 and self.cov):
            if self.last_best is not None:
                self.fit_ablt[0] += self.last_best - self.child_fitness[0]
                self.fit_ablt[1] += 1
                self.Improved_100p = self.Improved_100p_50 = 0
            else:
                self.last_best = self.child_fitness[0]
        elif self.isImproved:
            self.fit_ablt[0] += self.last_best - self.child_fitness[0]
            self.fit_ablt[1] += 1

        elif self.progs[0].length >= ((2 ** (self.depth_limit)) - 1) * 0.5:
            self.Improved_100p += 1

            if self.Improved_100p >= 5 and self.depth_limit < PyGP.DEPTH_MAX:
                self.depth_limit += 1
                self.Improved_100p = self.Improved_100p_50 = 0
                self.fit_ablt = [0, 1]
        elif not self.isImproved:
            self.Improved_100p_50 += 1
            if self.Improved_100p_50 >= 10 and self.depth_limit < PyGP.DEPTH_MAX:
                self.depth_limit += 1
                self.Improved_100p = self.Improved_100p_50 = 0
                self.fit_ablt = [0, 1]


        if self.last_best > self.child_fitness[0]:
            self.last_best = self.child_fitness[0]
        self.cov = False
        end = time.time()
        times.append(end - start)
        start = time.time()
        self.cur_R2 = self.child_R2.copy()
        self.cur_fitness = self.child_fitness.copy()
        self.cur_prlt = self.child_prlt.copy()

        end = time.time()
        times.append(end - start)

        # self.pprogs = [self.progs[i].copy(i) for i in range(len(self.progs))]
        self.pprogs = [prog_list[x] for i, x in enumerate(fit_slt)]
        for i in range(len(self.pprogs)):
            self.pprogs[i].setId(i)
        # print('selection time: ', times, len(fit_slt))
        # if (np.array(times) > 1).any():
        #     assert(0 == 1)

    def get_rk_posi(self, idx):
        if self.rk_posi[idx] >= self.pop_size:
            return self.rk_posi[idx] - self.pop_size
        else:
            while(self.rk_posi[idx] < self.pop_size):
                idx += 1
            return self.rk_posi[idx] - self.pop_size

    def initDataset(self, array, fitness, rg = None):
        self.dataset = array
        if isinstance(fitness, np.ndarray):
            self.fitness = fitness.tolist()
        elif isinstance(fitness, list):
            self.fitness = fitness
        else:
            raise ValueError('fitness array can only support for numpy and list, not %d' % fitness.dtype)
        self.cur_fitness = None
        self.cur_prlt = None
        self.child_fitness = np.empty(self.pop_size).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
        self.child_prlt = np.empty(self.pop_size).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
        self.n_terms = len(self.dataset)
        self.dataset_len = len(fitness)
        self.data_rg = rg
        for i in range(self.n_terms):
            if len(self.dataset[i]) != len(fitness):
                raise ValueError(
                    "the dataset size should be equal to the fitness size, where dataset size is %d, and fitness size is %d",
                    len(self.dataset[i]), len(fitness))

        avr_fitness = np.mean(self.fitness)
        self.var = np.dot(np.array(self.fitness) - avr_fitness, np.array(self.fitness) - avr_fitness) / len(self.fitness)

        self.subdataset_size = -1

    def changeDataset(self, array, fitness):
        self.dataset = array
        if isinstance(fitness, np.ndarray):
            self.fitness = fitness.tolist()
        elif isinstance(fitness, list):
            self.fitness = fitness
        else:
            raise ValueError('fitness array can only support for numpy and list, not %d' % fitness.dtype)
        self.n_terms = len(self.dataset)
        self.dataset_len = len(fitness)
        self.t_dataset = None
        for i in range(self.n_terms):
            if len(self.dataset[i]) != len(fitness):
                raise ValueError(
                    "the dataset size should be equal to the fitness size, where dataset size is %d, and fitness size is %d",
                    len(self.dataset[i]), len(fitness))

        self.subdataset_size = -1
        self.cashClear()
        self.cashReset()
        self.semantics=self.PROC_MANAGER.PopSemantic(data_rg=self.data_rg, base_dict=dill.dumps(self.getBasecore()), seed=self.seed)# self.semantics.renew(data_rg=self.data_rg, base_dict=dill.dumps(self.getBasecore()), seed=self.seed)# 
        self.cur_fitness = self.verify(self.dataset, self.fitness, 2)

    def cashClear(self):
        list(map(lambda x: PyGP.cashClear_prog(x), self.progs))
        list(map(lambda x: PyGP.cashClear_prog(x), self.pprogs))
        for i in range(self.pop_size):
            self.progs[i].cash_size = self.cash_size
            self.pprogs[i].cash_size = self.cash_size
        self.CASH_MANAGER.collectReleaseNodes([])
        self.cuda_mem_manager.cash_clear()
    def cashPrepare(self):
        cash_collect = []
        id_collect = {}
        cash_perprog = self.CASH_MANAGER.remainSize() / self.pop_size
        record = 0
        record_1 = 0
        if len(self.pprogs) > 0:
            for i in range(self.pop_size):
                self.pprogs[i].seman_sign = -1
        if PyGP.CASH_OPEN:
            for i in range(self.pop_size):
                self.progs[i].childSizeRenew()
                record_1 += cashUpdate(self.progs[i], self.CASH_MANAGER, id_collect, i)
                record += cashGenerate(self.progs[i], self.CASH_MANAGER, cash_collect, cash_perprog)
            cash_perprog = self.CASH_MANAGER.remainSize() / self.pop_size
            if len(self.pprogs) > 0:
                for i in range(self.pop_size):
                    self.pprogs[i].seman_sign = -1
                    record_1 += cashUpdate(self.pprogs[i], self.CASH_MANAGER, id_collect, i + self.pop_size)
                    record += cashGenerate(self.pprogs[i], self.CASH_MANAGER, cash_collect, cash_perprog)
        if self.iter_id % PyGP.CASH_SZIE_MULTI == 1:
            if not PyGP.CASH_OPEN:
                for i in range(self.pop_size):
                    record_1 += cashUpdate(self.progs[i], self.CASH_MANAGER, id_collect, i)
                if len(self.pprogs) > 0:
                    for i in range(self.pop_size):
                        self.pprogs[i].seman_sign = -1
                        record_1 += cashUpdate(self.pprogs[i], self.CASH_MANAGER, id_collect, i + self.pop_size)
            self.ID_MANAGER.collect(list(id_collect.keys()))
            # self.CASH_MANAGER.collectReleaseNodes(cash_collect)

    def prepare(self, option=0):

        dt_clts = DataCollects()
        e_iposi = [0]
        e_clts = []
        cvals = []
        id_altr = [self.n_terms + 1 + self.pop_size + self.CASH_MANAGER.getCashSize(1) * 2]

        if len(self.backfuncs) > 0 and option == 0: 
            self.backfuncs = []
            self.semantic_data = {}
        s_clts_ = []
        s_t = time.time()
        snodes_num = 0
        bpinfos_ = []
        for i in range(self.pop_size):
            if option == 0:  # execution
                (e_clts_, s_clts, bpinfos) = dt_clts(self.progs[i], id_altr, cvals, True, True)
                if PyGP.SEMANTIC_SIGN:
                    s_clts_.append(s_clts)
                    bpinfos.bfuncs_reverse()
                    bpinfos_.append(bpinfos)
            elif option == 1:  # verify
                (e_clts_, s_clts) = dt_clts(self.progs[i], id_altr, cvals, False, False)
            elif option == 2:  # verify, pprogs
                (e_clts_, s_clts) = dt_clts(self.pprogs[i], id_altr, cvals, False, False)
            e_clts.extend(e_clts_)
            e_iposi.append(len(e_clts))
            snodes_num += len(s_clts.s_nodes)
        s_smt = dill.dumps(s_clts_)
        # print('d_clts: ', time.time() - s_t)
        self.semantics.extend(s_smt)
        if option == 0:
            return (e_clts, e_iposi, id_altr[0], cvals, bpinfos_)
        else:
            return (e_clts, e_iposi, id_altr[0], cvals)

    def verify(self, dataset, fitness, type=1, exp_check=None, inverse_transform=True): 
        from .data_funcs import sc_y
        global sc_y
        verify_fitness = np.empty(self.pop_size).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
        dataset_len = len(fitness)
        exp_attr = self.prepare(type)
        (subdataset_size, input_gpu, input_pitch) = self.cuda_mem_manager.input_alloc(-1, exp_attr[2],
                                                                                          dataset_len) 
        if exp_check is not None:
            if exp_check != exp_attr[2]:
                print(len(exp_check), len(exp_attr[2]))
                print(exp_check)
                print(exp_attr[2])
                assert (0 == 1)
        exps_gpu = self.cuda_mem_manager.exp_alloc(len(exp_attr[0]) * 4)
        initposi_gpu = self.cuda_mem_manager.initposi_alloc(
            len(exp_attr[1]) * 4)  # host_to_gpu(np.array(exp_attr[1]), len(exp_attr[1]) * 4)
        output_gpu = self.cuda_mem_manager.output_alloc(dataset_len * PyGP.DATA_TYPE)
        fitness_gpu = self.cuda_mem_manager.fitness_alloc(self.pop_size * PyGP.DATA_TYPE)
        const_vals_gpu = self.cuda_mem_manager.const_alloc(len(exp_attr[3]) * PyGP.DATA_TYPE)

        self.cuda_mem_manager.host2device(np.array(fitness, dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64), output_gpu)
        
        self.cuda_mem_manager.host2device(np.array(exp_attr[3], dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64), const_vals_gpu)
        self.cuda_mem_manager.host2device(np.array(exp_attr[0], dtype=np.int32), exps_gpu)
        self.cuda_mem_manager.host2device(np.array(exp_attr[1], dtype=np.int32), initposi_gpu)

        exe_record = cuda.Event()
        eval_record = cuda.Event()
        streams = [cuda.Stream() for i in range(4)]

        iter_time = int(dataset_len / subdataset_size)
        info = Info(PyGP.BATCH_NUM, subdataset_size, self.funcs.max_arity(), self.pop_size, dataset_len)
        eval_info = EvalParaInfo(subdataset_size, dataset_len)

        execution_GPU = mod.get_function("execution_GPU")
        evaluation_GPU = mod.get_function("evaluation_GPU_2")
        t_dataset = PyGP.dataset_transform(iter_time, dataset, subdataset_size)
        for i in range(iter_time):
            info.set_iterid(i)
            buffer = [i % 2, self.CASH_MANAGER.getCashSize(1), self.CASH_MANAGER.getCashPosi()]
            info.set_buffer(buffer)
            eval_info.set_batch_idx(i - 1)
            eval_info.set_batch_idx(i - 1)
            if i > 0:
                eval_info.set_offset(input_pitch * (self.n_terms + 1), subdataset_size * (i - 1))

                #for normalize
                if inverse_transform:
                    output = np.empty(subdataset_size * self.pop_size).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
                    self.cuda_mem_manager.memcopy_2D(output,
                                                     subdataset_size * PyGP.DATA_TYPE,
                                                     input_gpu, input_pitch,
                                                     subdataset_size * PyGP.DATA_TYPE, self.pop_size,
                                                     cuda.Stream(), 0,
                                                     src_y_offset=int(self.n_terms + 1))
                    autoinit.context.synchronize()
                    output = sc_y.inverse_transform(output.reshape(-1,1))
                    self.cuda_mem_manager.memcopy_2D(
                                                     input_gpu, input_pitch,
                                                     output,
                                                     subdataset_size * PyGP.DATA_TYPE,
                                                     subdataset_size * PyGP.DATA_TYPE, self.pop_size,
                                                     streams[2], 0,
                                                     dst_y_offset=int(self.n_terms + 1))


                evaluation_GPU(output_gpu, input_gpu, fitness_gpu, np.int64(input_pitch),
                               cuda.In(np.array(eval_info.get_tuple(), np.int32)),
                               block=(128, 1, 1), grid=(self.pop_size, 1, 1), stream=streams[2], shared=128 * PyGP.DATA_TYPE)

                eval_record.record(streams[2])

            streams[i % 2].wait_for_event(exe_record)
            self.cuda_mem_manager.memcopy_2D(input_gpu, input_pitch,
                                             t_dataset[0][i], subdataset_size * PyGP.DATA_TYPE,
                                             subdataset_size * PyGP.DATA_TYPE, self.n_terms, streams[i % 2], 0)
            # print(self.n_terms * self.subdataset_size, self.n_terms, self.subdataset_size)

            if i > 0:  
                streams[i % 2].wait_for_event(eval_record)
            execution_GPU(exps_gpu, initposi_gpu, input_gpu, np.int64(input_pitch),
                          cuda.In(np.array(info.get_tuple(), dtype=np.int32)), const_vals_gpu,
                          block=(int(self.subpop_size * 32), 1, 1),
                          grid=(int((self.pop_size / self.subpop_size) * PyGP.BATCH_NUM), 1, 1),
                          stream=streams[i % 2])

            exe_record.record(streams[i % 2])
            streams[2].wait_for_event(exe_record)

        streams[2].wait_for_event(exe_record)

        eval_info.set_batch_idx(iter_time - 1)
        eval_info.set_offset(input_pitch * (self.n_terms + 1), subdataset_size * (iter_time - 1))
        #
        # # for normalize
        # if inverse_transform:
        output = np.empty(subdataset_size * self.pop_size).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
        self.cuda_mem_manager.memcopy_2D(output,
                                         subdataset_size * PyGP.DATA_TYPE,
                                         input_gpu, input_pitch,
                                         subdataset_size * PyGP.DATA_TYPE, self.pop_size,
                                         cuda.Stream(), 0,
                                         src_y_offset=int(self.n_terms + 1))
        autoinit.context.synchronize()


        # for normalize
        if inverse_transform:
            output = sc_y.inverse_transform(output.reshape(-1,1))
            output = output.reshape(1, -1).squeeze()
            self.cuda_mem_manager.memcopy_2D(
                input_gpu, input_pitch,
                output,
                subdataset_size * PyGP.DATA_TYPE,
                subdataset_size * PyGP.DATA_TYPE, self.pop_size,
                streams[2], 0,
                dst_y_offset=int(self.n_terms + 1))
            autoinit.context.synchronize()

        fit_cpu = []
        output = output.reshape(self.pop_size, -1)
        # print(output.shape)
        for i in range(self.pop_size):
            fit_cpu.append(np.sqrt(np.dot(output[i] - fitness, output[i] - fitness) / len(fitness)))


        # evaluation_GPU(output_gpu, input_gpu, fitness_gpu, np.int64(input_pitch),
        #                cuda.In(np.array(eval_info.get_tuple(), np.int32)),
        #                block=(128, 1, 1), grid=(self.pop_size, 1, 1), stream=streams[2], shared=128 * PyGP.DATA_TYPE)

        # cuda.memcpy_dtoh_async(verify_fitness, fitness_gpu, stream=streams[2])
        autoinit.context.synchronize()

        # R2
        avr_fitness = np.mean(fitness)
        var = np.dot(np.array(fitness) - avr_fitness, np.array(fitness) - avr_fitness) / len(fitness)
        # R2 = 1 - (verify_fitness / var)
        fit_cpu = np.array(fit_cpu)
        R2 = 1 - (fit_cpu ** 2 / var)

        # test_printf = PyGP.test_module.test_mod.get_function("printf_")
        # test_printf(input_gpu, np.int32(input_pitch * (self.n_terms + 1)), np.int32(50), np.int32(input_pitch),
        #             block=(1, 1, 1), grid=(1, 1, 1))
        # autoinit.context.synchronize()
        # test_printf(input_gpu, np.int32(input_pitch * (0)), np.int32(50), np.int32(input_pitch),
        #             block=(1, 1, 1), grid=(1, 1, 1))
        # autoinit.context.synchronize()
        return (np.sqrt(fit_cpu ** 2), R2, exp_attr[2], fit_cpu)

    def execution(self, sign=-1):

        start = time.time()
        end = time.time()
        # print('backpSelect: ', end - start)

        start = time.time()
        self.cashPrepare()
        end = time.time()
        # print('execution prepare 0 : ', end - start)
        start = time.time()
        exp_attr = self.prepare()
        end = time.time()
        # print('execution prepare 1 : ', end - start)

        # print("execution random2:  ", [random.randint(0, 100) for i in range(10)], np.random.randint(0, 100))
        start = time.time()
        d_len = self.dataset_len
        c_size = self.CASH_MANAGER.getCashSize(1)
        c_mng = self.cuda_mem_manager
        # print(exp_attr[1][0:100])

        end = time.time()
        # print('execution prepare 2: ', end - start)
        start = time.time()

        # 数据初始化
        exe_record = cuda.Event()
        eval_record = cuda.Event()
        semsave_record = cuda.Event()
        streams = [cuda.Stream() for i in range(4)]

        height = int(exp_attr[2])

        if PyGP.SEMANTIC_SIGN:
            bp_infos = exp_attr[4]
            s_nodes = self.semantics.snode_merge()
            (s_bfs, bfs_posi) = PyGP.bfuncs_merge(bp_infos)
            semsave_cpu = np.empty(len(s_nodes) * d_len).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
            tsematic_cpu = np.empty((len(bfs_posi) - 1) * d_len).astype(np.float64)
            tderivate_cpu = np.empty((len(bfs_posi) - 1) * d_len).astype(np.float64)

        (self.subdataset_size, input_gpu, input_pitch) =\
            c_mng.input_alloc(-1, height, d_len)  

        sd_len = self.subdataset_size

        pearson_gpu = c_mng.fitness_alloc(self.pop_size * PyGP.DATA_TYPE)
        exps_gpu = c_mng.exp_alloc(len(exp_attr[0]) * 4)
        initposi_gpu = c_mng.initposi_alloc(len(exp_attr[1]) * 4)
        output_gpu = c_mng.output_alloc(d_len * PyGP.DATA_TYPE)
        fitness_gpu = c_mng.fitness_alloc(self.pop_size * PyGP.DATA_TYPE)
        cvals_gpu = c_mng.const_alloc(len(exp_attr[3]) * PyGP.DATA_TYPE)

        # if ((any(np.isnan(exp_attr[3])) or any(np.isinf(exp_attr[3])))):
        #     print(exp_attr[3])
        #     print(any(np.isnan(exp_attr[3])), any(np.isinf(exp_attr[3])))
        #     assert (0 == 1)
        c_mng.host2device(np.array(self.fitness).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64), output_gpu)
        c_mng.host2device(np.array(exp_attr[3]).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64), cvals_gpu)
        c_mng.host2device(np.array(exp_attr[0]).astype(np.int32), exps_gpu)
        c_mng.host2device(np.array(exp_attr[1]).astype(np.int32), initposi_gpu)

        if PyGP.SEMANTIC_SIGN and len(s_bfs) > 0:
            s_exp_gpu = cuda.mem_alloc(len(s_bfs) * 4)
            b_posi_gpu = cuda.mem_alloc(len(bfs_posi) * 4)
            results_gpu = cuda.mem_alloc((len(bfs_posi) - 1) * d_len * 8)
            derivate_gpu = cuda.mem_alloc((len(bfs_posi) - 1) * d_len * 8)

            cuda.memcpy_htod(s_exp_gpu, np.array(s_bfs).astype(np.int32))
            cuda.memcpy_htod(b_posi_gpu, np.array(bfs_posi).astype(np.int32))

            fit_gpu = cuda.mem_alloc(d_len * 8)
            c_mng.host2device(np.array(self.fitness).astype(np.float64), fit_gpu)

            for i in range(len(bfs_posi) - 1):
                cuda.memcpy_dtod(int(results_gpu) + d_len * 8 * i, fit_gpu,
                                 d_len * 8) 


        iter_time = int(d_len / sd_len)
        info = Info(PyGP.BATCH_NUM, sd_len, self.funcs.max_arity(), self.pop_size, d_len)
        eval_info = EvalParaInfo(sd_len, d_len)

        c_mng.cash_alloc(iter_time * self.pop_size * self.cash_size)
        cash_gpu_attr = c_mng.get_cash_attr()

        if self.CASH_MANAGER.getCashSize(0) > 0:
            c_mng.memcopy_2D(
                input_gpu, input_pitch,
                cash_gpu_attr[0], cash_gpu_attr[1],
                cash_gpu_attr[1],
                self.CASH_MANAGER.getCashSize(0),
                streams[0], 1,
                dst_y_offset=self.n_terms + 1 + self.pop_size,
            )
            eval_record.record(streams[0])

        if iter_time > 1:
            c_mng.memcopy_2D(
                input_gpu, input_pitch,
                cash_gpu_attr[0], cash_gpu_attr[1],
                cash_gpu_attr[1],
                self.CASH_MANAGER.getCashSize(0),
                streams[1], 2,
                dst_y_offset=self.n_terms + 1 + self.pop_size + c_size,
                src_y_offset=c_size)

        execution_GPU = mod.get_function("execution_GPU")
        evaluation_GPU = mod.get_function("evaluation_GPU")
        prlt_GPU = mod.get_function("pearson_rlt_GPU")
        backpropagation_GPU = mod.get_function("backpropagation")
        derivative_GPU = mod.get_function("derivative")

        # test_printf = PyGP.test_module.test_mod.get_function("printf_")
        # print(self.fitness[0:10])

        # print('iter_time: ', iter_time)

        if self.t_dataset is None or self.t_dataset[1] != sd_len:
            self.t_dataset = PyGP.dataset_transform(iter_time, self.dataset, sd_len)
        # print(iter_time, len(self.dataset), len(self.dataset[0]))
        assert (iter_time == 1)
        for i in range(iter_time):
            info.set_iterid(i)
            # print("iter_time: ", iter_time, d_len, sd_len, input_pitch, len(self.t_dataset[0][0]), d_len)
            buffer = [i % 2, c_size, self.CASH_MANAGER.getCashPosi()]
            info.set_buffer(buffer)
            eval_info.set_batch_idx(i - 1)
            if i > 0:
                eval_info.set_offset(input_pitch * (self.n_terms + 1), sd_len * (i - 1))
                evaluation_GPU(output_gpu, input_gpu, fitness_gpu, np.int64(input_pitch),
                               cuda.In(np.array(eval_info.get_tuple(), np.int32)),
                               block=(128, 1, 1), grid=(self.pop_size, 1, 1), stream=streams[2], shared=128 * PyGP.DATA_TYPE)
                eval_record.record(streams[2])

            streams[i % 2].wait_for_event(exe_record)
            c_mng.memcopy_2D(input_gpu, input_pitch,
                                             self.t_dataset[0][i], sd_len * PyGP.DATA_TYPE,
                                             sd_len * PyGP.DATA_TYPE, self.n_terms, streams[i % 2], 0)

            if i > 0: 
                streams[i % 2].wait_for_event(eval_record)
                streams[i % 2].wait_for_event(semsave_record)
            execution_GPU(exps_gpu, initposi_gpu, input_gpu, np.int64(input_pitch),
                          cuda.In(np.array(info.get_tuple(), dtype=np.int32)), cvals_gpu,
                          block=(int(self.subpop_size * 32), 1, 1),
                          grid=(int((self.pop_size / self.subpop_size) * PyGP.BATCH_NUM), 1, 1),
                          stream=streams[i % 2])
            # autoinit.context.synchronize()

            exe_record.record(streams[i % 2])
            streams[2].wait_for_event(exe_record)

            end = time.time()
            # print('execution: ', end - start)

            start = time.time()
            if PyGP.SEMANTIC_SIGN and len(s_bfs) > 0:
                streams[3].wait_for_event(exe_record)
                for j in range(len(s_nodes)):
                    if s_nodes[j] >= 0:
                        c_mng.memcopy_2D(semsave_cpu,
                                         sd_len * PyGP.DATA_TYPE,
                                         input_gpu, input_pitch,
                                         sd_len * PyGP.DATA_TYPE, 1,
                                         streams[3], 0,
                                         src_y_offset=s_nodes[j],
                                         dst_y_offset=j * iter_time + i)  
                    else:
                        offset = (j * iter_time + i) * sd_len
                        semsave_cpu[offset: offset + sd_len] = exp_attr[3][-s_nodes[j] - 1]
                semsave_record.record(streams[3])

                backpropagation_GPU(s_exp_gpu, b_posi_gpu, results_gpu, derivate_gpu, input_gpu,
                                    np.int64(input_pitch), cuda.In(np.array(info.get_tuple(), dtype=np.int32)),
                                    cvals_gpu, np.int64(len(bfs_posi) - 1),
                                    block=(int(self.subpop_size * 32), 1, 1),
                                    grid=(int(self.pop_size / self.subpop_size), 1, 1),
                                    stream=streams[i % 2])
                # derivative_GPU(s_exp_gpu, b_posi_gpu, derivate_gpu, input_gpu,
                #                     np.int32(input_pitch), cuda.In(np.array(info.get_tuple(), dtype=np.int32)),
                #                     cvals_gpu, np.int32(len(bfs_posi) - 1),
                #                     block=(int(self.subpop_size * 32), 1, 1),
                #                     grid=(int(self.pop_size / self.subpop_size), 1, 1),
                #                     stream=streams[i % 2])

                # ffuncs_ds = PyGP.ffuncs_d_clts(bp_infos)
                # for j in range(len(ffuncs_ds)):
                #     if len(ffuncs_ds[j]) >= 0:
                #         for z in range(len(ffuncs_ds[j])):
                #             if ffuncs_ds[j][z][1] >= 0:
                #                 ffuncs_d = np.empty(sd_len).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
                #                 c_mng.memcopy_2D(ffuncs_d,
                #                                  sd_len * PyGP.DATA_TYPE,
                #                                  input_gpu, input_pitch,
                #                                  sd_len * PyGP.DATA_TYPE, 1,
                #                                  streams[3], 0,
                #                                  src_y_offset=ffuncs_ds[j][z][1])
                #                 self.semantics.ffuncs_d_set(ffuncs_ds[j][z][1], ffuncs_d, i)
                    # elif i == 0:
                    #     self.semantics.ffuncs_d_set(ffuncs_ds[j], exp_attr[3][-ffuncs_ds[j] - 1], i)

            c_mng.memcopy_2D(cash_gpu_attr[0],
                             cash_gpu_attr[1],
                             input_gpu, input_pitch,
                             cash_gpu_attr[1],
                             self.CASH_MANAGER.getCashSize(0),
                             streams[i % 2], 1,
                             dst_y_offset=c_size * i,
                             src_y_offset=self.n_terms + 1 + self.pop_size +c_size * (i % 2))

            if i < iter_time - 2:
                c_mng.memcopy_2D(
                    input_gpu, input_pitch,
                    cash_gpu_attr[0], cash_gpu_attr[1],
                    cash_gpu_attr[1],
                    self.CASH_MANAGER.getCashSize(0),
                    streams[i % 2], 2,
                    dst_y_offset=self.n_terms + 1 + self.pop_size + c_size * (i % 2),
                    src_y_offset=c_size * (i + 2))

        streams[2].wait_for_event(exe_record)

        if PyGP.SEMANTIC_SIGN:
            cuda.memcpy_dtoh_async(tsematic_cpu, results_gpu, stream=streams[2])
            cuda.memcpy_dtoh_async(tderivate_cpu, derivate_gpu, stream=streams[2])

        eval_info.set_batch_idx(iter_time - 1)
        eval_info.set_offset(input_pitch * (self.n_terms + 1), sd_len * (iter_time - 1))

        # evaluation_GPU(output_gpu, input_gpu, fitness_gpu, np.int64(input_pitch),
        #                cuda.In(np.array(eval_info.get_tuple(), np.int32)),
        #                block=(128, 1, 1), grid=(self.pop_size, 1, 1), stream=streams[2], shared=128 * PyGP.DATA_TYPE)

        # cuda.memcpy_dtoh_async(self.child_fitness, fitness_gpu, stream=streams[2])

        output = np.empty(d_len * self.pop_size).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
        self.cuda_mem_manager.memcopy_2D(output,
                                         d_len * PyGP.DATA_TYPE,
                                         input_gpu, input_pitch,
                                         d_len * PyGP.DATA_TYPE, self.pop_size,
                                         cuda.Stream(), 0,
                                         src_y_offset=int(self.n_terms + 1))
        autoinit.context.synchronize()
        output = output.reshape(self.pop_size, -1)
        self.child_R2 = []
        for i in range(self.pop_size):
            self.child_fitness[i] = np.sqrt(np.dot(output[i] - self.fitness, output[i] - self.fitness) / len(self.fitness))

            # R2
            avr_fitness = np.mean(output[i])
            var = np.dot(np.array(self.fitness) - avr_fitness, np.array(self.fitness) - avr_fitness) / len(self.fitness)
            R2 = 1 - (self.child_fitness[i] ** 2 / var)
            self.child_R2.append(R2)
        # autoinit.context.synchronize()
        # print("child_fitness: ", self.child_fitness)

        # XYZ = cuda.mem_alloc(eval_info.dataset_size * 3 * PyGP.DATA_TYPE * self.pop_size)
        # prlt_GPU(output_gpu, input_gpu, pearson_gpu, np.int64(input_pitch),
        #          cuda.In(np.array(eval_info.get_tuple(), np.int32)), XYZ,
        #          block=(128, 1, 1), grid=(self.pop_size, 1, 1), stream=streams[2])

        # cuda.memcpy_dtoh_async(self.child_prlt, pearson_gpu, stream=streams[2])

        # autoinit.context.synchronize()
        # print("child_prlt: ", self.child_prlt)
        end = time.time()
        # print('semantic execution: ', end - start)
        # print('tnode_again:')
        # print('self.fitness[0]: ', self.fitness[0])
        # print('self.x0: ', self.dataset)
        # self.semantics.library[46][0][1].exp_draw()
        # node = self.semantics.get_tg_node(46)
        # node.exp_draw()
        # offset = self.semantics.semantics[46].get_snode_idx(node)
        # # semsave_gpu = cuda.mem_alloc(len(exp_attr[8]) * self.dataset_len * 4)
        # # cuda.memcpy_htod(semsave_gpu, semsave_cpu)
        # # print(semsave_cpu[sd_len * offset: sd_len * (offset + 1)])
        # # print(len(s_nodes))
        # print(bfs_posi[0], bfs_posi[1], s_bfs[bfs_posi[0]:bfs_posi[1]])
        # if len(s_nodes) > 0:
        #     print(len(s_nodes), s_nodes[offset: offset+100])
        #     if s_nodes[offset] > 0:
        #         test_printf(input_gpu, np.int32(input_pitch * s_nodes[offset]), np.int32(sd_len), np.int32(input_pitch), block=(1, 1, 1), grid=(1, 1, 1), stream=streams[2])
        #
        # autoinit.context.synchronize()
        # test_printf(derivate_gpu, np.int32(46 * 4 * sd_len), np.int32(sd_len), np.int32(input_pitch),
        #             block=(1, 1, 1), grid=(1, 1, 1), stream=streams[2])
        # autoinit.context.synchronize()
        #
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!")
        autoinit.context.synchronize()

        # self.child_fitness = fit_cpu#np.sqrt(self.child_fitness)

        if PyGP.SEMANTIC_SIGN and len(s_bfs) > 0:
            start = time.time()
            semsave_cpu = [semsave_cpu[sd_len * i:sd_len * (i + 1)] for i in
                                range(len(s_nodes))]
            tsematic_cpu = [tsematic_cpu[d_len * i:d_len * (i + 1)] for i in
                                 range(len(bfs_posi) - 1)]
            # for i in range(len(bfs_posi) - 1):
            #     if (np.isnan(tderivate_cpu[d_len * i:d_len * (i + 1)]).any()):
            #         self.progs[i].exp_draw()
            #         node = self.semantics.get_tg_node(i)
            #         node.exp_draw()
            #         np.set_printoptions(threshold=np.inf)
            #         print('tsematic_cpu', tsematic_cpu[i:i + 1])
            #         raise ValueError(self.pop_size / self.subpop_size, i, len(bfs_posi) - 1, tderivate_cpu[d_len * i:d_len * (i + 1)], )
            tderivate_cpu = [tderivate_cpu[d_len * i:d_len * (i + 1)] for i in
                                 range(len(bfs_posi) - 1)]
            # for i, t in enumerate(tderivate_cpu):
            #     if any(np.isnan(t)) or any(np.isinf(t)):
            #         print(t)
            #         self.progs[i].print_exp()
            #         node = self.semantics.get_tg_node(i)
            #         node.exp_draw()
            #         self.progs[i].exp_draw()
            #
            #         test_printf = PyGP.test_module.test_mod.get_function("printf_")
            #         test_printf(input_gpu, np.int32(input_pitch * (self.n_terms + 1 + i)), np.int32(d_len), np.int32(input_pitch),
            #                     block=(1, 1, 1), grid=(1, 1, 1))
            #         autoinit.context.synchronize()
            #         print(self.child_fitness[i])
            #         assert (0 == 1)
            # print(tderivate_cpu)
            # assert (0 == 1)

            # for i, t in enumerate(semsave_cpu):
            #     if any(np.isnan(t)) or any(np.isinf(t)):
            #         print(t)
            #         test_printf = PyGP.test_module.test_mod.get_function("printf_")
            #         test_printf(input_gpu, np.int32(input_pitch * s_nodes[i]), np.int32(d_len), np.int32(input_pitch),
            #                     block=(1, 1, 1), grid=(1, 1, 1))
            #         autoinit.context.synchronize()
            #         print(self.t_dataset[0][0])
            #         print("=============================0=================================")
            #         test_printf(input_gpu, np.int32(0), np.int32(d_len), np.int32(input_pitch),
            #                     block=(1, 1, 1), grid=(1, 1, 1))
            #         autoinit.context.synchronize()
            #         print("=============================1=================================")
            #         test_printf(input_gpu, np.int32(input_pitch), np.int32(d_len), np.int32(input_pitch),
            #                     block=(1, 1, 1), grid=(1, 1, 1))
            #         autoinit.context.synchronize()
            #         # print(self.child_fitness[i])
            #         exps = self.semantics.snode_exp_merge()
            #         assert(len(exps) == len(s_nodes))
            #         print(exps[i])
            #         print(s_nodes[i])
            #         assert (0 == 1)
            avg_fit = 0
            for i in range(len(self.child_fitness)):
                avg_fit += self.child_fitness[i]
            # print('avg_fit: ', avg_fit)
            self.semantics.data_load(tsematic_cpu, semsave_cpu, tderivate_cpu)
            end = time.time()
            # print('semsave_cpu: ', end - start)
            start = time.time()
            self.semantics.smt_clts()
            end = time.time()
            # print('smt_clts: ', end - start)
        self.iter_id += 1
    def getAverSize(self):
        aver_size = 0
        for i in range(self.pop_size):
            aver_size += self.progs[i].length
        aver_size /= self.pop_size
        return aver_size

    def getOutput(self, idx:int, d_len):
        output = np.empty(d_len).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
        (input_gpu, input_pitch) = self.cuda_mem_manager.get_inputgpu()
        self.cuda_mem_manager.memcopy_2D(output,
                                         d_len * PyGP.DATA_TYPE,
                                         input_gpu, input_pitch,
                                         d_len * PyGP.DATA_TYPE, 1,
                                         cuda.Stream(), 0,
                                         src_y_offset=int(self.n_terms + 1 + idx))
        # print("idx: ", idx)
        # test_printf = PyGP.test_module.test_mod.get_function("printf_")
        # test_printf(input_gpu, np.int32(input_pitch * (self.n_terms + 1)), np.int32(d_len), np.int32(input_pitch),
        #             block=(1, 1, 1), grid=(1, 1, 1))
        # test_printf(input_gpu, np.int32(input_pitch * (0)), np.int32(d_len), np.int32(input_pitch),
        #             block=(1, 1, 1), grid=(1, 1, 1))
        # test_printf(input_gpu, np.int32(input_pitch * (1)), np.int32(d_len), np.int32(input_pitch),
        #             block=(1, 1, 1), grid=(1, 1, 1))
        autoinit.context.synchronize()
        return output
    def cashInit(self, cash_mem_size=None):
        cash_size = int(2 * (1024 ** 3) / self.dataset_len / PyGP.DATA_TYPE)
        if cash_mem_size is None:
            cash_size = int(2 * (1024 ** 3) / self.dataset_len / PyGP.DATA_TYPE)
            if cash_size > PyGP.CASHNUM_PERBLOCK * self.pop_size * PyGP.CASH_SZIE_MULTI:
                cash_size = PyGP.CASHNUM_PERBLOCK * self.pop_size * PyGP.CASH_SZIE_MULTI
        else:
            cash_size = int(cash_mem_size / self.dataset_len / PyGP.DATA_TYPE)
        if self.CASH_MANAGER is None:
            super()._cash_set(self.n_terms + self.pop_size + 1, cash_size, self.seed)

        height = self.pop_size * (self.dataset_len / self.subdataset_size + 1) * self.CASH_MANAGER.getCashSize(0)

        # self.cuda_mem_manager.cash_alloc(height)

    def cashReset(self, cash_mem_size=None):
        cash_size = int(2 * (1024 ** 3) / self.dataset_len / PyGP.DATA_TYPE)
        if cash_mem_size is None:
            cash_size = int(2 * (1024 ** 3) / self.dataset_len / PyGP.DATA_TYPE)
            if cash_size > PyGP.CASHNUM_PERBLOCK * self.pop_size:
                cash_size = PyGP.CASHNUM_PERBLOCK * self.pop_size
        else:
            cash_size = int(cash_mem_size / self.dataset_len / PyGP.DATA_TYPE)
        super()._cash_set(self.n_terms + self.pop_size + 1, cash_size)
        self.cash_size = int(self.CASH_MANAGER.getCashSize(1) / self.pop_size)

    def gpu_memalloc(self):
        raise NotImplementedError

    def preExecution(self):

        raise NotImplementedError

