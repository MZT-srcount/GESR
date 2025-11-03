import ctypes
import time
import PyGP
from PyGP import Base, SManager
import dill
import numpy as np
import random


class FuncCall:
    def __init__(self, gname, func, pop, *args):
        self.oper = func(*args)
        self.func = func
        self.args = args
        self.gname = gname
        self.pop = pop
    
    def return_callback(self, res):
        for j in res:
            if j != -1:
                if(j == 0):
                    print("HEREEEEEEE")
                self.pop.progs[j] = PyGP.unzip(self.pop.sharedmemory[j])

    def __call__(self, *args):
        self.oper = self.func(*self.args)
        if not hasattr(self.oper, 'preprocess'):
            raise NotImplementedError("%s not implement the preprocess yet")
        seqs = self.oper.preprocess()
        split_seqs = np.array_split(seqs, PyGP.PARALLEL)
        mp_progs = []
        base_dict = dill.dumps(self.pop.getBasecore())
        self.pop.parallel_prepare()
        seeds_tmp = self.pop.seedseq.spawn(PyGP.PARALLEL)#np.random.randint(2 ** 31, size=PyGP.PARALLEL)
        
        for i in range(1, PyGP.PARALLEL, 1):
            shared_list = []
            for j in split_seqs[i]:
                progs_zip = self.pop.progs[j].zip()
                self.pop.sharedmemory[j][:len(progs_zip)] = progs_zip
                shared_list.append(j)
            mp_progs.append(self.pop.pool.apply_async(self.oper.__call__, args=(shared_list, seeds_tmp[i], self.pop.semantics, *args), callback=self.pop.return_callback))
        new_progs = self.oper([self.pop.progs[i] for i in split_seqs[0]], seeds_tmp[0], self.pop.semantics, *args)
        
        for i, x in enumerate(split_seqs[0]):
            self.pop.progs[x] = new_progs[i]
        for i in range(0, PyGP.PARALLEL - 1):
            mp_progs[i].wait()
        for i in range(self.pop.pop_size):
            self.pop.progs[i].setId(i)
        self.pop.parallel_restore()


class BaseOperator:
    def __init__(self):
        NotImplementedError("__init__ function should be implemented first")

    def preprocess(self):
        seq = list(range(self.pop_size))
        return seq

    def run(self):
        NotImplementedError("crossover function should be implemented first")

    def __call__(self, pprogs, rd_state, semantics, *args):
        # print('!!!rd_state', rd_state, type(rd_state))
        self.semantics = SManager(semantics)
        self.rg = np.random.default_rng(rd_state)
        if isinstance(pprogs[0], np.bytes_):
            pprogs = [dill.loads(prog) for prog in pprogs]
            progs = self.run(pprogs, *args)
        elif isinstance(pprogs[0], PyGP.Program):
            progs = self.run(pprogs, *args)
            for i in range(len(progs)):
                if progs[i] is None:
                    progs[i] = pprogs[i]
            return progs
        elif isinstance(pprogs[0], np.int32) or isinstance(pprogs[0], np.int64):
            progs = [PyGP.unzip(PyGP.sharedList[idx]) for idx in pprogs]
            progs = self.run(progs, *args)
            for i in range(len(pprogs)):
                if progs[i] is not None:
                
                    if pprogs[i] == 0:
                        print("pprogs: ", pprogs[i])
                    prog_zip = progs[i].zip()
                    if pprogs[i] >= PyGP.POP_SIZE or len(prog_zip) > PyGP.MAX_SIZE * 6:
                        raise ValueError(pprogs[i], len(prog_zip))
                    PyGP.sharedList[pprogs[i]][:len(prog_zip)] = prog_zip
                else:
                    pprogs[i] = -1
            return pprogs
        else:
            raise NotImplementedError
        return []