import ctypes
import time
import PyGP
from PyGP import Base, SManager
import dill
import numpy as np
import random
class BaseCrossover:
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
        # if pprogs[0] == 10:
        #     print(PyGP.sharedList[10][:100])
        #     assert (0 == 1)
        # Base.__dict__.update(property)
        
        # self.rd = dill.loads(rd_state)
        # self.rd = np.random.RandomState(rd_state)
        # print("here..!..!...", type(pprogs[0]))
        if isinstance(pprogs[0], np.bytes_):
            pprogs = [dill.loads(prog) for prog in pprogs]
            progs = self.run(pprogs, *args)
            # progs = [dill.dumps(prog) for prog in progs]
        elif isinstance(pprogs[0], PyGP.Program):
            # print('0-rd_state', rd_state)
            # print('0-rd_state-f', rd_state)
            progs = self.run(pprogs, *args)
            for i in range(len(progs)):
                if progs[i] is None:
                    progs[i] = pprogs[i]
            return progs
        elif isinstance(pprogs[0], np.int32) or isinstance(pprogs[0], np.int64):
            # print('1-rd_state', rd_state)
            # print('1-rd_state-f', rd_state)
            # np.random.seed(rd_state)
            progs = [PyGP.unzip(PyGP.sharedList[idx]) for idx in pprogs]
            # print("!!!!!here")
            # print(rd_state, type(rd_state))
            # random.seed(int(rd_state))
            # print(rd_state, '------------------')
            progs = self.run(progs, *args)
            for i in range(len(pprogs)):
                if progs[i] is not None:#只有真正修改的才需要返回
                
                    if pprogs[i] == 0:
                        print("pprogs: ", pprogs[i])
                    prog_zip = progs[i].zip()
                    if pprogs[i] >= PyGP.POP_SIZE or len(prog_zip) > PyGP.MAX_SIZE * 6:
                        raise ValueError(pprogs[i], len(prog_zip))
                    PyGP.sharedList[pprogs[i]][:len(prog_zip)] = prog_zip
                else:
                    pprogs[i] = -1

            # avg = 0
            # for i in range(len(progs)):
            #     # print("here...!..")
            #     if progs[i] is not None:
            #         avg += progs[i].length
            # print('avg_size: ', pprogs[0], avg)

            return pprogs
            # print(PyGP.Base.ID_MANAGER.getPool())
            # progs = [prog.zip() for prog in progs]
        else:
            raise NotImplementedError
        return []