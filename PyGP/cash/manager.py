'''
Author: your name
Date: 2023-08-05 16:00:09
LastEditTime: 2023-08-08 16:05:23
LastEditors: your name
Description: 
FilePath: \PyGP\PyGP\cash.py
'''
import multiprocessing
import multiprocessing.managers as manager
import random

import numpy as np
import dill
class SharedManager(manager.BaseManager):
    def _init(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        pass


class IdBase:
    id_iter = 0
    id_collects = []
    id_max = 9e8
    POOL_SIZE = 10000
    ID_MAX = 9e8
    def idSupplement(self):
        if len(self.id_collects) > 0:
            raise NotImplementedError
            id_clts = self.id_collects
            id_rg = None
            self.id_collects = []
        else:
            id = self.id_iter
            id_rg = (id, id + self.POOL_SIZE)
            # id_clts = list(range(id, id+5000, 1))
            id_clts = None
            self.id_iter += self.POOL_SIZE
        return (id_clts, id_rg)
    def collect(self, id_collects):
        self.id_collects.extend(id_collects)
        # for i in range(0, self.id_iter):
        #     if id_collects.get(i) is None:
        #         self.id_collects.append(i)

    def cur_iter(self):
        return self.id_iter

    def fflush_needed(self):
        return self.id_iter > self.id_max

    def reflush(self, id_init, id_max):
        self.id_iter = id_init
        self.id_max = id_max

    def get_poolsize(self):
        return self.POOL_SIZE

    def get_limit(self):
        return self.ID_MAX

class IdAllocator:
    id_pool = []
    id_rg = []
    id_max = 0
    id_iter_s = -1
    lock = multiprocessing.Lock()
    def __init__(self):
        from PyGP import Base
        self.base = Base().PROC_MANAGER.IdBase()
    def idAllocate(self):
        if self.id_iter_s == self.id_max - 1:
            (id_clts, id_rg) = self.base.idSupplement()
            if id_clts is not None:
                self.id_pool.extend(id_clts)
            if id_rg is not None:
                self.id_rg.append(id_rg)
                self.id_max = id_rg[1]
                self.id_iter_s = id_rg[0]
            else:
                raise NotImplementedError
        idx = self.id_iter_s
        self.id_iter_s += 1
        return idx

    def collect(self, id_collects):
        if self.base.fflush_needed():
            print("here.................................")
            cur_minid = min(id_collects)
            self.base.reflush(0, cur_minid)
            if cur_minid < 1e5:
                print(min(id_collects), max(id_collects))
                assert (0 == 1)
                cur_maxid = max(id_collects)
                self.base.reflush(cur_maxid + 2 * self.base.get_poolsize(), self.base.get_limit())

    def poolClear(self):
        self.id_pool = []
        self.id_iter_s = -1
        self.id_max = 0

    def getPool(self):
        return self.id_pool


class CashManager:
    def __init__(self, init_posi, cash_size):
        self.cash_node = {}
        self.cash_counter = {}
        self.free_pointer = []
        self.cash_pointer = 0
        self.init_storeposi = init_posi
        self.cash_size = cash_size

    def update(self, dict_):

        self.__dict__.update(dict_)

    def get_dict(self):#for manager
        return self.__dict__

    def reset(self, init_posi, cash_size):
        self.cash_node = {}
        self.cash_counter = {}
        self.free_pointer = []
        self.cash_pointer = 0
        self.init_storeposi = init_posi
        self.cash_size = cash_size

    def currentSize(self):
        return len(self.cash_counter)

    def getCashPosi(self):
        return self.init_storeposi

    def addCash(self, treeNode):
        if self.free_pointer:
            cash_alloc = self.free_pointer.pop()
        else:
            if(self.cash_pointer >= self.cash_size):
                raise ValueError("cash pointer out of permitted cash size, cash_counter:", len(self.cash_counter), "free_pointer:", len(self.free_pointer), "cash_pointer:", self.cash_pointer, "cash_size: ", self.cash_size)
            cash_alloc = self.cash_pointer
            self.cash_pointer += 1

        if self.cash_node.get(treeNode.node_id) is not None:
            self.releaseNode(treeNode)

        treeNode.changeCashState(2)
        self.cash_node[treeNode.node_id] = cash_alloc
        self.cash_counter[cash_alloc] = 1
        #print('cash_id: ', cash_alloc + self.init_storeposi)
        treeNode.setCashId(cash_alloc + self.init_storeposi)

    def countUp(self, treeNode, node):
        if self.cash_node.get(node.node_id) is not None:
            self.cash_counter[self.cash_node[node.node_id]] += 1
            self.cash_node[treeNode.node_id] = self.cash_node[node.node_id]
            treeNode.setCashId(self.cash_node[node.node_id] + self.init_storeposi)
        else:
            # self.addCash(treeNode)
            raise ValueError("cash pointer out of permitted cash size, cash_counter:", len(self.cash_counter),
                             "free_pointer:", len(self.free_pointer), "cash_pointer:", self.cash_pointer,
                             "cash_size: ", self.cash_size, node.getCashState(), self.cash_node.get(node.node_id))

    def releaseNode(self, treeNode):
        if self.cash_node.get(treeNode.node_id) is None:
            return
        cash_point = self.cash_node[treeNode.node_id]
        self.cash_counter[cash_point] -= 1
        treeNode.changeCashState(0)
        if self.cash_counter[cash_point] == 0:
            self.free_pointer.append(cash_point)
            self.cash_counter.pop(cash_point)
        self.cash_node.pop(treeNode.node_id)
    
    def collectReleaseNodes(self, treeNodes):
        cash_nodes = self.cash_node.copy()
        nodes_record = {}
        for i in range(len(treeNodes)):
            if cash_nodes.get(treeNodes[i].node_id) is not None:
                # print('treeNodes[i].node_id ', treeNodes[i].node_id)
                cash_nodes.pop(treeNodes[i].node_id)
                nodes_record[treeNodes[i].node_id] = 1

                if (treeNodes[i].getCashState() == 2 and self.cash_counter[self.cash_node[treeNodes[i].node_id]] > 1):
                    raise ValueError("cash state can not be 2 here")
            else:
                raise ValueError('sth wrong in collectReleaseNodes, a node not exist', self.cash_node.get(treeNodes[i].node_id), nodes_record.get(treeNodes[i].node_id), treeNodes[i].getCashState(), treeNodes[i].node_id)
        
        # if len(cash_nodes) == len(treeNodes) and len(cash_nodes) != 0:
        #     raise ValueError("it needs deep copy here, cash_nodes&treenodes size: ", len(cash_nodes), "self.cash_nodes.size: ", len(self.cash_node))

        for key, value in cash_nodes.items():

            self.cash_node.pop(key)
            self.cash_counter[value] -= 1
            if self.cash_counter[value] == 0:
                self.cash_counter.pop(value)
                self.free_pointer.append(value)

    def isAvailable(self):
        return self.currentSize() < (self.cash_size - 1)

    def getCashSize(self, type):
        if type == 0: #current max size
            return self.cash_pointer
        elif type == 1: #capacity
            return self.cash_size
        else:
            raise ValueError("Not implement yet")

    def remainSize(self):
        return self.cash_size - self.currentSize() - 2

from PyGP import PopSemantic
SharedManager.register("IdAllocator", IdAllocator)
SharedManager.register("IdBase", IdBase)
SharedManager.register("CashManager", CashManager)
SharedManager.register("list", list)
