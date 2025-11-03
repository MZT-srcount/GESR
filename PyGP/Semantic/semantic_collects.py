import itertools
import random

import numpy as np
import math

import time
import PyGP
from PyGP import LIBRARY_SIZE
import dill


class Semantic:
    def __init__(self):
        self.node = (None, None)
        self.s_backprogs = [] 
        self.s_forwardfuncs = []  

class BPInfos:
    def __init__(self):
        self.semantic = []
    def add_bfuncs(self, exp_unit, id):
        while id >= len(self.semantic):
            self.semantic.append(Semantic())
        self.semantic[id].s_backprogs.append(exp_unit)
    def add_ffuncs(self, exp_unit, id):
        while id >= len(self.semantic):
            self.semantic.append(Semantic())
        self.semantic[id].s_forwardfuncs.append(exp_unit)
    def bfuncs_merge(self):
        flatten_bfs = []
        bfs_posi = []
        for i in range(len(self.semantic)):
            bfuncs = map(lambda x: x[0] + [x[1]], self.semantic[i].s_backprogs)
            flatten_bfs.extend(list(itertools.chain(*bfuncs)) + [-1])
            bfs_posi.append(len(flatten_bfs))
        return (flatten_bfs, bfs_posi)
    def bfuncs_reverse(self):
        for i in range(len(self.semantic)):
            self.semantic[i].s_backprogs.reverse()

    def ffuncs_merge(self):
        return list(map(lambda x: x.s_forwardfuncs, self.semantic))
class SemanticPerIndiv: 
    def __init__(self):
        self.s_nodes = []  
        self.s_idx = {}  
        self.s_idx_reverse = []
        self.semantic = []
        self.s_num = 0

        self.snodes_cpu = []
        self.tg_smt = [] 
        self.tg_drvt = [] 

    def upper(self):
        self.s_num += 1
        self.semantic.append(Semantic())

    def add_snode(self, gpu_posi, exp):
        # exp = node.print_exp_subtree(noparent=True)
        self.s_nodes.append(gpu_posi)
        # self.s_idx[exp] = len(self.s_nodes) - 1
        self.s_idx_reverse.append(exp)

    def set_snodes_dict(self):
        for i in range(len(self.s_idx_reverse)):
            self.s_idx[self.s_idx_reverse[i]] = i

    def get_snode_idx(self, exp):
        return self.s_idx[exp]



    def set_bf_node(self, node, rlt_posi, id):
        # self.semantic[id].node = (PyGP.unzip(node, tr=True), rlt_posi)
        self.semantic[id].node = (node, rlt_posi)

    def set_snodes_d(self, data):
        self.snodes_cpu = data

    def set_bfuncs_d(self, data):
        self.tg_smt = data

    def set_drvt_d(self, data):
        self.tg_drvt = data

    @property
    def count(self):
        return self.s_num

class PopSemantic:
    _defaults = {
        "semantics":    [],
        "ffuncs_d":     {},
        "tsematic_cpu": [],
        "snodes_cpu":   [],
    }

    def __init__(self, data_rg=None, base_dict=None, seed=None):
        self.library_idx = {} #{depth:{nodes}, ...}
        self.library_data = {}#{node:data, ...}
        self.data_rg = data_rg
        self.library_rg = {}
        self.semantics = []
        self.ffuncs_d = {}
        self.const_statistics = {}
        self.lbr_keys = {}
        if base_dict is not None:#update the base parameter of new process
            from PyGP import Base
            base = Base()
            base.update(*dill.loads(base_dict))
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def get_datarg(self):
        return self.data_rg

    def set_datarg(self, data_rg):
        self.data_rg = data_rg
    def append(self, semantic: SemanticPerIndiv):
        self.semantics.append(semantic)

    def extend(self, semantics):
        semantics = dill.loads(semantics)
        for i in range(len(semantics)):
            semantics[i].set_snodes_dict()
        self.semantics.extend(semantics)

    def smt_clts(self):
        for i in range(len(self.semantics)):
            (smts, trs) = self.get_snode_alld(i)
            if len(self.library[i]) == 0:
                continue
            if np.isnan(smts).any():
                continue
            for j in range(len(trs)):
                tr = PyGP.tr_copy_nc(trs[j])
                PyGP.cashClear_tr(tr)
                tr_exp = tr.print_exp_subtree(noparent=True)
                ht = tr.height()
                if not self.library_idx.get(ht):
                    self.library_idx[ht] = {}
                if not self.library_data.get(tr_exp):
                    self.library_idx[ht][tr_exp] = tr
                    self.library_data[tr_exp] = (smts[j], ht)
                    if self.data_rg is not None:
                        self.library_rg[tr_exp] = tr.getRange(self.data_rg)
                    const_num = PyGP.tr_const_num(tr)
                    if not self.const_statistics.get(const_num):
                        self.const_statistics[const_num] = []
                    self.const_statistics[const_num].insert(0, tr_exp)
        if len(self.library_data) > LIBRARY_SIZE:
            # self.const_statistics = sorted(self.const_statistics, key=lambda x: x[0])
            keys_ = list(self.const_statistics.keys())
            if len(self.library_data) - LIBRARY_SIZE > 0:
                remain = len(self.library_data) - LIBRARY_SIZE
                remain = np.random.randint(remain, remain * 2)
            else:
                remain = 0
            while(remain > 0):
                # if len(keys_) > 1:
                #     key_id = random.randint(1, len(keys_) - 1)
                # else:
                key_id = np.random.randint(0, len(keys_))
                key = keys_[key_id]
                # key_id = len(keys_) - 1
                # key = keys_[key_id]

                np.random.shuffle(self.const_statistics[key])
                items_ = self.const_statistics[key]
                init_rposi = np.random.randint(0, len(items_))
                for i in range(len(items_) - 1, init_rposi, -1):
                    t = items_[i]
                    if not self.library_data.get(t):
                        self.const_statistics[key].pop(i)
                    else:
                        (_, ht) = self.library_data[t]
                        del self.library_idx[ht][t]
                        del self.library_data[t]
                        del self.library_rg[t]
                        self.const_statistics[key].pop(i)
                        remain -= 1
                    if (remain == 0):
                        break;
                if len(self.const_statistics[key]) == 0:
                    del self.const_statistics[key]
                    keys_.pop(key_id)

        self.lbr_keys = {}
        for i in self.library_idx.keys():
            self.lbr_keys[i] = list(self.library_idx[i].keys())

    def get_lib_size(self):
        return len(self.library_data)

    def get_smt_size(self, height, init_height=None):
        h_rg = []
        if init_height is None:
            for i in range(height + 1):
                if self.library_idx.get(i):
                    h_rg.append(i)
        else:
            for i in range(init_height, height + 1):
                if self.library_idx.get(i):
                    h_rg.append(i)
        len_hrg = len(h_rg)
        size_ = 0

        for i in range(len_hrg):
            size_ += len(self.lbr_keys[h_rg[i]])
        
        return (size_, h_rg)

    def get_smt_trs(self, h_rg, r_idxs):
        exprs = []
        smts = []
        tr_size = []
        
        len_hrg = len(h_rg)

        h_list = []
        idx_list = []
        for i in range(len_hrg):
            h_list.extend([h_rg[i]] * len(self.lbr_keys[h_rg[i]]))
            idx_list.extend(list(range(0, len(self.lbr_keys[h_rg[i]]))))
        if len_hrg == 0:
            return (None, None, None)

        trs_clt = []

        segs = []
        for i in range(len(r_idxs)):
            segs.append((h_list[r_idxs[i]], idx_list[r_idxs[i]]))
            expr = self.get_expr((h_list[r_idxs[i]], idx_list[r_idxs[i]]))
            exprs.append(expr)
            smts.append(self.get_semantic(expr))
        return (segs, exprs, smts)


    def get_range(self, tr):
        return self.library_rg[tr]

    def get_semantic(self, expr):
        return self.library_data[expr][0]

    def get_inner_size(self, idx):
        return self.library_idx[idx[0]][self.lbr_keys[idx[0]][idx[1]]].inner_size

    def get_tr(self, idx):
        return self.library_idx[idx[0]][self.lbr_keys[idx[0]][idx[1]]].copy()

    def get_expr(self, idx):
        return self.lbr_keys[idx[0]][idx[1]]

    def reset(self):
        self.semantics_p = self.semantics
        # self.ffuncs_d_p = self.ffuncs_d
        self.semantics = []
        # self.ffuncs_d = {}

    def select(self, rks):

        # print(rks, len(self.semantics_p, self.semantics))
        smt_tmp = list(self.semantics_p) + list(self.semantics)
        # ffuncs_tmp = list(self.ffuncs_d_p) + list(self.ffuncs_d)
        self.semantics = [smt_tmp[x] for x in rks]
        # self.ffuncs_d = [ffuncs_tmp[x] for x in rks]


    def set_library(self, prog_sn): 
        self.library = [[(item[0], PyGP.unzip(item[1], tr=True)) if len(item) == 2
                         else (item[0], PyGP.unzip(item[1], tr=True), item[2])
                         for item in prog] for prog in prog_sn]

    def snode_merge(self):
        snodes = list(map(lambda x: x.s_nodes, self.semantics))
        return list(itertools.chain(*snodes))


    def snode_exp_merge(self):
        snodes_exp = list(map(lambda x: x.s_idx_reverse, self.semantics))
        return list(itertools.chain(*snodes_exp))


    def ffuncs_d_set(self, id, data, offset):
        if offset == 0:
            self.ffuncs_d[id] = data
        else:
            self.ffuncs_d[id] = np.concatenate([self.ffuncs_d[id], data])

    def data_load(self, tsematic, snodes, tderivate): 
        offset = 0
        len_bn = list(map(lambda x: x.count, self.semantics))
        for i in range(len(self.semantics)):
            # print(offset, len_bn[i])
            self.semantics[i].set_bfuncs_d(tsematic[offset: offset + len_bn[i]])
            # print(self.semantics[i].tg_smt)
            offset += len_bn[i]
        # print(tsematic)
        # assert (0==1)

        offset = 0
        len_sn = list(map(lambda x: len(x.s_nodes), self.semantics))
        for i in range(len(self.semantics)):
            self.semantics[i].set_snodes_d(snodes[offset: offset + len_sn[i]])
            offset += len_sn[i]

        offset = 0
        len_dn = list(map(lambda x: x.count, self.semantics))
        for i in range(len(self.semantics)):
            assert (offset + len_dn[i] <= len(tderivate))
            self.semantics[i].set_drvt_d(tderivate[offset: offset + len_dn[i]])
            offset += len_dn[i]

    def get_indiv_semantic(self, prog_id):
        new_s = self.semantics[prog_id]
        return self.semantics[prog_id]
    def get_snode_d(self, prog_id, exp): 
        # exp = node.print_exp_subtree(noparent=True)
        idx = self.semantics[prog_id].get_snode_idx(exp)
        return self.semantics[prog_id].snodes_cpu[idx]

    def get_snode_idx(self, prog_id, exp):
        return self.semantics[prog_id].get_snode_idx(exp)

    def get_snode_alld(self, prog_id): 
        return (list(map(lambda x: self.get_snode_d(x[0], x[1].print_exp_subtree(noparent=True)), self.library[prog_id])),#semantic of treenodes
                list(map(lambda x: x[1], self.library[prog_id])))#treenodes

    def get_tgsmt_d(self, prog_id, smt_id = 0): 
        return self.semantics[prog_id].tg_smt[smt_id]

    def get_drvt_d(self, prog_id, smt_id = 0):
        return self.semantics[prog_id].tg_drvt[smt_id]

    def get_node(self, prog_id, idx): 
        return self.semantics[prog_id].s_idx_reverse[idx]

    def get_tg_node(self, prog_id, smt_id = 0):
        return self.semantics[prog_id].semantic[smt_id].node[0]

    def get_tgnode_posi(self, prog_id, smt_id = 0):
        if prog_id >= len(self.semantics) or smt_id >= len(self.semantics[prog_id].semantic):
            print(prog_id, smt_id, len(self.semantics), len(self.semantics[prog_id].semantic))
            assert (0==1)
        return self.semantics[prog_id].semantic[smt_id].node[1]

    def get_snode_tgsmt(self, prog_id, smt_id = 0): 
        tg_node = self.semantics[prog_id].semantic[smt_id].node[0]
        return self.get_snode_d(prog_id, tg_node)

    def compute_tg(self, prog_id):
        resmax = 0
        tg = 0
        tgs = []

        res_id = -1
        res_max = 1e-10
        res_vals = []
        posis = []
        for i in range(self.semantics[prog_id].count):
            tgsmt = self.get_tgsmt_d(prog_id, i)
            cdd = self.get_snode_tgsmt(prog_id, i)
            tgdrvt = np.fabs(self.get_drvt_d(prog_id, i))
            posi = self.semantics[prog_id].semantic[i].node[1]
            #tgdrvt_f_idx = PyGP.cluster(tgdrvt)[0]
            if posi not in posis:
                if not (tgdrvt == 0).all() and not((np.isnan(tgsmt) | np.isinf(tgsmt)).any()):
                    tgs.append(i)
                    posis.append(posi)
                    # resmax = resval
                    #
                    mask = ~(np.isnan(tgsmt) | np.isinf(tgsmt))
                    tgsmt = tgsmt[mask]
                    tgdrvt = PyGP.abs_normalize(tgdrvt[mask])
                    cdd = cdd[mask]
                    vec = np.subtract(tgsmt, cdd) * tgdrvt
                    resval = np.sqrt(np.dot(vec, vec))
                    if resval > res_max:
                        res_max = resval
                        res_id = i
                    res_vals.append(resval)
                # elif (np.isnan(tgsmt)).any():
                #     tgs.append(i)
                #     res_vals.append(0)
                # else:
                #     tgs.append(i)
                #     res_vals.append(-1)
        if len(res_vals) == 0:
            tgs.append(np.random.randint(0, self.semantics[prog_id].count))
            res_vals.append(0)
        res_max = max(res_vals)
        res_vals = [res_max if val == -1 else val for val in res_vals]
        return (tgs, res_vals)



def bfuncs_merge(bfuncs):
    
    flatten_bfs = []
    bfs_posi = [0]
    for i in range(len(bfuncs)):
        (f_bfs_, bfs_posi_) = bfuncs[i].bfuncs_merge()
        bfs_posi.extend(np.add(bfs_posi_, len(flatten_bfs)))
        flatten_bfs.extend(f_bfs_)

    return(flatten_bfs, bfs_posi)

def ffuncs_d_clts(ffuncs):
    ffuncs = list(map(lambda x: x.ffuncs_merge()[0:], ffuncs))
    flatten_d = list(itertools.chain(*ffuncs))
    return flatten_d