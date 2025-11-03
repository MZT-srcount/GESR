from PyGP import TreeNode, Program
import numpy as np
import random
import PyGP
import math
def randSubtree(prog, depth_limit, rand_state, data_rg):
    init_func = prog.funcs.funcSelect(rand_state.randint(0, prog.funcs.len() - 1))
    if depth_limit <= 1:
        if prog.const_range is None:
            terminal = rand_state.randint(0, prog.n_terms - 1)
        else:
            len_nterms = prog.n_terms * 10
            terminal = rand_state.randint(0, len_nterms)
            if terminal == len_nterms:
                terminal = rand_state.uniform(prog.const_range[0], prog.const_range[1])
            else:
                terminal = len_nterms % prog.n_terms

        root = TreeNode(terminal)
        return root
    root = TreeNode(init_func)
    stack = [root]
    tstack = []
    depth = 1
    funcs_size = prog.funcs.len()
    divide_check = []
    while stack:
        node = stack.pop()
        childs = []
        if PyGP.INTERVAL_COMPUTE and node.dtype == "Func" and (node.nodeval.name == '/' or node.nodeval.name == 'log'):
            divide_check.append(node)
        for i in range(node.getArity()):
            if depth + 1 < depth_limit and rand_state.randint(0, funcs_size + prog.n_terms - 1) < funcs_size:
                r_oper = rand_state.randint(0, funcs_size - 1)
                tr = TreeNode(prog.funcs.funcSelect(r_oper), parent=(node, i))
                childs.append(tr)
            else:
                if prog.const_range is None:
                    terminal = rand_state.randint(0, prog.n_terms - 1)
                else:
                    len_nterms = prog.n_terms * 10
                    terminal = rand_state.randint(0, len_nterms)
                    if terminal == len_nterms:
                        terminal = rand_state.uniform(prog.const_range[0], prog.const_range[1])
                    else:
                        terminal = len_nterms % prog.n_terms

                childs.append(TreeNode(terminal, parent=(node, i)))
        if childs:
            node.setChilds(childs)
        tstack.extend(childs)
        if not stack and tstack:
            stack = tstack
            tstack = []
            depth += 1
    divide_check.reverse()
    for node in divide_check:
        childs = node.getChilds()
        if node.nodeval.name == '/':
            smt_rg = childs[1].getRange(data_rg)
        elif node.nodeval.name == 'log':
            smt_rg = childs[0].getRange(data_rg)
        if (smt_rg[0] <= 0. <= smt_rg[1] or math.fabs(smt_rg[0]) == 0.0 or math.fabs(smt_rg[1]) == 0.0)\
                and ((node.nodeval.name == '/' and childs[0].print_exp_subtree() != childs[1].print_exp_subtree()) or node.nodeval.name == 'log'):
            if node.nodeval.name == '/':
                new_tr = TreeNode(1., parent=(node, 1))
                node.setChilds([childs[0], new_tr])
            if node.nodeval.name == 'log':
                new_tr = TreeNode(1., parent=(node, 0))
                node.setChilds([new_tr])

    return root

def bounds_check(subtr:TreeNode, smt_rg, data_rg):

    rg = (smt_rg[0], smt_rg[1])
    ancestors = subtr.getAncestors()
    for x in ancestors:
        trs = x[0].getChilds().copy()
        if x[0].nodeval.name == '/':
            tr_exp_0 = trs[0].print_exp_subtree()
            tr_exp_1 = trs[1].print_exp_subtree()
            if x[1] == 1 and (rg[0] <= 0. <= rg[1] or math.fabs(rg[0]) == 0.0 or math.fabs(rg[1]) == 0.0) and tr_exp_0 != tr_exp_1:
                return False
            if x[1] == 0:
                rg_1 = trs[1].getRange(data_rg)
                if (rg_1[0] <= 0. <= rg_1[1] or math.fabs(rg_1[0]) == 0.0 or math.fabs(rg_1[1]) == 0.0) and tr_exp_0 != tr_exp_1:
                    return False
        elif x[0].nodeval.name == 'log':
            if x[1] == 0 and (rg[0] <= 0. <= rg[1] or math.fabs(rg[0]) == 0.0 or math.fabs(rg[1]) == 0.0):
                return False

        # trs.pop(x[1])
        rgs = [node.getRange(data_rg) for node in trs]
        # rgs.insert(x[1], rg)
        rg = PyGP.rg_compute(x[0], 0, rgs)
    return True

def mutation(progs: [Program], smts, mut_rate, funcs):
    data_rg = smts.get_datarg()
    cst_r = True if progs[0].const_range is not None else False 
    slted_progs = np.squeeze(np.where(np.random.random(len(progs) - 1) < mut_rate))
    rnode_depth = list(map(lambda x: np.random.randint(0, x.depth) if 0 < x.depth else 0, progs))
    rnode = np.fromiter(map(lambda x: np.random.randint(progs[x].depth_nnum(rnode_depth[x] - 1) + 1 if rnode_depth[x] > 0 else 0, progs[x].depth_nnum(rnode_depth[x]) + 1) if rnode_depth[x] > 0 else 0, slted_progs), dtype=np.int32)
    
    r_prob = np.random.random(len(slted_progs))
    split_progs = slted_progs[r_prob < 0.9]
    split_rnode = rnode[r_prob < 0.9]
    time = 5
    # assert (not cst_r)
    for (key, i) in enumerate(split_progs):
        sub_root = progs[i].getSubTree(split_rnode[key])
        h_max = sub_root.height()#progs[i].depth - sub_root.relative_depth()
        count = 0
        while(count < time):
            r_subtr = randSubtree(progs[i], random.randint(0, h_max), random, data_rg)
            smt_rg = r_subtr.getRange(data_rg)
            if sub_root.parent is not None:
                sub_root.reset_subtree(r_subtr)
            if PyGP.INTERVAL_COMPUTE and not bounds_check(r_subtr, smt_rg, data_rg):
                r_subtr.reset_subtree(sub_root)
            else:
                if r_subtr.parent is None:
                    progs[i].root = r_subtr
                progs[i].sizeUpdate()
                break
            count += 1

    split_progs = slted_progs[r_prob >= 0.1]
    split_rnode = rnode[r_prob >= 0.1]
    len_nterms = progs[0].n_terms * 10
    tmn = list(map(lambda x: random.randint(0, len_nterms) if cst_r else random.randint(0, len_nterms - 1), split_progs))
    for (key, i) in enumerate(split_progs):
        sub_root = progs[i].getSubTree(split_rnode[key])
        terminal = tmn[key]

        count = 0
        if sub_root.dtype == 'Func':
            func_list = funcs.arity_function_set[sub_root.getArity()]
            while(count < time):
                r_oper = func_list[random.randint(0, len(func_list) - 1)]

                func_origin = sub_root.nodeval
                sub_root.reset(r_oper)
                childs = sub_root.getChilds()
                child_rg = []
                for i in range(len(childs)):
                    child_rg.append(childs[0].getRange(data_rg))
                new_rg = PyGP.rg_compute(r_oper, 0, child_rg)
                if PyGP.INTERVAL_COMPUTE and (((r_oper.name == '/' or r_oper.name == 'log') and (child_rg[0][0] <= 0. <= child_rg[0][1] or math.fabs(child_rg[0][0]) == 0.0 or math.fabs(child_rg[0][1]) == 0.0)) or not bounds_check(sub_root, new_rg, data_rg)):
                    sub_root.reset(func_origin)
                else:
                    break
                count += 1
        else:
            if terminal == len_nterms:
                terminal = random.uniform(progs[i].const_range[0], progs[i].const_range[1])
                new_rg = (terminal, terminal)
            else:
                terminal = int(terminal % progs[i].n_terms)
                new_rg = (data_rg[terminal][0], data_rg[terminal][1])
            tmn_origin = sub_root.nodeval
            sub_root.reset(terminal)
            if PyGP.INTERVAL_COMPUTE and not bounds_check(sub_root, new_rg, data_rg):
                sub_root.reset(tmn_origin)

        sub_root.dtype_update()

        progs[i].sizeUpdate()

class RtnMutation:
    def __call__(self, progs, smts, mut_rate, funcs):
        return mutation(progs, smts, mut_rate, funcs)