'''
Author: your name
Date: 2023-08-08 15:50:06
LastEditTime: 2023-08-08 15:50:27
LastEditors: your name
Description: 
FilePath: \PyGP\PyGP\helper.py
可以输入预定的版权声明、个性签名、空行等
'''
import multiprocessing
import random

import PyGP
from PyGP import TreeNode, Program
import numpy as np
import math
import warnings

warn_path = "./logs/warnings_log.txt"
warn_file = open(warn_path, 'w+')
warnings.filterwarnings('always', '', Warning, append=True)
warnings.showwarning = lambda message, category, filename, lineno, file=warn_file, line=0: warn_file.write(f"{filename}:{lineno}:{category.__name__}:{message}\n")

sharedList = []

def ignore_warnings():
    warnings.filterwarnings("ignore", message="RuntimeWarning")
    
def init_shared(sharedmemory,  base_dict, seed):
    import dill
    global sharedList
    sharedList.extend(sharedmemory)
    base_dict = dill.loads(base_dict)
    base = Base()
    base.update(*base_dict)
    cur_worker = multiprocessing.current_process()
    random.seed(seed + cur_worker._identity[0])
    np.random.seed(seed + cur_worker._identity[0])
    # random.seed(seed)
    # np.random.seed(seed)

def time_record(time_s, func, *args):

    start = time.time()
    func(*args)
    end = time.time()
    time_s.append(end - start)

def dataset_transform(k, dataset, sub_datasetsize):
    np_dataset = []
    dataset_size = len(dataset[0])
    for i in range(k):
        subset = []
        for j in range(len(dataset)):
            if i == k - 1:
                list_tmp = dataset[j][i * sub_datasetsize:]
                subset.extend(list_tmp)
                padding_size = sub_datasetsize - len(list_tmp)
                for z in range(padding_size):
                    subset.append(0)
            else:
                subset.extend(dataset[j][i * sub_datasetsize: (i + 1) * sub_datasetsize])
        np_dataset.append(np.array(subset, dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64))
    return (np_dataset, sub_datasetsize)

from PyGP import Base
def unzip(mainbody, pop_id=0, tr=False):
    idx = 0
    if not tr:
        prog_part = {'pop_id':round(mainbody[idx]), 'prog_id':round(mainbody[idx+1]),'seman_sign':round(mainbody[idx+2])}
        idx += 3

    def rebuild_tr(array):
        # [*nodeval, self.node_id, *self.cash, self.visited]
        node_dict = {}
        if round(array[0]) == 0:
            node_dict['nodeval'] = Base.pop_dict[pop_id]['funcs'].funcSelect_oid(round(array[1]))
        elif round(array[0]) == 1:
            node_dict['nodeval'] = round(array[1])
        else:
            node_dict['nodeval'] = array[1]
        node_dict['node_id'] = round(array[2])
        node_dict['cash'] = [round(array[3]), round(array[4])]
        node_dict['visited'] = round(array[5])
        return node_dict
    root = TreeNode(**(rebuild_tr(mainbody[idx:idx + 6])))
    idx += 6
    stack = [root]

    # restore root
    while stack:
        pnode = stack.pop(0)
        arity = pnode.getArity()
        if arity > 0:
            childs = []
            for i in range(arity):
                childs.append(TreeNode(parent=(pnode, i), **(rebuild_tr(mainbody[idx:idx + 6]))))
                idx += 6
            pnode.setChilds(childs)
            stack.extend(pnode.getChilds())

    if not tr:
        return Program(**prog_part, root=root)
    else:
        return root


def unzip_old(mainbody):
    prog_part = mainbody.pop(0)
    root = TreeNode(**(mainbody.pop(0)))
    stack = [root]

    # restore root
    while stack:
        pnode = stack.pop(0)
        arity = pnode.getArity()
        if arity > 0:
            childs = [TreeNode(parent=(pnode, i), **(mainbody.pop(0))) for i in range(arity)]
            pnode.setChilds(childs)
            stack.extend(pnode.getChilds())
    return Program(**prog_part, root=root)
def inorder_traversal(tnode: TreeNode, str_, stack):
    if tnode.dtype == "Func":
        childs = tnode.getChilds()
        inorder_traversal(childs[len(childs) - 1], str_, stack)
        for i in range(1, len(childs)):
            inorder_traversal(childs[len(childs) - i - 1], str_, stack)
        if tnode.parent is not None and (tnode.parent[0].nodeval.priority > tnode.nodeval.priority or \
                ((tnode.parent[0].nodeval.name == '-' or tnode.parent[0].nodeval.name == '/') and tnode.parent[0].nodeval.priority == tnode.nodeval.priority and tnode.parent[1] > 0)):#默认父代也为Func类型
            if tnode.getArity() == 2:
                str_[0] = '(' + stack.pop() + ' ' + tnode.nodeval.name + ' ' + stack.pop() + ')'
            elif tnode.getArity() == 1:
                str_[0] = '(' + tnode.nodeval.name + '(' + stack.pop() + '))'
        else:
            if tnode.getArity() == 2:
                str_[0] = stack.pop() + ' ' + tnode.nodeval.name + ' ' + stack.pop()
            elif tnode.getArity() == 1:
                str_[0] = tnode.nodeval.name + '(' + stack.pop() + ')'
        stack.append(str_[0])
    else:
        if isinstance(tnode.nodeval, int):
            str_[0] = 'x' + str(tnode.nodeval)
        elif isinstance(tnode.nodeval, float):
            str_[0] = str(tnode.nodeval)#str(round(tnode.nodeval, 6))
        else:
            str_[0] = str(tnode.nodeval)
        stack.append(str_[0])

from PyGP import Program
def tnode_depth_select_(prog: Program, depth_select):#根据深度选择相应语义节点
    stack = [prog.root]
    dep_stack = []
    standby_stack = []
    depth = -1
    idx = 0
    tnode_select = -1
    cnode = prog.root
    achieve = False
    while stack:
        if depth_select == depth and not achieve:
            tnode_select = random.randint(idx, prog.length - 1)
            achieve = True
            # tnode_ret = prog.getSubTree_depbased(tnode_select)
            # print(depth, depth_select, prog.getSubTree_depbased(tnode_select).height(), prog.getSubTree(idx).height(), idx)
            # if not isinstance(tnode_ret, TreeNode):
            #     raise ValueError("sth wrong..", tnode_ret)
            # return tnode_ret
        if tnode_select == idx:
            if cnode.relative_depth() < depth:
                raise ValueError(depth, depth_select, cnode.height(), cnode.relative_depth(), prog.getSubTree_depbased(tnode_select).height(), idx, prog.root.height(), prog.depth, cnode.dtype)
            return cnode
        cnode = stack.pop(0)
        if cnode.dtype == "Func":
            dep_stack.extend(cnode.getChilds())
        idx += 1
        if len(stack) == 0:
            stack = dep_stack
            if len(dep_stack) > 0:
                standby_stack = stack.copy()
            dep_stack = []
            depth += 1
    return standby_stack[random.randint(0, len(standby_stack) - 1)] if len(standby_stack) > 0 else prog.root#没有找到适合深度的节点，返回叶子节点

def tnode_depth_select(prog: Program, depth_select):
    if depth_select == 0:
        id = 0
    else:
        init_id = prog.depth_nnum(depth_select - 1) + 1
        end_id = prog.depth_nnum(depth_select)
        id = random.randint(init_id, end_id)
    return prog.getSubTree(id)

from PyGP import Population
import time

def semanticSearch(treenode):#查找该子树是否存在语义标识
    if PyGP.SEMANTIC_SIGN:
        stack = [treenode]
        while stack:
            node = stack.pop()
            if node.dtype == "Func":
                stack.extend(node.getChilds())
            if node.semantic_sign == 0 or node.semantic_save == 0:
                return True
    return False

def data_filter(dataset, fitness, range_, num):
    n_terms = len(dataset)
    d_len = len(dataset[0])
    data = []

    #设置初始中心点
    for i in range(d_len):
        data.append((np.array([dataset[j][i] for j in range(n_terms)]), fitness[i]))
    print(data)
    data = sorted(data, key=lambda x: x[0][0])
    step = (range_[0][1] - range_[0][0]) / num
    step_num = d_len / num
    group = [[] for i in range(num)]
    center = [np.array([range_[0][0] + step / 2 + step * i if j == 0 else data[int(step_num * i + step_num / 2)][0][j] for j in range(n_terms)]) for i in range(num)]

    #k-mean聚类
    times = 50
    count = 0
    while(count < times):
        dist_arg = list(map(lambda x: np.argmin([np.sqrt(np.dot(x[0] - center[i], x[0] - center[i]))for i in range(num)]), data))
        list(map(lambda x: group[x[1]].append(data[x[0]]), enumerate(dist_arg)))

        for i in range(num):
                if(len(group[i]) == 0):
                    group[i].append(data[random.randint(0, len(data) - 1)])
                center_ = np.zeros(n_terms, dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
                for j in range(len(group[i])):
                    center_ += group[i][j][0]
                center[i] = center_ / len(group[i])
        count += 1

    data_filter = [[] for i in range(n_terms)]
    res_filter = []
    #寻找最近点
    for i in range(num):
        # assert(len(group[i]) > 0)
        dist = list(map(lambda x: np.sqrt(np.dot(x[0] - center[i], x[0] - center[i])), group[i]))
        idx = np.argmin(dist)
        list(map(lambda x: data_filter[x[0]].append(x[1]), enumerate(group[i][idx][0])))
        res_filter.append(group[i][idx][1])
    return (data_filter, res_filter)

def cluster(array):#二类聚类，根据k-聚类改编

    # resval = np.absolute(np.subtract(tgsmt, cdd))
    # array = resval * array
    # print('res: ', resval)
    # print('array ', array)
    x0 = np.max(array)
    x1 = np.min(array)
    time = 10
    count = 0

    if not np.isinf(array).all() and not np.isinf(x0 + x1):
        while(count < time):
            dis_0 = np.absolute(list(map(lambda x: x - x0, array)))
            dis_1 = np.absolute(list(map(lambda x: x - x1, array)))
            group_0 = np.where(np.squeeze(dis_0 <= dis_1))
            group_1 = np.where(np.squeeze(dis_0 > dis_1))
            x0 = np.mean(array[group_0])
            x1 = np.mean(array[group_1])
            count += 1

    else:
        return [range(len(array))]

    return (group_0, group_1)
def abs_normalize(array):
    x = np.array(np.absolute(array), dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64)#取绝对值
    x_max = np.max(x)
    x_min = np.min(x)
    if(x_max == x_min):
        return np.fabs(array)
    return (x - x_min) / (x_max - x_min)
    # return (x - np.mean(x)) / np.std(x) if np.std(x) != 0 else np.ones(len(array))#均值归一

def IterRun(population_size, generation, cross_rate, mut_rate, init_method , init_depth, data_train, fit_train, function_set=['add', 'sub', 'mul', 'div']):
    pop = Population(population_size, cross_rate=cross_rate, mut_rate=mut_rate, function_set=['add', 'sub', 'mul', 'div'])

    pop.initDataset(data_train, fit_train)

    pop.initialization(init_method, init_depth)
    pop.execution()
    for i in range(generation):
        start = time.time()
        print('------------------ iteration: %d --------------------' % i)
        pop.crossover()
        pop.mutation()
        pop.execution()
        pop.selection()

        end = time.time()
        print('程序运行时间为：', end - start, '秒', 'aver_size: ', pop.getAverSize(), '\n\n')


def rg_compute(node, idx, data_rg):
    opera = node.nodeval.id if isinstance(node, TreeNode) else node
    rg_0 = rg_1 = 0
    # try:
    #     rg__ = data_rg[idx][0] + data_rg[idx+1][0]
    # except RuntimeWarning:
    #     print(data_rg[idx][0], data_rg[idx+1][0], data_rg)
    #     assert (0 == 1)
    if opera == 0:
        rg_0 = data_rg[idx][0] + data_rg[idx+1][0]
        rg_1 = data_rg[idx][1] + data_rg[idx+1][1]
    elif opera == 1:
        rg_0 = data_rg[idx][0] - data_rg[idx + 1][1]
        rg_1 = data_rg[idx][1] - data_rg[idx + 1][0]
    elif opera == 2:
        rg_0 = min(data_rg[idx][0] * data_rg[idx + 1][0],
                   data_rg[idx][0] * data_rg[idx + 1][1],
                   data_rg[idx][1] * data_rg[idx + 1][0],
                   data_rg[idx][1] * data_rg[idx + 1][1], )
        rg_1 = max(data_rg[idx][0] * data_rg[idx + 1][0],
                   data_rg[idx][0] * data_rg[idx + 1][1],
                   data_rg[idx][1] * data_rg[idx + 1][0],
                   data_rg[idx][1] * data_rg[idx + 1][1], )
        if node.childs[0].print_exp_subtree() == node.childs[1].print_exp_subtree() and rg_0 < 0:
            rg_0 = 0.
    elif opera == 3:#[ ]idx+1的区间包含0时并不准确，不过如果可以用来排除0的话就行了
        if data_rg[idx + 1][0] <= 0. <= data_rg[idx + 1][1] or math.fabs(data_rg[idx + 1][0]) == 0.0 or math.fabs(data_rg[idx + 1][1]) == 0.0:
            rg_0, rg_1 = -1.e10, 1.e10
        else:
            rg_0 = min(data_rg[idx][0] / data_rg[idx + 1][0],
                       data_rg[idx][0] / data_rg[idx + 1][1],
                       data_rg[idx][1] / data_rg[idx + 1][0],
                       data_rg[idx][1] / data_rg[idx + 1][1], )
            rg_1 = max(data_rg[idx][0] / data_rg[idx + 1][0],
                       data_rg[idx][0] / data_rg[idx + 1][1],
                       data_rg[idx][1] / data_rg[idx + 1][0],
                       data_rg[idx][1] / data_rg[idx + 1][1], )
        if node.childs[0].print_exp_subtree() == node.childs[1].print_exp_subtree():
            rg_0 = rg_1 = 1
    elif opera == 4:
        if data_rg[idx][0] < data_rg[idx][1]:
            rg_0 = data_rg[idx][0]
            rg_1 = data_rg[idx][1]
        else:
            rg_0 = data_rg[idx][1]
            rg_1 = data_rg[idx][0]

        left_min = math.floor(rg_0 / (math.pi * 2)) * math.pi * 2

        if rg_1 - rg_0 >= 2 * math.pi or rg_0 <= left_min + math.pi <= rg_1:#约算，主要看是否过零点
            rg_0, rg_1 = -1, 1
        else:
            if rg_1 - left_min < math.pi:
                rg_0, rg_1 = min(math.sin(rg_0), math.sin(rg_1)), 1
            else:
                rg_0, rg_1 = -1, max(math.sin(rg_0), math.sin(rg_1))

    elif opera == 5:
        if data_rg[idx][0] < data_rg[idx][1]:
            rg_0 = data_rg[idx][0]
            rg_1 = data_rg[idx][1]
        else:
            rg_0 = data_rg[idx][1]
            rg_1 = data_rg[idx][0]

        left_min = math.floor(rg_0 / (math.pi * 2)) * math.pi * 2 - math.pi / 2

        if rg_1 - rg_0 >= 2 * math.pi or rg_0 <= left_min + math.pi <= rg_1:#约算，主要看是否过零点
            rg_0, rg_1 = -1, 1
        else:
            if rg_1 - left_min < math.pi:
                rg_0, rg_1 = min(math.sin(rg_0 - left_min), math.sin(rg_1 - left_min)), 1
            else:
                rg_0, rg_1 = -1, max(math.sin(rg_0 - left_min), math.sin(rg_1 - left_min))

    elif opera == 6:
        rg_0 = math.log(math.fabs(data_rg[idx][0])) if data_rg[idx][0] != 0 else -1e7
        rg_1 = math.log(math.fabs(data_rg[idx][1])) if data_rg[idx][1] != 0 else -1e7

    elif opera == 7:
        if data_rg[idx][0] < 8:
            rg_0 = math.exp(data_rg[idx][0])
        else:
            rg_0 = math.exp(8)
        if data_rg[idx][1] < 8:
            rg_1 = math.exp(data_rg[idx][1])
        else:
            rg_1 = math.exp(8)
    
    elif opera == 8:
        rg_0 = math.sqrt(math.fabs(data_rg[idx][0])) 
        rg_1 = math.sqrt(math.fabs(data_rg[idx][1]))

    elif opera == 9:
        rg_0 = math.fabs(data_rg[idx][0])
        rg_1 = math.fabs(data_rg[idx][1])

    if isinstance(node, TreeNode):
        return (rg_0, rg_1, node.getArity())
    else:
        return (rg_0, rg_1)



def tr_const_num(tr):
    depth_ = -1
    stack = [tr]
    cst_num = 0
    while stack:
        node = stack.pop()
        if node.dtype == "Func":
            stack.extend(node.getChilds())
        if node.dtype == "Const":
            cst_num += 1
    return cst_num
# data_size = 20
# test_size = 1000
# n_terms = 2
# range_ = [-2, 2]
# range_curve = [range_ for i in range(n_terms)]
# data = [[random.uniform(range_[0], range_[1]) for i in range(data_size)] for j in range(n_terms)]
# fitness = [8.0 / (2.0 + data[0][i] ** 2 + data[1][i] ** 2) for i in range(data_size)]
#
# (d_new, f_new) = data_filter(data, fitness, range_curve, 10)
#
# print('\n\n', d_new, '\n', f_new)
