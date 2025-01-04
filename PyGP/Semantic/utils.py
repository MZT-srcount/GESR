import PyGP
import random
import dill


def smtSign_db(prog, rand_state):  # 支持多个？标记一个用于backpropagation的语义点
    r_depth = rand_state.randint(0, prog.depth + 1)
    if r_depth == 0:
        id = 0
    else:
        init_id = prog.depth_nnum(r_depth - 1) + 1
        end_id = prog.depth_nnum(r_depth)
        id = rand_state.randint(init_id, end_id + 1)
    tnode = prog.getSubTree(id)
    # while tnode.semantic_sign == 0:
    #     tnode = self.getSubTree(rand_state.randint(1, self.length - 1))
    prog_depth = prog.depth
    if prog_depth >= PyGP.DEPTH_MAX_SIZE and rand_state.uniform(0, 1) < 1:
        r_depth = rand_state.randint(1, prog_depth - 2 + 1)
        init_id = prog.depth_nnum(r_depth - 1) + 1
        end_id = prog.depth_nnum(r_depth)
        id = rand_state.randint(init_id, end_id + 1)
        tnode = prog.getSubTree(id)
    depth = tnode.relative_depth()
    height = tnode.height()
    tnode.semantic_sign = 0
    prog.seman_sign = prog.prog_id
    return (depth, height, id)



def smtSign_nb(prog, rand_state):  # 支持多个？标记一个用于backpropagation的语义点
    id = rand_state.randint(0, prog.length)
    tnode = prog.getSubTree(id)
    # while tnode.semantic_sign == 0:
    #     tnode = self.getSubTree(rand_state.randint(1, self.length - 1))
    prog_depth = prog.depth
    if prog_depth >= PyGP.PyGP.DEPTH_MAX_SIZE and rand_state.uniform(0, 1) < 1:
        length = prog.depth_nnum(prog_depth - 2)
        id = rand_state.randint(1, length + 1)
        tnode = prog.getSubTree(id)
    depth = tnode.relative_depth()
    height = tnode.height()
    tnode.semantic_sign = 0
    prog.seman_sign = prog.prog_id
    return (depth, height, id)

def semanticSave_tnode(tnode):
    tnode.semantic_save = 0

def backpSelect(pop, select_num, slt_depth, tg_list, lb_list):#选择语义backpropagation节点
    sem_selects = [[] for i in range(pop.pop_size)]
    sem_posi = []
    for i in tg_list:
        for j in range(PyGP.SEMANTIC_CDD):#选取多个目标点
            if random.uniform(0, 1) < 0.1:
                (depth, height, id) = smtSign_db(pop.progs[i], pop.rand_state)#标记一个语义点
            else:
                (depth, height, id) = smtSign_nb(pop.progs[i], pop.rand_state)#标记一个语义点
            sem_posi.append(id)
            tnode_ = pop.progs[i].getSubTree(id)
            # sem_selects[i].append((i, tnode_.zip(), id))
            semanticSave_tnode(tnode_)
    for i in lb_list:
        for j in range(select_num):
            # rand_val = random.randint(1, pop.pop_size - 1)
            if slt_depth is not None:
                rand_depth = slt_depth
            else:
                rand_depth = random.randint(0, pop.progs[i].depth)
            # height_ = pop.progs[i].depth - depth
            # depth = pop.progs[rand_val].depth - height_ + 2

            # r_depth = random.randint(1, pop.progs[rand_val].depth)
            # init_id = pop.progs[rand_val].depth_nnum(r_depth - 1) + 1
            # end_id = pop.progs[rand_val].depth_nnum(r_depth)
            # id = random.randint(init_id, end_id)
            # tnode = pop.progs[rand_val].getSubTree(id)
            tnode = PyGP.tnode_depth_select(pop.progs[i], rand_depth)#为了不增加深度，应该增加2
            # if tnode.relative_depth() <= tnode_.relative_depth() + 1 and tnode.height() != 1:
            #     raise ValueError(tnode.height(), tnode_.height(), pop.progs[rand_val].depth, depth + 2)
            semanticSave_tnode(tnode)
            sem_selects[i].append((i, tnode.zip()))
    return sem_selects
