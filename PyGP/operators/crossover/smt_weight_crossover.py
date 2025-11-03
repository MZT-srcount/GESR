import random

import numpy as np
import math

import PyGP
from PyGP import Program, PopSemantic, TreeNode

def cluster(array, tgsmt, cdd):

    resval = np.absolute(np.subtract(tgsmt, cdd))
    # array = resval * array
    # print('res: ', resval)
    # print('array ', array)
    x0 = np.max(array)
    x1 = np.min(array)
    time = 1
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
        return array

    return (group_0, group_1)

def r_snodes_select(smt_len, num):
        slts = np.random.choice(range(smt_len), size=num, replace=False)
        return np.sort(slts)

def indivSelect_sem(tsematic, candidate, tgdrvt, tgdrvt_f_idx):
    
    idx_min = [-1, -1]
    candidate_min = [candidate[0], candidate[1]]
    tgdrvt_f = tgdrvt[tgdrvt_f_idx]
    tsematic_f = tsematic[tgdrvt_f_idx]
    candidate_f = list(map(lambda x: x[tgdrvt_f_idx], candidate))

    rsdls = list(map(lambda x: np.subtract(tsematic, x), candidate))
    dis_all = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls))
    dis_all_w = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls * tgdrvt))
    rsdls_f = list(map(lambda x: np.subtract(tsematic_f, x), candidate_f))
    dis_all_f_w = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls_f * tgdrvt_f))

    idx = np.argmin(dis_all_f_w)
    dis_min = dis_all[idx]
    candidate_min[0] = candidate[idx]
    idx_min[0] = idx

    
    def helon_dist(x, y):
        rsdl = np.subtract(candidate_f[idx], x) * tgdrvt_f
        rlt_dis = np.sqrt(np.dot(rsdl, rsdl))
        p = (dis_all_f_w[idx] + rlt_dis + y) / 2
        helon_dis_ = p * (p - dis_all_w[idx]) * (p - rlt_dis) * (p - y)
        if helon_dis_ < 1e-5:
            return 0 if rlt_dis > 0 else dis_all_f_w[idx]
        else:
            return np.sqrt(helon_dis_) / (rlt_dis * 2)

    
    idx_1 = np.argmin(list(map(helon_dist, candidate_f, dis_all_f_w)))
    candidate_min[1] = candidate[idx_1]
    idx_min[1] = idx_1

    
    return (idx_min, candidate_min)

def least_square_method(tsematic, candidate_1, candidate_2, ccd):
    numerator = np.dot(candidate_2 - candidate_1, tsematic - candidate_1)
    denominator = np.dot(candidate_1 - candidate_2, candidate_1 - candidate_2)
    if denominator < 1e-4:
        return 0
    if np.isinf(tsematic).any() or np.isinf(denominator) or (tsematic == 2e20).any() or np.isnan(tsematic).any() or np.isinf(numerator):
        return 0
    if (math.isnan(numerator / denominator)):
        raise ValueError("why here..", numerator, denominator, candidate_1, candidate_2, tsematic, len(ccd))
    return numerator / denominator

def effect_test(tsematic, origin, candidate_1, candidate_2, k, tgdrvt, serious = False):
    candidate = (1 - k) * candidate_1 + k * candidate_2
    vec = np.subtract(tsematic, candidate) * tgdrvt
    effect = np.sqrt(np.dot(vec, vec))
    vec = np.subtract(tsematic, origin) * tgdrvt
    origin_effect = np.sqrt(np.dot(vec, vec))
    if not serious:
        return (effect < origin_effect or math.fabs(effect - origin_effect) < 1e-5, effect, origin_effect)#, origin_effect_1, origin_effect_2, effect, effect - origin_effect, origin_effect)
    else:
        return (effect < origin_effect and math.fabs(effect - origin_effect) > 1e-2, effect, origin_effect)#, origin_effect_1, origin_effect_2, effect, effect - origin_effect, origin_effect)

def m_normalize(array):
    x = np.array(np.absolute(array), dtype=np.float32)
    x_max = np.max(x)
    x_min = np.min(x)
    if(x_max == x_min):
        return array
    return (x - x_min) / (x_max - x_min)
    

def crossover(pprogs: [Program], smts: PopSemantic, funcs, r_slt):
    progs = []
    idx = 0
    # print("len(progs): ", len(progs))
    prog_depth_max = 0
    for i in range(len(pprogs)):
        indiv1:Program = pprogs[i]
        child = indiv1.copy(i)

        if indiv1.seman_sign >= 0:
            idx += 1
            id = indiv1.seman_sign
            indiv1.seman_sign=-1
            rand_uniform = indiv1.rd_st.uniform(0, 1)

            (candidate, trs_cdd) = smts.get_snode_alld(id)
            r_num = random.randint(1, len(candidate[0])) if r_slt else len(candidate[0])
            # r_snodes = r_snodes_select(len(candidate[0]), r_num)
            r_snodes = [i for i in range(len(candidate[0]))]
            candidate = [candidate[i][r_snodes] for i in range(len(candidate))]

            tr_origin = smts.get_tg_node(id)
            cdd_origin = smts.get_snode_tgsmt(id)[r_snodes]

            candidate = candidate[1:]
            trs_cdd = trs_cdd[1:]
            tgsmt = smts.get_tgsmt_d(id)[r_snodes]
            tgdrvt = smts.get_drvt_d(id)[r_snodes]
            tgdrvt = m_normalize(tgdrvt)

            if (np.isnan(tgdrvt).any()):
                print("id: ", id)
                print('tgdrvt ', tgdrvt)
                print('tgsmt ', tgsmt)
                print('cdd_origin ', cdd_origin)
                print(pprogs[id].exp_draw())
                print(smts.get_tg_node(id).exp_draw())
                assert (1 == 0)
            tgdrvt_f_idx = cluster(tgdrvt, tgsmt, cdd_origin)[0]

            (indiv_idx, indivs) = indivSelect_sem(tsematic=tgsmt, candidate=candidate, tgdrvt=tgdrvt, tgdrvt_f_idx=tgdrvt_f_idx)

            k:float = float(least_square_method(tgsmt[tgdrvt_f_idx], indivs[0][tgdrvt_f_idx], indivs[1][tgdrvt_f_idx], indivs))
            effect_better = effect_test(tgsmt, cdd_origin,
                                        indivs[0], indivs[1], k, tgdrvt, serious=True)
            subtree3:TreeNode = smts.get_tg_node(id)
            rlt_posi = subtree3.rlt_posi()
            subtree3 = child.getSubTree(rlt_posi)
            # if(id == 0):
            #     print('tr_origin: ', cdd_origin)
            #     print('tgsmt: ', tgsmt)
            #     subtree3.exp_draw()
            #     tr_origin.exp_draw()

            # a = False
            if not effect_better[0] and 10 - (subtree3.relative_depth() + subtree3.height()) >= 2:
                # a = True
                candidate.insert(0, cdd_origin)
                trs_cdd.insert(0, tr_origin)
                (indiv_idx, indivs) = indivSelect_sem(tsematic=tgsmt, candidate=candidate, tgdrvt=tgdrvt, tgdrvt_f_idx=tgdrvt_f_idx)
                k = float(least_square_method(tgsmt[tgdrvt_f_idx], indivs[0][tgdrvt_f_idx], indivs[1][tgdrvt_f_idx], indivs))
                effect_better = effect_test(tgsmt,cdd_origin,
                                            indivs[0], indivs[1], k, tgdrvt, serious=True)
                # if(not (not effect_better[0] or (indiv_idx[0] != 0 and indiv_idx[1] != 0))):
                #     print(subtree3.height(), trs_cdd[indiv_idx[0]].height(), trs_cdd[indiv_idx[1]].height(), tr_origin.height())
            if effect_better[0]:
                subtree1:TreeNode = trs_cdd[indiv_idx[0]].copy(indiv1.c_mngr)
                subtree2:TreeNode = trs_cdd[indiv_idx[1]].copy(indiv1.c_mngr)

                # if (not (not effect_better[0] or (indiv_idx[0] != 0 and indiv_idx[1] != 0)) and a):
                #     print(subtree3.height(), trs_cdd[indiv_idx[0]].height(),
                #           trs_cdd[indiv_idx[1]].height(), subtree1.height(),
                #           subtree2.height())
                # assert(indiv_idx[0] != indiv_idx[1])
                if(math.fabs(k) < 1e-5):
                    tr3 = subtree1
                elif(math.fabs(k - 1) < 1e-5):
                    tr3 = subtree2
                else:
                    if subtree1.dtype == 'Const':
                        tr1 = TreeNode(PyGP.ID_MANAGER.idAllocate(), subtree1.nodeval * (1 - k))
                    else:
                        tr1 = TreeNode(PyGP.ID_MANAGER.idAllocate(), funcs.funcSelect_n('mul'))
                        tr1.setChilds([subtree1, TreeNode(PyGP.ID_MANAGER.idAllocate(), 1 - k, parent=(tr1, 1))])
                        subtree1.setParent((tr1, 0))
                    if subtree2.dtype == 'Const':
                        tr2 = TreeNode(PyGP.ID_MANAGER.idAllocate(), subtree2.nodeval * k)
                    else:
                        tr2 = TreeNode(PyGP.ID_MANAGER.idAllocate(),funcs.funcSelect_n('mul'))
                        tr2.setChilds([subtree2, TreeNode(PyGP.ID_MANAGER.idAllocate(), k, parent=(tr2, 1))])
                        subtree2.setParent((tr2, 0))
                    if tr1.dtype == 'Const' and tr2.dtype == 'Const':
                        tr3 = TreeNode(PyGP.ID_MANAGER.idAllocate(), tr1.nodeval + tr2.nodeval)
                    else:
                        tr3 = TreeNode(PyGP.ID_MANAGER.idAllocate(), funcs.funcSelect_n('add'))
                        tr3.setChilds([tr1, tr2])
                        tr1.setParent((tr3, 0))
                        tr2.setParent((tr3, 1))


                if subtree3.parent is not None:
                    tr3.setParent(subtree3.parent)
                else:
                    child.root = tr3

            # if (not (not effect_better[0] or (indiv_idx[0] != 0 and indiv_idx[1] != 0)) and a):
            #     print(subtree3.height(), trs_cdd[indiv_idx[0]].height(),
            #           trs_cdd[indiv_idx[1]].height(), tr3.height(), subtree1.height(), subtree2.height())
                # assert (not effect_better[0] or (indiv_idx[0] != 0 and indiv_idx[1] != 0))
        child.sizeUpdate()
        progs.append(child)
        if prog_depth_max < child.depth:
            prog_depth_max = child.depth
    print('crossover time: ', idx, prog_depth_max)
    return progs

class SMT_Weight_Crossover:
    def __call__(self, pprogs, smts, funcs, r_slt=False):
        return crossover(pprogs, smts, funcs, r_slt)