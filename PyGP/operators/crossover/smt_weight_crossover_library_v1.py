import random

import numpy as np
import math

import PyGP
from PyGP import Program, PopSemantic, TreeNode

def r_snodes_select(smt_len, num):
        slts = np.random.choice(range(smt_len), size=num, replace=False)
        return np.sort(slts)

def indivSelect_sem(tsematic, candidate, tgdrvt, tgdrvt_f_idx, tgdrvt_origin):
    
    idx_min = [-1, -1]
    candidate_min = [candidate[0], candidate[1]]
    tgdrvt_f = tgdrvt[tgdrvt_f_idx]
    tsematic_f = tsematic[tgdrvt_f_idx]
    candidate_f = list(map(lambda x: x[tgdrvt_f_idx], candidate))

    rsdls = list(map(lambda x: np.subtract(tsematic, x), candidate))
    dis_all = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls))
    dis_all_w = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls * tgdrvt))
    rsdls_f = list(map(lambda x: np.subtract(tsematic_f, x), candidate_f))
    dis_all_f_w = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls_f))

    idx = np.argmin(dis_all_w)
    dis_min = dis_all[idx]
    candidate_min[0] = candidate[idx]
    idx_min[0] = idx


    
    def lsm_dist(x):
        if x == idx:
            k = 0
        else:
            k = Levenberg_Marquarelt(tgdrvt_origin, tsematic, candidate[x], candidate_min[0])
        cdd = (1 - k) * candidate[x] + k * candidate_min[0]
        vec = np.subtract(tsematic, cdd) * tgdrvt
        return np.sqrt(np.dot(vec, vec))


    
    idx_1 = np.argmin(list(map(lsm_dist, range(len(candidate)))))
    candidate_min[1] = candidate[idx_1]
    idx_min[1] = idx_1

    
    return (idx_min, candidate_min)

def Levenberg_Marquarelt(tgdrvt, tsematic, candidate_1, candidate_2):
    time = 50
    count = 0
    k = least_square_method(tsematic, candidate_1, candidate_2)
    if k == 0 or k == 1:
        return k

    cdd = (1 - k) * candidate_1 + k * candidate_2
    vec = np.subtract(cdd, tsematic)
    vec_last = np.dot(vec * tgdrvt, vec)
    u0 = 100
    k_best = k
    # print("Begin..", k)
    # print(candidate_2)
    # print(candidate_1)
    # print(tgdrvt)
    # print("..", k)

    JX = tgdrvt * (candidate_2 - candidate_1)
    JX0 = tgdrvt * candidate_1
    JX1 = tgdrvt * candidate_2
    JX_s = np.array([JX0, JX1])
    JXTJX_s = np.dot(JX_s, np.transpose(JX_s))
    JXTJX = np.dot(JX, JX)
    while(count < time):
        # try:
        #     delta_ks = np.linalg.solve(JXTJX_s + u0 * np.ones(shape=(2, 2)), np.dot(JX_s, vec * tgdrvt))
        # except np.linalg.LinAlgError:
        #     delta_ks = np.linalg.pinv(JXTJX_s + u0 * np.ones(shape=(2, 2))) @ np.dot(JX_s, vec * tgdrvt)
        #
        # k0 = k0 + delta_ks[0]
        # k1 = k1 + delta_ks[1]

        delta_k = -(1.0 / (JXTJX + u0)) * (np.dot(JX, vec * tgdrvt))
        k = k + delta_k

        cdd = (1 - k) * candidate_1 + k * candidate_2
        vec = np.subtract(cdd, tsematic)

        vec_now = np.dot(vec * tgdrvt, vec)
        if vec_now > vec_last:
            u0 *= 2
        else:
            k_best = k
            u0 /= 3
        vec_last = vec_now
        count += 1
    return k_best


def least_square_method(tsematic, candidate_1, candidate_2):
    numerator = np.dot(candidate_2 - candidate_1, tsematic - candidate_1)
    denominator = np.dot(candidate_1 - candidate_2, candidate_1 - candidate_2)
    if denominator < 1e-4:
        return 0
    # if np.isinf(tsematic).any() or np.isinf(denominator) or (tsematic == 2e20).any() or np.isnan(tsematic).any() or np.isinf(numerator):
    #     return 0
    if (math.isnan(numerator / denominator)):
        raise ValueError("why here..", numerator, denominator, candidate_1, candidate_2, tsematic)
    return numerator / denominator

def effect_test(tsematic, origin, candidate_1, candidate_2, k, tgdrvt, serious = False):
    cdd = (1 - k) * candidate_1 + k * candidate_2
    vec = np.subtract(tsematic, cdd) * tgdrvt
    effect = np.sqrt(np.dot(vec, vec))
    vec = np.subtract(tsematic, candidate_1) * tgdrvt
    effect_1 = np.sqrt(np.dot(vec, vec))
    vec = np.subtract(tsematic, candidate_2) * tgdrvt
    effect_2 = np.sqrt(np.dot(vec, vec))
    vec = np.subtract(tsematic, origin) * tgdrvt
    origin_effect = np.sqrt(np.dot(vec, vec))
    # if effect_1 < effect:
    #     k = 0
    #     effect = effect_1
    # if effect_2 < effect:
    #     k = 1
    #     effect = effect_2
    if not serious:
        return (effect < origin_effect or math.fabs(effect - origin_effect) < 1e-5, k, effect, origin_effect)#, origin_effect_1, origin_effect_2, effect, effect - origin_effect, origin_effect)
    else:
        return (effect < origin_effect and math.fabs(effect - origin_effect) > 1e-2, k, effect, origin_effect)#, origin_effect_1, origin_effect_2, effect, effect - origin_effect, origin_effect)



def crossover(pprogs: [Program], progs_: [Program], smts: PopSemantic, funcs, r_slt):
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
            tg_idx = smts.compute_tg(id)

            subtree3:TreeNode = smts.get_tg_node(id, tg_idx)
            rlt_posi = subtree3.rlt_posi()
            subtree3 = child.getSubTree(rlt_posi)

            h_limit = subtree3.height() - 2#child.depth - subtree3.relative_depth() - 2
            h_limit = 1 if h_limit <= 0 else h_limit

            (candidate, trs_cdd) = smts.get_smt_trs(h_limit, PyGP.SEMANTIC_NUM)
            if candidate is not None:
                tr_origin = smts.get_tg_node(id, tg_idx)
                cdd_origin = smts.get_snode_tgsmt(id, tg_idx)

                tgsmt = smts.get_tgsmt_d(id, tg_idx)
                tgdrvt_origin = smts.get_drvt_d(id, tg_idx)
                tgdrvt = np.fabs(tgdrvt_origin)

                if (np.isnan(tgdrvt).any()):
                    print("id: ", id, tg_idx, smts.semantics[id].count)
                    print('tgdrvt ', tgdrvt)
                    print('tgdrvt ', tgdrvt)
                    print('bfuncs ', smts.semantics[id].bfuncs_merge())
                    print('tgsmt ', tgsmt)
                    print('cdd_origin ', cdd_origin)
                    progs_[id].exp_draw()
                    smts.get_tg_node(id, tg_idx).exp_draw()
                    for j in range(smts.semantics[id].count):
                        smts.semantics[id].semantic[j].node.exp_draw()
                    assert (1 == 0)
                tgdrvt_f_idx = PyGP.cluster(tgdrvt)[0]
                tgdrvt = PyGP.abs_normalize(tgdrvt)

                (indiv_idx, indivs) = indivSelect_sem(tsematic=tgsmt, candidate=candidate, tgdrvt=tgdrvt, tgdrvt_f_idx=tgdrvt_f_idx, tgdrvt_origin=tgdrvt_origin)

                k:float = float(Levenberg_Marquarelt(tgdrvt_origin, tgsmt, indivs[0], indivs[1]))
                effect_better = effect_test(tgsmt, cdd_origin,
                                            indivs[0], indivs[1], k, tgdrvt, serious=True)
                if not effect_better[0] and 9 - (subtree3.relative_depth() + subtree3.height()) >= 2:
                    # a = True
                    (candidate, trs_cdd) = smts.get_smt_trs(subtree3.height(), PyGP.SEMANTIC_NUM)
                    candidate.insert(0, cdd_origin)
                    trs_cdd.insert(0, tr_origin)
                    (indiv_idx, indivs) = indivSelect_sem(tsematic=tgsmt, candidate=candidate, tgdrvt=tgdrvt, tgdrvt_f_idx=tgdrvt_f_idx, tgdrvt_origin=tgdrvt_origin)
                    k = float(Levenberg_Marquarelt(tgdrvt_origin, tgsmt, indivs[0], indivs[1]))
                    effect_better = effect_test(tgsmt,cdd_origin,
                                                indivs[0], indivs[1], k, tgdrvt, serious=True)
                    # if(not (not effect_better[0] or (indiv_idx[0] != 0 and indiv_idx[1] != 0))):
                    #     print(subtree3.height(), trs_cdd[indiv_idx[0]].height(), trs_cdd[indiv_idx[1]].height(), tr_origin.height())

                # if math.fabs(effect_better[1] - k) > 1e-6:
                #     k = effect_better[1]
                if effect_better[0] or random.uniform(0, 1) < 0.7:
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

class SMT_Weight_Crossover_LV1:
    def __call__(self, pprogs, progs, smts, funcs, r_slt=False):
        return crossover(pprogs, progs, smts, funcs, r_slt)