import random
import time

import numpy as np
import math

import PyGP
from PyGP import Program, PopSemantic, TreeNode, TR


def r_snodes_select(smt_len, num):
    slts = rd.choice(range(smt_len), size=num, replace=False)
    return np.sort(slts)

def indivSelect_sem_4(tsematic_, candidates, tgdrvt_, tgdrvt_origin_, candidate_origin_,
                      depth_limit, mask, s3_size, tr_origin, org):  # 用于语义的个体选择
    candidate_ = [candidates[i].semantic for i in range(len(candidates))]
    tsematic = tsematic_ * mask
    candidate = candidate_ * mask
    tgdrvt = tgdrvt_ * mask
    # 选一个最近点
    idx_min = [-1, -1]
    
    cdd_mean_list = list(map(lambda x: np.mean(x), candidate))
    y_mean = np.mean(tsematic)
    b_list = list(map(lambda x: np.cov(candidate[x], tsematic)[0][1] / np.var(candidate[x]),#candidate[x] - cdd_mean_list[x]) * (tsematic - y_mean) / ((tsematic - y_mean) * (tsematic - y_mean)), 
                      range(len(candidate))))
    a_list = list(map(lambda x: y_mean - b_list[x] * cdd_mean_list[x], range(len(candidate))))


    rsdls_ = list(map(lambda x: np.subtract(tsematic, candidate[x] * b_list[x] + a_list[x]), range(len(candidate))))
    # dis_all_w = list(map(lambda x: np.sqrt(np.dot(x * tgdrvt_, x)), rsdls_))  # 加权距离
    dis_all_w = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls_))  # 加权距离
    dis_sorted = np.argsort(dis_all_w)

    candidate_min = [candidate[dis_sorted[0]], candidate[len(candidate) - 1]]
    idx_min = [int(dis_sorted[0]), int(len(candidate) - 1)]
    k = [b_list[dis_sorted[0]], a_list[dis_sorted[0]]]
    
    # 返回该两个点
    return (idx_min, candidate_min, None, k, True)


def indivSelect_sem_blist(tsematic_, candidate_, tgdrvt_, tgdrvt_origin_, candidate_origin_, origin_size, trs_size,
                          depth_limit, mask, idx=None):  # 用于语义的个体选择
    tsematic = tsematic_ * mask
    candidate = candidate_ * mask
    tgdrvt = tgdrvt_ * mask
    tgdrvt_origin = tgdrvt_origin_ * mask
    candidate_origin = candidate_origin_ * mask
    # 选一个最近点
    idx_min = [-1, -1]
    candidate_min = [candidate[0], candidate[1]]
    # tgdrvt_f = tgdrvt[tgdrvt_f_idx]
    # tsematic_f = tsematic[tgdrvt_f_idx]
    # candidate_f = list(map(lambda x: x[tgdrvt_f_idx], candidate))

    rsdls_cur = np.subtract(tsematic, candidate_origin)
    dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt, rsdls_cur))

    center = np.sum(candidate, axis=0) / len(candidate)

    tgsmt_mean = np.dot(tsematic, tsematic)
    cdd_mean2_list = list(map(lambda x: np.dot(x, x), candidate))
    b_list = list(map(lambda x: np.dot(candidate[x], tsematic) / cdd_mean2_list[x] if cdd_mean2_list[x] != 0 else 0,
                      range(len(candidate))))

    # rsdls = list(map(lambda x: np.subtract(center, x), candidate))
    # # dis_all = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls))
    # dis_all_w = list(map(lambda x: np.sqrt(np.dot(x * tgdrvt, x)), rsdls))#加权距离
    rsdls_ = list(map(lambda x: np.subtract(tsematic, candidate[x] * b_list[x]), range(len(candidate))))

    # dis_all = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls))
    dis_all_w_ = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls_))  # 加权距离
    # dis_all_w = np.add(dis_all_w_, dis_all_w)
    # rsdls_f = list(map(lambda x: np.subtract(tsematic_f, x), candidate_f))
    # dis_all_f_w = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls_f))
    # idx = np.argmin(dis_all_w)
    # tsematic_l = np.sqrt(np.dot(tsematic, tsematic))
    # angles_cos = np.array(list(map(lambda x: np.dot(x, tsematic) / (np.sqrt(np.dot(x, x)) * tsematic_l), candidate)))

    dis_sorted = np.argsort(dis_all_w_)
    y_tmp = tsematic - tgsmt_mean
    x_tmp = list(map(lambda i: candidate[i] - np.sqrt(cdd_mean2_list[i]), range(len(candidate))))
    y_tmp_sigma = np.sqrt(np.dot(y_tmp, y_tmp))
    cdd_sigma = list(map(lambda i: np.sqrt(np.dot(x_tmp[i], x_tmp[i])), range(len(candidate))))
    r = list(map(lambda i: np.dot(x_tmp[i], y_tmp) / (y_tmp_sigma * cdd_sigma[i]) if cdd_sigma[
                                                                                         i] != 0 and y_tmp_sigma != 0 else (
        1 if cdd_sigma[i] == y_tmp_sigma else -1), range(len(candidate))))

    cdd_origin_tmp = candidate_origin - np.sqrt(np.dot(candidate_origin, candidate_origin))
    cdd_origin_sigma = np.sqrt(np.dot(cdd_origin_tmp, cdd_origin_tmp))
    cdd_origin_r = np.dot(cdd_origin_tmp, y_tmp) / (
                y_tmp_sigma * cdd_origin_sigma) if cdd_origin_sigma != 0 and y_tmp_sigma != 0 else (
        1 if cdd_origin_sigma == y_tmp_sigma else -1)
    min_size = origin_size
    r_min = -1
    if idx is None:
        init_id = 0
        idx = dis_sorted[init_id]
        while init_id < 5 and init_id < len(dis_sorted) and dis_all_w_[dis_sorted[init_id]] < dis_cur:

            # if trs_size[dis_sorted[init_id]] < min_size:
            #     min_size = trs_size[dis_sorted[init_id]]
            #     idx = dis_sorted[init_id]
            #     # break

            if r[dis_sorted[init_id]] > r_min:
                r_min = r[dis_sorted[init_id]]
                idx = dis_sorted[init_id]
            if r[dis_sorted[init_id]] > 1 or r[dis_sorted[init_id]] < -1:
                print("!!!!!!!!!!!!!!!!!!wrong..", r[dis_sorted[init_id]])

            init_id += 1
    # if idx is None:
    #     idx = np.argmax(angles_cos)

    # dis_min = dis_all[idx]
    candidate_min[0] = candidate[idx]
    idx_min[0] = idx

    tsematic_mean = np.mean(tsematic)
    cdd_mean = np.mean(candidate_min[0])

    b = [b_list[idx]]
    #
    # if cdd_mean != 0:
    #     cdd_mean_s = np.dot(candidate_min[0], candidate_min[0])
    #     b = -np.dot(candidate_min[0], tsematic) / cdd_mean_s

    # for i in range(len(tsematic)):
    #     b += (tsematic[i] - tsematic_mean) / (candidate_min[0][i] - cdd_mean)
    # print('=====================', b, tsematic_mean, cdd_mean)
    # print('candidate_min=====================', dis_all_w[idx], idx, np.mean(center), np.argmin(dis_all_w))
    # if dis_all_w[idx] < dis_cur and trs_size[idx] < origin_size:
    #     return (idx_min[0], candidate_min[0])
    k_list = list(map(lambda x: least_square_method(tsematic, candidate_min[0] * b_list[idx], candidate[x], tgdrvt),
                      range(len(candidate))))

    # 直接最小二乘计算出k值后估计

    tgsmt_mean_ = np.dot(tsematic_, tsematic_)
    y_tmp_ = tsematic_ - tgsmt_mean_
    y_tmp_sigma_ = np.sqrt(np.dot(y_tmp_, y_tmp_))

    def lsm_dist(x):
        [k0, k1, vec, vec_1] = Levenberg_Marquarelt_2(tgdrvt_, tgdrvt_origin_, tsematic_, candidate_[idx] * b_list[idx],
                                                      candidate_[x], 0, r_=[y_tmp_, y_tmp_sigma_],
                                                      init_k=[1 - k_list[x], k_list[x]])
        # return (vec, vec_1)
        return vec

    # 以该最近点为基础，选另一个横线上的最近点
    # res = list(map(lsm_dist, range(len(candidate))))
    # res_0, res_1 = np.argsort([vec[0] for vec in res]), np.argsort([vec[1] for vec in res])
    # res = np.zeros(len(res))
    # for i in range(len(res_0)):
    #     res[res_0[i]] += i
    #     res[res_1[i]] += i
    # res = np.argsort(res)
    lsm_vals = list(map(lsm_dist, range(len(candidate))))
    lsm_sorted = np.argsort(lsm_vals)[::-1]
    init_id = 0
    idx_1 = lsm_sorted[init_id]
    # idx_1 = idx_sorted[0]
    # while idx_1 < len(idx_sorted):

    # vec = np.subtract(tgsmt, cdd_origin)
    # origin_effect = np.sqrt(np.dot(vec * tgdrvt_test, vec))

    succeed = False
    # init_id = 0
    # idx_1 = lsm_sorted[init_id]
    # while init_id < len(lsm_sorted) and lsm_vals[lsm_sorted[init_id]] < dis_cur:
    #     cdd_size = (trs_size[idx_min[0]], trs_size[lsm_sorted[init_id]], origin_size, depth_limit)
    #     len_adv = (
    #             2 ** cdd_size[3] - (float(cdd_size[0] + cdd_size[1] + 3) - float(cdd_size[2])))
    #     if len_adv > 0 and math.fabs(dis_cur / lsm_vals[lsm_sorted[init_id]]) > 2 ** cdd_size[3] / len_adv:
    #         idx_1 = lsm_sorted[init_id]
    #         succeed = True
    #         break
    #     init_id += 1

    b.append(b_list[idx_1])
    candidate_min[1] = candidate[idx_1]
    idx_min[1] = idx_1

    [k0, k1, vec, vec_1] = Levenberg_Marquarelt_2(tgdrvt, tgdrvt_origin, tsematic,
                                                  candidate_min[0] * b_list[idx_min[0]], candidate[idx_1], 0)
    if lsm_vals[idx_1] > cdd_origin_r:
        succeed = True
        # [k0_, k1_, vec, vec_1] = Levenberg_Marquarelt_2(tgdrvt, tgdrvt_origin, tsematic, candidate_min[0] * b_list[idx_min[0]], candidate[idx_1], 200)
        # if vec > lsm_vals[idx_1]:
        #     k = [k0_ * b_list[idx_min[0]], k1_]
        # else:
        k = [k0 * b_list[idx_min[0]], k1]
    else:
        k = [k0 * b_list[idx_min[0]], k1]
    # 返回该两个点
    return (idx_min, candidate_min, b, k, succeed)


def indivSelect_sem_3(tsematic_, candidates, tgdrvt_, tgdrvt_origin_, candidate_origin_,
                      depth_limit, mask, s3_size, tr_origin, org):  # 用于语义的个体选择
    candidate_ = [candidates[i].semantic for i in range(len(candidates))]
    tsematic = tsematic_ * mask
    candidate = candidate_ * mask
    tgdrvt = tgdrvt_ * mask
    # 选一个最近点
    idx_min = [-1, -1]
    candidate_min = [candidate[0], candidate[1]]
    k = None
    rsdls_cur = np.subtract(tsematic_, candidate_origin_)
    dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt, rsdls_cur))
    
    # dis_cur_vec = np.sqrt(np.dot(rsdls_cur * tgdrvt, rsdls_cur) / len(rsdls_cur)) / (
    #             0.9999 ** (s3_size[1]))

    if PyGP.NEW_BVAL:
        cdd_mean2_list = list(map(lambda x: np.mean(x), candidate))
        
        y_mean = np.mean(tsematic)
        b_list = list(map(lambda x: (np.dot(candidate[x] - cdd_mean2_list[x], tsematic - y_mean) / len(candidate)) / np.var(candidate[x]),#candidate[x] - cdd_mean_list[x]) * (tsematic - y_mean) / ((tsematic - y_mean) * (tsematic - y_mean)), 
                        range(len(candidate))))
    else:
        cdd_mean2_list = list(map(lambda x: np.dot(x, x), candidate))
        b_list = list(map(lambda x: np.dot(candidate[x], tsematic) / cdd_mean2_list[x] if cdd_mean2_list[x] != 0 else 0,
                        range(len(candidate))))

    rsdls_ = list(map(lambda x: np.subtract(tsematic, candidate[x] * b_list[x]), range(len(candidate))))

    dis_all_w = list(map(lambda x: np.sqrt(np.dot(x * tgdrvt_, x)), rsdls_))  # 加权距离
    dis_sorted = np.argsort(dis_all_w)

    E_2 = np.sqrt(np.dot(rsdls_cur * tgdrvt_ - np.mean(rsdls_cur * tgdrvt_),
                         rsdls_cur * tgdrvt_ - np.mean(rsdls_cur * tgdrvt_)) / len(rsdls_cur))

    init_num = 1#int(np.sqrt(PyGP.SEMANTIC_NUM))
    min_vec = (2e20, 2e20)
    vec_tmp = []
    vec_tmp_ = []
    for i in range(init_num):
        
        tr_0 = None
        if init_num >= len(dis_sorted):
            break
        cdd_0 = candidate_[dis_sorted[i]]
        tg = i + 1
        
        # new_size = candidates[dis_sorted[i]].tree.inner_size + 3 - s3_size[0] + s3_size[1]

        while tg - (i) < PyGP.SEMANTIC_NUM / init_num and tg < len(dis_sorted):
            # if new_size + candidates[dis_sorted[tg]].tree.inner_size > PyGP.SIZE_LIMIT / 2.:
            #     tg += 1
            #     continue
            cdd_1 = candidate_[dis_sorted[tg]]
            k_ = least_square_method(tsematic, candidate[dis_sorted[i]] * b_list[dis_sorted[i]],
                                     candidate[dis_sorted[tg]] * b_list[dis_sorted[tg]], tgdrvt_)

            cdd = (1 - k_) * cdd_0 * b_list[dis_sorted[i]] + k_ * cdd_1 * b_list[dis_sorted[tg]]
            rsdls_0 = np.subtract(tsematic_, cdd) * tgdrvt_
            E_0 = np.sqrt(np.dot(rsdls_0 - np.mean(rsdls_0),
                                 rsdls_0 - np.mean(rsdls_0)) / len(rsdls_0))

            vec_ = np.subtract(cdd, tsematic_)
            # vec = np.sqrt(np.dot(vec_ * tgdrvt, vec_))
            vec = np.sqrt(np.dot(vec_ * tgdrvt, vec_))
            

            if vec < dis_cur * 1.01 and E_0 < E_2 * 1.01:
                # if tr_0 is None:
                #     tr_0 = candidates[dis_sorted[i]].tree
                # tr_1 = candidates[dis_sorted[tg]].tree
                # tr_0_size, tr_1_size = tr_0.inner_size, tr_1.inner_size
                # tmp_size = tr_0_size + tr_1_size + 3
                # vec = np.sqrt(np.dot(vec_ * tgdrvt, vec_) / len(vec_)) / (
                #             0.9999 ** (s3_size[1] - s3_size[0] + tmp_size))
                k = [(1 - k_) * b_list[dis_sorted[i]], k_ * b_list[dis_sorted[tg]]]
                vec_tmp.append([vec, dis_sorted[i], dis_sorted[tg], k[0], k[1]])
                vec_tmp_.append(vec)
            tg += 1
    succeed = False

    if len(vec_tmp) > 0:
        v_sort = np.argsort(vec_tmp_)

        for id, v_id in enumerate(v_sort):
            k = [vec_tmp[v_id][3], vec_tmp[v_id][4]]
            k0_tmp = [k[0]]
            k1_tmp = [k[1]]
            cdd_0_ = candidate_[int(vec_tmp[v_id][1])]
            cdd_1_ = candidate_[int(vec_tmp[v_id][2])]

            candidate_min = [cdd_0_, cdd_1_]
            idx_min = [int(vec_tmp[v_id][1]), int(vec_tmp[v_id][2])]
            trs_cdd = [candidates[idx_min[0]].tree.inner_size, candidates[idx_min[1]].tree.inner_size]
            new_size = trs_cdd[0] + trs_cdd[1] + 3 - s3_size[0] + s3_size[1]
            cdd_size = (trs_cdd[0], trs_cdd[1], s3_size[0], s3_size[1], depth_limit)
            if new_size < PyGP.SIZE_LIMIT / 2. and effect_test(tsematic_, candidate_origin_,
                            cdd_0_, cdd_1_, k, tgdrvt_, cdd_size, True, mask):
                succeed = True
                cdd = k[0] * cdd_0_ + k[1] * cdd_1_
                # E_0 = np.sqrt(np.dot(cdd - np.mean(cdd), cdd - np.mean(cdd)) / len(cdd))
                # E_1 = np.sqrt(np.dot(tsematic_ - np.mean(tsematic_), tsematic_ - np.mean(tsematic_)) / len(tsematic_))
                k = [k[0], k[1]]

                break
    # 返回该两个点
    return (idx_min, candidate_min, None, k, succeed)


def indivSelect_sem(tsematic_, candidates, tgdrvt_, tgdrvt_origin_, candidate_origin_,
                    depth_limit, mask, s3_size, tr_origin, org):  # 用于语义的个体选择
    candidate_ = [candidates[i].semantic for i in range(len(candidates))]
    tsematic = tsematic_ * mask
    candidate = candidate_ * mask
    tgdrvt = tgdrvt_ * mask
    # tgdrvt_origin = tgdrvt_origin_ * mask
    # candidate_origin = candidate_origin_ * mask
    #
    # tgdrvt_r = tgdrvt_ * ((1 + mask) % 2)

    # E_cdd = [np.sqrt(np.dot(x - np.mean(x),
    #                      x - np.mean(x)) / len(x)) for x in candidate]
    # E_tg = np.sqrt(np.dot(tsematic - np.mean(tsematic),tsematic - np.mean(tsematic)) / len(tsematic))
    #
    # cdds_map = [x * E_tg / E_cdd for x in candidate]
    # E_1 = np.sqrt(np.dot(tsematic_ - np.mean(tsematic_),
    #                      tsematic_ - np.mean(tsematic_)) / len(tsematic_))

    # 选一个最近点
    idx_min = [-1, -1]
    candidate_min = [candidate[0], candidate[1]]
    k = None
    rsdls_cur = np.subtract(tsematic_, candidate_origin_)
    dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt, rsdls_cur))
    # dis_cur_c = np.sqrt(np.dot(rsdls_cur, rsdls_cur))
    # mlen_cur = np.max(np.fabs(rsdls_cur))
    #
    # dis_cur_r = np.sqrt(np.dot(rsdls_cur * tgdrvt_r, rsdls_cur))
    #
    # center = np.sum(candidate, axis=0) / len(candidate)

    # tgsmt_mean = np.dot(tsematic, tsematic)
    # cdd_mean2_list = list(map(lambda x: np.dot(x, x), candidate))

    cdd_mean2_list = list(map(lambda x: np.dot(x, x), candidate))
    b_list = list(map(lambda x: np.dot(candidate[x], tsematic) / cdd_mean2_list[x] if cdd_mean2_list[x] != 0 else 0,
                      range(len(candidate))))

    rsdls_ = list(map(lambda x: np.subtract(tsematic, candidate[x] * b_list[x]), range(len(candidate))))

    dis_all_w = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls_))  # 加权距离
    dis_sorted = np.argsort(dis_all_w)
    # dis_all_w_1 = list(map(lambda x: np.sqrt(np.dot(x * tgdrvt, x)), rsdls_))  # 加权距离
    # dis_sorted_1 = np.argsort(dis_all_w_1)

    E_2 = np.sqrt(np.dot(rsdls_cur - np.mean(rsdls_cur),
                         rsdls_cur - np.mean(rsdls_cur)) / len(rsdls_cur))

    # for i in range(0):
    #     cdd_0_ = candidate_[dis_sorted[i]]
    #     if not (dis_all_w[dis_sorted[i]] < dis_cur_c):
    #         break
    #     succeed = True
    #     min_vec = 10000
    #     # for z in range(0):

    #     #     mask_ = rd.choice([0, 1], size=len(tsematic_))

    #     #     tgdrvt = tgdrvt_ * mask_
    #     #     tgdrvt_r = tgdrvt_ * ((1 + mask_) % 2)
    #     #     tsematic = tsematic_ * mask_

    #     #     cdd_0 = cdd_0_ * mask_
    #     #     cdd_mean = np.dot(cdd_0, cdd_0)
    #     #     b = np.dot(cdd_0, tsematic) / cdd_mean if cdd_mean != 0 else 0

    #     #     cdd = cdd_0 * b
    #     #     # vec_ = np.subtract(cdd, tsematic_)
    #     #     # vec = np.sqrt(np.dot(vec_ * tgdrvt, vec_))

    #     #     # mlen_vec = np.max(np.fabs(vec_ * tgdrvt))
    #     #     # vec_1 = np.sqrt(np.dot(vec_ * tgdrvt_r, vec_))
    #     #     # mlen_vec_1 = np.max(np.fabs(vec_ * tgdrvt_r))

    #     #     # dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt, rsdls_cur))
    #     #     # mlen_cur = np.max(np.fabs(rsdls_cur * tgdrvt))
    #     #     # dis_cur_r = np.sqrt(np.dot(rsdls_cur * tgdrvt_r, rsdls_cur))
    #     #     # mlen_cur_r = np.max(np.fabs(rsdls_cur * tgdrvt_r))

    #     #     # if not((vec < dis_cur) and (vec_1 < dis_cur_r)):
    #     #     #     succeed = False
    #     #     #     break

    #     #     rsdls_0 = np.subtract(tsematic_, cdd_0_ * b)
    #     #     E_0 = np.sqrt(np.dot(rsdls_0 - np.mean(rsdls_0),
    #     #                          rsdls_0 - np.mean(rsdls_0)) / len(rsdls_0))
    #     #     # E_1 = np.sqrt(np.dot(tsematic_ - np.mean(tsematic_),
    #     #     #                      tsematic_ - np.mean(tsematic_)) / len(tsematic_))

    #     #     vec_ = np.subtract(cdd_0_ * b, tsematic_)
    #     #     vec = np.sqrt(np.dot(vec_ * tgdrvt_, vec_))
    #     #     dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt_, rsdls_cur))
    #     #     if not(vec < dis_cur and E_0 < E_2):
    #     #         succeed = False
    #     #         break

    #     if succeed:

    #         cdd_mean = np.dot(cdd_0_, cdd_0_)
    #         b = np.dot(cdd_0_, tsematic_) / cdd_mean if cdd_mean != 0 else 0

    #         # [k0_, k1_, vec, vec_1] = Levenberg_Marquarelt_2(tgdrvt_, tgdrvt_origin_, tsematic_,
    #         #                                                 cdd_0_ * b, cdd_1_, 200, k)
    #         # k = [k0_, k1_]

    #         # k = [np.mean(k0_tmp), np.mean(k1_tmp)]

    #         tgdrvt = tgdrvt_ * mask

    #         tgdrvt_r = tgdrvt_ * ((1 + mask) % 2)

    #         cdd = b * cdd_0_

    #         rsdls_0 = np.subtract(tsematic_, cdd)
    #         E_0 = np.sqrt(np.dot(rsdls_0 - np.mean(rsdls_0),
    #                              rsdls_0 - np.mean(rsdls_0)) / len(rsdls_0))
    #         # E_1 = np.sqrt(np.dot(tsematic_ - np.mean(tsematic_),
    #         #                      tsematic_ - np.mean(tsematic_)) / len(tsematic_))

    #         # if E_0 == 0 and E_2 != 0:
    #         #     continue
    #         vec_ = np.subtract(cdd, tsematic_)
    #         vec = np.sqrt(np.dot(vec_ * tgdrvt, vec_))
    #         # vec_1 = np.sqrt(np.dot(vec_ * tgdrvt_r, vec_))
    #         #
    #         # mlen_vec = np.max(np.fabs(vec_ * tgdrvt))

    #         dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt, rsdls_cur))

    #         # dis_cur_r = np.sqrt(np.dot(rsdls_cur * tgdrvt_r, rsdls_cur))
    #         # mlen_cur = np.max(np.fabs(rsdls_cur * tgdrvt))

    #         if not (vec < dis_cur):
    #             succeed = False
    #         else:

    #             candidate_min = cdd_0_
    #             idx_min = dis_sorted[i]
    #             # if org:
    #             #     trs_cdd = [smts.get_tr(trs[idx_min[0] - 1]) if idx_min[0] > 0 else tr_origin,
    #             #            smts.get_tr(trs[idx_min[1] - 1]) if idx_min[1] > 0 else tr_origin]
    #             # else:
    #             trs_cdd = candidates[idx_min].tree
    #             longer = trs_cdd.inner_size + 1 >= s3_size
    #             cdd_size = (trs_cdd.inner_size, None, s3_size, depth_limit)

    #             if (vec == 0 or
    #             ((2 ** depth_limit - (float(trs_cdd.inner_size + 1) - float(s3_size))) > 0 and
    #             math.fabs(dis_cur / vec) > 2 ** depth_limit / (2 ** depth_limit - (float(trs_cdd.inner_size + 1) - float(s3_size))))):

    #                 k=b
    #                 return (idx_min, candidate_min, None, k, succeed)
    #             else:
    #                 succeed=False

    # y_tmp = tsematic - tgsmt_mean
    # x_tmp = list(map(lambda i: candidate[i] - np.sqrt(cdd_mean2_list[i]), range(len(candidate))))
    # y_tmp_sigma = np.sqrt(np.dot(y_tmp, y_tmp))
    # cdd_sigma = list(map(lambda i: np.sqrt(np.dot(x_tmp[i], x_tmp[i])), range(len(candidate))))
    # r = list(map(lambda i: np.dot(x_tmp[i], y_tmp) / (y_tmp_sigma * cdd_sigma[i]) if cdd_sigma[
    #                                                                                      i] != 0 and y_tmp_sigma != 0 else (
    #     1 if cdd_sigma[i] == y_tmp_sigma else -1), range(len(candidate))))

    # cdd_origin_tmp = candidate_origin - np.sqrt(np.dot(candidate_origin, candidate_origin))
    # cdd_origin_sigma = np.sqrt(np.dot(cdd_origin_tmp, cdd_origin_tmp))
    # cdd_origin_r = np.dot(cdd_origin_tmp, y_tmp) / (
    #             y_tmp_sigma * cdd_origin_sigma) if cdd_origin_sigma != 0 and y_tmp_sigma != 0 else (
    #     1 if cdd_origin_sigma == y_tmp_sigma else -1)

    # tgsmt_mean_ = np.dot(tsematic_, tsematic_)
    # y_tmp_ = tsematic_ - tgsmt_mean_
    # y_tmp_sigma_ = np.sqrt(np.dot(y_tmp_, y_tmp_))

    init_num = int(PyGP.SEMANTIC_NUM / 10)
    min_vec = (2e20, 2e20)
    vec_tmp = []
    vec_tmp_ = []
    for i in range(init_num):

        if init_num >= len(dis_sorted):
            break
        cdd_0 = candidate_[dis_sorted[i]]
        tg = i + 1
        while tg - (i) < PyGP.SEMANTIC_NUM / init_num and tg < len(dis_sorted):
            cdd_1 = candidate_[dis_sorted[tg]]
            k_ = least_square_method(tsematic, candidate[dis_sorted[i]] * b_list[dis_sorted[i]],
                                     candidate[dis_sorted[tg]] * b_list[dis_sorted[tg]], tgdrvt)
            # [k0, k1, vec, vec_1] = Levenberg_Marquarelt_2(tgdrvt, tgdrvt_origin, tsematic,
            #                                               candidate[dis_sorted[i]] * b_list[dis_sorted[i]],
            #                                               candidate[dis_sorted[tg]], 0, r_=[y_tmp_, y_tmp_sigma_],
            #                                               init_k=[1 - k_, k_])

            cdd = (1 - k_) * cdd_0 * b_list[dis_sorted[i]] + k_ * cdd_1 * b_list[dis_sorted[tg]]

            # E_0 = np.sqrt(
            #     np.dot(cdd - np.mean(cdd), cdd - np.mean(cdd)) / len(cdd))
            # E_1 = np.sqrt(np.dot(tsematic_ - np.mean(tsematic_),
            #                      tsematic_ - np.mean(tsematic_)) / len(tsematic_))
            # E_2 = np.sqrt(np.dot(candidate_origin_ - np.mean(candidate_origin_),
            #                      candidate_origin_ - np.mean(candidate_origin_)) / len(candidate_origin_))

            rsdls_0 = np.subtract(tsematic_, cdd)
            E_0 = np.sqrt(np.dot(rsdls_0 - np.mean(rsdls_0),
                                 rsdls_0 - np.mean(rsdls_0)) / len(rsdls_0))
            # E_1 = np.sqrt(np.dot(tsematic_ - np.mean(tsematic_),
            #                      tsematic_ - np.mean(tsematic_)) / len(tsematic_))

            # if E_0 == 0 and E_1 != 0:
            #     tg += 1
            #     continue

            vec_ = np.subtract(cdd, tsematic_)
            vec = np.sqrt(np.dot(vec_ * tgdrvt, vec_))
            # vec_1 = np.sqrt(np.dot(vec_ * tgdrvt_r, vec_))
            # mlen_vec = np.max(np.fabs(vec_ * tgdrvt))
            # mlen_cur = np.max(np.fabs(rsdls_cur * tgdrvt))

            if vec < dis_cur:
                # inner_1 = candidates[dis_sorted[i]].inner_size
                # inner_2 = candidates[dis_sorted[tg]].inner_size
                # vec = math.e ** (- (dis_cur - vec)) / math.e ** (-(inner_1 + inner_2 + 3 - s3_size))
                # min_vec = (vec, vec_1)
                # candidate_min = [cdd_0, cdd_1]
                # idx_min = [dis_sorted[i], dis_sorted[tg]]
                k = [(1 - k_) * b_list[dis_sorted[i]], k_ * b_list[dis_sorted[tg]]]
                vec_tmp.append([vec, dis_sorted[i], dis_sorted[tg], k[0], k[1]])
                vec_tmp_.append(vec)
            tg += 1
    succeed = False

    # vec_tmp.append([min_vec[0] + min_vec[1], idx_min[0], idx_min[1], k[0], k[1]])

    # print(min_vec, dis_cur)
    if len(vec_tmp) > 0:
        v_sort = np.argsort(vec_tmp_)

        for id, v_id in enumerate(v_sort):
            succeed = True
            k = [vec_tmp[v_id][3], vec_tmp[v_id][4]]
            k0_tmp = [k[0]]
            k1_tmp = [k[1]]
            cdd_0_ = candidate_[int(vec_tmp[v_id][1])]
            cdd_1_ = candidate_[int(vec_tmp[v_id][2])]
            # for z in range(0):

            #     mask_ = rd.choice([0, 1, 1, 1], size=len(tsematic_))

            #     tgdrvt = tgdrvt_ * mask_
            #     # tgdrvt_r = tgdrvt_ * ((1 + mask_) % 2)
            #     tsematic = tsematic_ * mask_

            #     cdd_0 = cdd_0_ * mask_
            #     cdd_1 = cdd_1_ * mask_
            #     cdd_mean = np.dot(cdd_0, cdd_0)
            #     b = np.dot(cdd_0, tsematic) / cdd_mean if cdd_mean != 0 else 0
            #     k_ = least_square_method(tsematic, cdd_0 * b, cdd_1, tgdrvt)

            #     k0_tmp.append((1 - k_) * b)
            #     k1_tmp.append(k_)

            #     cdd = (1 - k_) * cdd_0_ * b + k_ * cdd_1_
            #     # vec_ = np.subtract(cdd, tsematic_)
            #     # vec = np.sqrt(np.dot(vec_ * tgdrvt, vec_))

            #     # mlen_vec = np.max(np.fabs(vec_ * tgdrvt))
            #     # vec_1 = np.sqrt(np.dot(vec_ * tgdrvt_r, vec_))
            #     # mlen_vec_1 = np.max(np.fabs(vec_ * tgdrvt_r))

            #     # dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt, rsdls_cur))
            #     # mlen_cur = np.max(np.fabs(rsdls_cur * tgdrvt))
            #     # dis_cur_r = np.sqrt(np.dot(rsdls_cur * tgdrvt_r, rsdls_cur))
            #     # mlen_cur_r = np.max(np.fabs(rsdls_cur * tgdrvt_r))

            #     # if not((vec < dis_cur) and (vec_1 < dis_cur_r)):
            #     #     succeed = False
            #     #     break

            #     rsdls_0 = np.subtract(tsematic_, cdd)
            #     E_0 = np.sqrt(np.dot(rsdls_0 - np.mean(rsdls_0),
            #                          rsdls_0 - np.mean(rsdls_0)) / len(rsdls_0))
            #     # E_1 = np.sqrt(np.dot(tsematic_ - np.mean(tsematic_),
            #     #                      tsematic_ - np.mean(tsematic_)) / len(tsematic_))

            #     vec_ = np.subtract(cdd, tsematic_)
            #     vec = np.sqrt(np.dot(vec_ * tgdrvt_, vec_))
            #     dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt_, rsdls_cur))
            #     if not(vec < dis_cur and E_0 < E_2):
            #         succeed = False
            #         break

            if succeed:

                # # cdd_mean = np.dot(cdd_0_, cdd_0_)
                # # b = np.dot(cdd_0_, tsematic_) / cdd_mean if cdd_mean != 0 else 0
                # # k_ = least_square_method(tsematic_, cdd_0_ * b, cdd_1_, tgdrvt_)
                # # k = [(1 - k_) * b, k_]
                # # k = [k[0], k[1]]

                # # [k0_, k1_, vec, vec_1] = Levenberg_Marquarelt_2(tgdrvt_, tgdrvt_origin_, tsematic_,
                # #                                                 cdd_0_ * b, cdd_1_, 200, k)
                # # k = [k0_, k1_]

                # # k = [np.mean(k0_tmp), np.mean(k1_tmp)]

                # tgdrvt = tgdrvt_ * mask

                # tgdrvt_r = tgdrvt_ * ((1 + mask) % 2)

                # cdd = k[0] * cdd_0_ + k[1] * cdd_1_

                # # E_0 = np.sqrt(
                # #     np.dot(cdd - np.mean(cdd), cdd - np.mean(cdd)) / len(cdd))
                # # E_1 = np.sqrt(np.dot(tsematic_ - np.mean(tsematic_),
                # #                      tsematic_ - np.mean(tsematic_)) / len(tsematic_))
                # # E_2 = np.sqrt(np.dot(candidate_origin_ - np.mean(candidate_origin_),
                # #                      candidate_origin_ - np.mean(candidate_origin_)) / len(candidate_origin_))

                # rsdls_0 = np.subtract(tsematic_, cdd)
                # E_0 = np.sqrt(np.dot(rsdls_0 - np.mean(rsdls_0),
                #                      rsdls_0 - np.mean(rsdls_0)) / len(rsdls_0))
                # # E_1 = np.sqrt(np.dot(tsematic_ - np.mean(tsematic_),
                # #                      tsematic_ - np.mean(tsematic_)) / len(tsematic_))

                # vec_ = np.subtract(cdd, tsematic_)
                # vec = np.sqrt(np.dot(vec_ * tgdrvt, vec_))
                # # vec_1 = np.sqrt(np.dot(vec_ * tgdrvt_r, vec_))

                # dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt, rsdls_cur))

                # # dis_cur_r = np.sqrt(np.dot(rsdls_cur * tgdrvt_r, rsdls_cur))
                # #
                # # mlen_vec = np.max(np.fabs(vec_ * tgdrvt))
                # #
                # # mlen_cur = np.max(np.fabs(rsdls_cur * tgdrvt))

                # if not (vec < dis_cur and E_0 < E_2):
                #     succeed = False
                # else:
                if True:

                    candidate_min = [cdd_0_, cdd_1_]
                    idx_min = [int(vec_tmp[v_id][1]), int(vec_tmp[v_id][2])]
                    # if org:
                    #     trs_cdd = [smts.get_tr(trs[idx_min[0] - 1]) if idx_min[0] > 0 else tr_origin,
                    #            smts.get_tr(trs[idx_min[1] - 1]) if idx_min[1] > 0 else tr_origin]
                    # else:
                    trs_cdd = [candidates[idx_min[0]].tree, candidates[idx_min[1]].tree]
                    longer = trs_cdd[0].inner_size + trs_cdd[1].inner_size + 3 >= s3_size
                    cdd_size = (trs_cdd[0].inner_size, trs_cdd[1].inner_size, s3_size, depth_limit)
                    if effect_test(tsematic_, candidate_origin_,
                                   cdd_0_, cdd_1_, k, tgdrvt_, cdd_size, True, mask):

                        cdd = k[0] * cdd_0_ + k[1] * cdd_1_

                        # E_0 = np.sqrt(np.dot(cdd - np.mean(cdd), cdd - np.mean(cdd)) / len(cdd))
                        # E_1 = np.sqrt(np.dot(tsematic_ - np.mean(tsematic_), tsematic_ - np.mean(tsematic_)) / len(tsematic_))
                        k = [k[0], k[1]]

                        break
                    else:
                        succeed = False
        # if not succeed:
        #     succeed = True
        #     k = [vec_tmp[v_sort[0]][3], vec_tmp[v_sort[0]][4]]
        #     cdd_0_ = candidate_[int(vec_tmp[v_sort[0]][1])]
        #     cdd_1_ = candidate_[int(vec_tmp[v_sort[0]][2])]
        #     candidate_min = [cdd_0_, cdd_1_]
        #     idx_min = [int(vec_tmp[v_sort[0]][1]), int(vec_tmp[v_sort[0]][2])]

    # 返回该两个点
    return (idx_min, candidate_min, None, k, succeed)


def indivSelect_sem_2(tsematic_, candidates, tgdrvt_, tgdrvt_origin_, candidate_origin_,
                      depth_limit, mask, s3_size, tr_origin, org):  # 用于语义的个体选择
    candidate_ = [candidates[i].semantic for i in range(len(candidates))]
    tsematic = tsematic_ * mask
    candidate = candidate_ * mask
    tgdrvt = tgdrvt_ * mask

    # 选一个最近点
    idx_min = [-1, -1]
    candidate_min = [candidate[0], candidate[1]]
    k = None
    rsdls_cur = np.subtract(tsematic_, candidate_origin_)
    dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt, rsdls_cur))
    dis_cur_c = np.sqrt(np.dot(rsdls_cur, rsdls_cur))

    # cdd_mean2_list = list(map(lambda x: np.dot(x, x), candidate))
    # b_list = list(map(lambda x: np.dot(candidate[x], tsematic) / cdd_mean2_list[x] if cdd_mean2_list[x] != 0 else 0, range(len(candidate))))

    rsdls_ = list(map(lambda x: np.subtract(tsematic, candidate[x] * b_list[x]), range(len(candidate))))

    dis_all_w = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls_))  # 加权距离
    dis_sorted = np.argsort(dis_all_w)

    E_2 = np.sqrt(np.dot(rsdls_cur - np.mean(rsdls_cur),
                         rsdls_cur - np.mean(rsdls_cur)) / len(rsdls_cur))

    # tgsmt_mean_ = np.dot(tsematic_, tsematic_)
    # y_tmp_ = tsematic_ - tgsmt_mean_
    # y_tmp_sigma_ = np.sqrt(np.dot(y_tmp_, y_tmp_))

    init_num = int(PyGP.SEMANTIC_NUM / 10)
    min_vec = (2e20, 2e20)
    vec_tmp = []
    vec_tmp_ = []
    for i in range(init_num):

        if init_num >= len(dis_sorted):
            break
        cdd_0 = candidate_[dis_sorted[i]]
        tg = i + 1
        while tg - (i) < PyGP.SEMANTIC_NUM / init_num and tg < len(dis_sorted):
            cdd_1 = candidate_[dis_sorted[tg]]
            k_ = least_square_method(tsematic, candidate[dis_sorted[i]] * b_list[dis_sorted[i]],
                                     candidate[dis_sorted[tg]], tgdrvt)

            cdd = (1 - k_) * cdd_0 * b_list[dis_sorted[i]] + k_ * cdd_1

            rsdls_0 = np.subtract(tsematic_, cdd)
            E_0 = np.sqrt(np.dot(rsdls_0 - np.mean(rsdls_0),
                                 rsdls_0 - np.mean(rsdls_0)) / len(rsdls_0))

            vec_ = np.subtract(cdd, tsematic_)
            vec = np.sqrt(np.dot(vec_ * tgdrvt, vec_))

            if True:
                rsdls_mean, count = 0, 0
                rsdls_quad = []  # [tsematic_[i] / cdd[i] if cdd[i] != 0 for i in range(len(tsematic_)) elif cdd[i] == 0 and tsematic_[i] != 0: 100000000 else: np.NAN]
                rsdls_quad = np.divide(tsematic_, cdd)
                # for z in range(len(tsematic_)):
                #     if cdd[z] != 0:
                #         rsdls_quad.append(tsematic_[z] / cdd[z])
                #     elif  cdd[z] == 0 and tsematic_[z] != 0:
                #         rsdls_quad.append(100000000)
                #     else:
                #         rsdls_quad.append(np.NAN)

                # for z in range(len(rsdls_quad)):
                #     if not np.isnan(rsdls_quad[z]):
                #         rsdls_mean += rsdls_quad[z]
                #         count += 1
                # rsdls_mean /= count
                # for z in range(len(rsdls_quad)):
                #     if np.isnan(rsdls_quad[z]):
                #         rsdls_quad[z] = rsdls_mean
                # rsdls_quad = np.array(rsdls_quad)
                E_quad = np.sqrt(np.dot(rsdls_quad - rsdls_mean,
                                        rsdls_quad - rsdls_mean) / len(rsdls_quad))

                k = [(1 - k_) * b_list[dis_sorted[i]], k_]
                vec_tmp.append([E_quad, dis_sorted[i], dis_sorted[tg], k[0], k[1]])
                vec_tmp_.append(E_quad)

            tg += 1
    succeed = False

    if len(vec_tmp) > 0:
        v_sort = np.argsort(vec_tmp_)

        for id, v_id in enumerate(v_sort):
            succeed = False
            k = [vec_tmp[v_id][3], vec_tmp[v_id][4]]
            k0_tmp = [k[0]]
            k1_tmp = [k[1]]
            cdd_0_ = np.array(candidate_[int(vec_tmp[v_id][1])])
            cdd_1_ = np.array(candidate_[int(vec_tmp[v_id][2])])
            cdd = k[0] * cdd_0_ + k[1] * cdd_1_
            # print(type(tgdrvt_), type(cdd))
            cdd_mean = np.dot(cdd * tgdrvt_, cdd * tgdrvt_)
            b = np.dot(cdd * tgdrvt_, tsematic * tgdrvt_) / cdd_mean if cdd_mean != 0 else 0
            k = [k[0] * b, k[1] * b]
            cdd = k[0] * cdd_0_ + k[1] * cdd_1_
            vec = np.subtract(tsematic_, cdd)
            effects = np.sqrt(np.dot(vec * tgdrvt_, vec))
            if effects < dis_cur:
                succeed = True
            if succeed:

                candidate_min = [cdd_0_, cdd_1_]
                idx_min = [int(vec_tmp[v_id][1]), int(vec_tmp[v_id][2])]

                trs_cdd = [candidates[idx_min[0]].tree, candidates[idx_min[1]].tree]
                longer = trs_cdd[0].inner_size + trs_cdd[1].inner_size + 3 >= s3_size
                cdd_size = (trs_cdd[0].inner_size, trs_cdd[1].inner_size, s3_size, depth_limit)
                if effect_test(tsematic_, candidate_origin_,
                               cdd_0_, cdd_1_, k, tgdrvt_, cdd_size, True, mask):
                    cdd = k[0] * cdd_0_ + k[1] * cdd_1_

                    k = [k[0], k[1]]

                    break

    # 返回该两个点
    return (idx_min, candidate_min, None, k, succeed)


def indivSelect_sem_(tsematic_, candidate_, tgdrvt_, tgdrvt_origin_, candidate_origin_, origin_size, trs_size,
                     depth_limit, mask, idx=None):  # 用于语义的个体选择
    tsematic = tsematic_ * mask
    candidate = candidate_ * mask
    tgdrvt = tgdrvt_ * mask
    tgdrvt_origin = tgdrvt_origin_ * mask
    candidate_origin = candidate_origin_ * mask

    tgdrvt_r = tgdrvt_ * ((1 + mask) % 2)

    # 选一个最近点
    idx_min = [-1, -1]
    candidate_min = [candidate[0], candidate[1]]

    rsdls_cur = np.subtract(tsematic_, candidate_origin_)
    dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt, rsdls_cur))

    dis_cur_r = np.sqrt(np.dot(rsdls_cur * tgdrvt_r, rsdls_cur))

    center = np.sum(candidate, axis=0) / len(candidate)

    tgsmt_mean = np.dot(tsematic, tsematic)
    cdd_mean2_list = list(map(lambda x: np.dot(x, x), candidate))

    cdd_mean2_list = list(map(lambda x: np.dot(x, x), candidate))
    b_list = list(map(lambda x: np.dot(candidate[x], tsematic) / cdd_mean2_list[x] if cdd_mean2_list[x] != 0 else 0,
                      range(len(candidate))))

    rsdls_ = list(map(lambda x: np.subtract(tsematic, candidate[x] * b_list[x]), range(len(candidate))))

    dis_all_w = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls_))  # 加权距离

    dis_sorted = np.argsort(dis_all_w)

    tgsmt_mean_ = np.dot(tsematic_, tsematic_)
    y_tmp_ = tsematic_ - tgsmt_mean_
    y_tmp_sigma_ = np.sqrt(np.dot(y_tmp_, y_tmp_))

    init_num = 100
    min_vec = (2e20, 2e20)
    succeed = False
    k = 1
    for i in range(init_num):
        cdd_0 = candidate_[dis_sorted[i]]

        cdd = cdd_0 * b_list[dis_sorted[i]]
        vec_ = np.subtract(cdd, tsematic_)
        vec = np.sqrt(np.dot(vec_ * tgdrvt, vec_))
        vec_1 = np.sqrt(np.dot(vec_ * tgdrvt_r, vec_))

        if vec < dis_cur and vec_1 < dis_cur_r:
            succeed = True
            min_vec = (vec, vec_1)
            candidate_min = cdd_0
            idx_min = dis_sorted[i]
            k = b_list[dis_sorted[i]]

            for z in range(0):

                # mask_ = rd.choice([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], size=len(tsematic_))

                tgdrvt = tgdrvt_ * mask_
                tgdrvt_r = tgdrvt_ * ((1 + mask_) % 2)
                tsematic = tsematic_ * mask

                cdd_mean = np.dot(cdd_0 * mask_, cdd_0 * mask_)
                b = np.dot(cdd_0, tsematic) / cdd_mean if cdd_mean != 0 else 0

                vec_ = np.subtract(cdd_0 * b, tsematic_)
                vec = np.sqrt(np.dot(vec_ * tgdrvt, vec_))
                vec_1 = np.sqrt(np.dot(vec_ * tgdrvt_r, vec_))

                dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt, rsdls_cur))

                dis_cur_r = np.sqrt(np.dot(rsdls_cur * tgdrvt_r, rsdls_cur))

                if not (vec < dis_cur and vec_1 < dis_cur_r):
                    succeed = False
                    break
        if succeed:

            cdd_mean = np.dot(candidate_[dis_sorted[i]], candidate_[dis_sorted[i]])
            k = b = np.dot(cdd_0, tsematic_) / cdd_mean if cdd_mean != 0 else 0

            tgdrvt = tgdrvt_ * mask

            tgdrvt_r = tgdrvt_ * ((1 + mask) % 2)

            cdd = b * candidate_[idx_min]
            vec_ = np.subtract(cdd, tsematic_)
            vec = np.sqrt(np.dot(vec_ * tgdrvt, vec_))
            vec_1 = np.sqrt(np.dot(vec_ * tgdrvt_r, vec_))

            dis_cur = np.sqrt(np.dot(rsdls_cur * tgdrvt, rsdls_cur))

            dis_cur_r = np.sqrt(np.dot(rsdls_cur * tgdrvt_r, rsdls_cur))

            if not (vec < dis_cur and vec_1 < dis_cur_r):
                succeed = False
            else:
                break
        # [k0_, k1_, vec, vec_1] = Levenberg_Marquarelt_2(tgdrvt, tgdrvt_origin, tsematic, candidate_min[0] * b_list[idx_min[0]], candidate[idx_1], 200)
        # if vec > lsm_vals[idx_1]:
        #     k = [k0_ * b_list[idx_min[0]], k1_]
        # else:
    # 返回该两个点
    return (idx_min, candidate_min, None, k, succeed)


def Levenberg_Marquarelt_2(tgdrvt, tgdrvt_origin, tsematic, candidate_1, candidate_2, time, init_k=None, r_=None):
    count = 0
    if not init_k:
        print("!!!!!!!!!!!!!!!!!!!")
        k = least_square_method(tsematic, candidate_1, candidate_2, tgdrvt)
        k0 = 1 - k
        k1 = k
    else:
        k0 = init_k[0]
        k1 = init_k[1]

    tsematic = np.array(tsematic)
    candidate_1 = np.array(candidate_1)
    candidate_2 = np.array(candidate_2)
    (tgdrvt, tgdrvt_origin) = (np.array(tgdrvt), np.array(tgdrvt_origin))
    # if (any(np.isnan(tgdrvt)) or any(np.isinf(tgdrvt))or any(np.isnan(candidate_1)) or any(np.isnan(candidate_2)) or any(np.isinf(candidate_1)) or any(np.isinf(candidate_2))):
    #     print((any(np.isnan(tgdrvt)), any(np.isinf(tgdrvt)), any(np.isnan(candidate_1)), any(np.isnan(candidate_2)), any(np.isinf(candidate_1)), any(np.isinf(candidate_2))))
    #     print(tgdrvt)
    #     assert (0 == 1)
    cdd = k0 * candidate_1 + k1 * candidate_2

    vec = np.subtract(cdd, tsematic)

    vec_last = np.dot(vec * tgdrvt, vec)
    vec_last_1 = np.dot(vec, vec)
    vec_best = vec_last
    vec_best_1 = vec_last_1

    k0_best = float(k0)
    k1_best = float(k1)

    # x_tmp = cdd - np.sqrt(np.dot(cdd, cdd))
    # if r_:
    #     x1 = np.sqrt(np.dot(x_tmp, x_tmp))
    #     vec_last = np.dot(x_tmp, r_[0]) / (x1 * r_[1]) if (x1 != 0 and r_[1] != 0) else (1 if x1 == r_[1] else -1)
    #     vec_best = vec_best_1 = vec_last
    # else:
    #
    #     tgsmt_mean = np.dot(tsematic, tsematic)
    #     y_tmp = tsematic - tgsmt_mean
    #     y_tmp_sigma = np.sqrt(np.dot(y_tmp, y_tmp))
    #     r_ = [y_tmp, y_tmp_sigma]
    #     x1 = np.sqrt(np.dot(x_tmp, x_tmp))
    #     vec_last = np.dot(x_tmp, r_[0]) / (x1 * r_[1]) if (x1 != 0 and r_[1] != 0) else (1 if x1 == r_[1] else -1)
    #     vec_best = vec_best_1 = vec_last
    # # print(x_tmp, r_[0], vec_best, vec_last)

    assert (not (np.isnan(k0) or np.isinf(k0)))

    JX0 = tgdrvt_origin * candidate_1
    JX1 = tgdrvt_origin * candidate_2
    JX_s = np.array([JX0, JX1])
    JXTJX_s = np.dot(JX_s, np.transpose(JX_s))
    u0 = np.max(JXTJX_s) * 10 ** -3
    if u0 == 0:
        u0 = rd.uniform(0, 1)
    res_JT = np.dot(JX_s, vec * tgdrvt_origin).astype(np.float64)
    while (count < time):
        try:
            delta_ks = np.linalg.solve((JXTJX_s + u0 * np.ones(shape=(2, 2))).astype(np.float64),
                                       res_JT)  # 0.1 * np.dot(JX_s, vec * tgdrvt_origin)
        except np.linalg.LinAlgError:
            delta_ks = np.linalg.pinv((JXTJX_s + u0 * np.ones(shape=(2, 2))).astype(np.float64)) @ res_JT
        if not (np.isnan(delta_ks[0]) or np.isnan(delta_ks[1]) or np.isinf(delta_ks[0]) or np.isinf(delta_ks[1])):

            # return (float(k0_best), float(k1_best), float(np.sqrt(vec_best)))
            k0 = k0 - delta_ks[0]
            k1 = k1 - delta_ks[1]

            cdd = k0 * candidate_1 + k1 * candidate_2

            # ===========================1
            vec = np.subtract(cdd, tsematic)

            vec_now = np.dot(vec * tgdrvt, vec)

            # #===========================2
            # x_tmp = cdd - np.sqrt(np.dot(cdd, cdd))
            # x1 = np.sqrt(np.dot(x_tmp, x_tmp))
            # vec_now = np.dot(x_tmp, r_[0]) / (x1 * r_[1]) if (x1 != 0 and r_[1] != 0) else (1 if x1 == r_[1] else -1)
        else:
            vec_now = vec_last
        if vec_now > vec_last:  # ! [ ]
            u0 *= 2
            k0 = k0_best
            k1 = k1_best
        else:
            res_JT = np.dot(JX_s, vec * tgdrvt_origin).astype(np.float64)
            u0 /= 3
            if vec_now < vec_best and not (np.isinf(k0) or np.isinf(k1) or np.isnan(k0) or np.isnan(k1)):
                k0_best = k0
                k1_best = k1
                vec_best = vec_now

        vec_last = vec_now
        count += 1
    assert (not (np.isnan(k0_best) or np.isnan(k1_best) or np.isinf(k0_best) or np.isinf(k1_best)))
    # print(k0_best, k1_best, np.sqrt(vec_best), np.sqrt(vec_best_1))

    # =============================1
    # return [float(k0_best), float(k1_best), float(np.sqrt(vec_best)), float(np.sqrt(vec_best_1))]
    # =============================1
    return [float(k0_best), float(k1_best), float(vec_best), float(vec_best_1)]


def least_square_method(tsematic, candidate_1, candidate_2, tgdrvt):
    numerator = np.dot((candidate_2 - candidate_1) * tgdrvt, tsematic - candidate_1)
    denominator = np.dot((candidate_1 - candidate_2) * tgdrvt, candidate_1 - candidate_2)
    if denominator == 0.:
        return 0
    if np.isinf(numerator).any() or np.isinf(denominator) or np.isnan(denominator).any() or np.isnan(numerator):
        return 0
    if (math.isnan(numerator / denominator)):
        raise ValueError("why here..", numerator, denominator, candidate_1, candidate_2, tsematic)
    return numerator / denominator


def effect_test(tsematic, origin, candidate_1, candidate_2, k, tgdrvt, cdd_size, serious=False, mask=None):
    cdd = k[0] * candidate_1 + k[1] * candidate_2
    vec = np.subtract(tsematic, cdd)

    # tgdrvt_1 = tgdrvt * mask
    # tgdrvt_2 = tgdrvt * ((mask + 1) % 2)
    # effect_1 = np.sqrt(np.dot(vec * tgdrvt_1, vec))
    # effect_2 = np.sqrt(np.dot(vec * tgdrvt_2, vec))

    effect = np.sqrt(np.dot(vec * tgdrvt, vec) / len(vec))

    # effect_1 = np.sqrt(np.dot(vec, vec))
    # vec = np.subtract(tsematic, candidate_1) * tgdrvt
    # effect_1 = np.sqrt(np.dot(vec, vec))
    # vec = np.subtract(tsematic, candidate_2) * tgdrvt
    # effect_2 = np.sqrt(np.dot(vec, vec))

    vec = np.subtract(tsematic, origin)
    origin_effect = np.sqrt(np.dot(vec * tgdrvt, vec) / len(vec))
    # origin_effect_1 = np.sqrt(np.dot(vec * tgdrvt_1, vec))
    # origin_effect_2 = np.sqrt(np.dot(vec * tgdrvt_2, vec))

    # origin_effect_1 = np.sqrt(np.dot(vec, vec))
    # if effect_1 < effect:
    #     k = 0
    #     effect = effect_1
    # if effect_2 < effect:
    #     k = 1
    #     effect = effect_2
    # vec_1 = float(cdd_size[0] + cdd_size[1] + 3) - float(cdd_size[2])
    crm_size = cdd_size[3] - cdd_size[2]

    if serious:
        return effect != 0 and (0.9999 ** (float(cdd_size[0] + cdd_size[1] + 3 + crm_size))) / effect > (0.9999 ** (
            float(cdd_size[3]))) / origin_effect * 0.99  # , origin_effect_1, origin_effect_2, effect, effect - origin_effect, origin_effect)
    else:
        return (effect - origin_effect < origin_effect * 0.0, k, effect,
                origin_effect)  # , origin_effect_1, origin_effect_2, effect, effect - origin_effect, origin_effect)


def bounds_check(subtr: TreeNode, smt_rg, k, data_rg):  # ![ ] 不支持有多个范围的情况
    # if k is None:
    #     rg_0, rg_1 = smt_rg[0][0], smt_rg[0][1]
    # else:
    #     rg_0 = (min(smt_rg[0][0] * k[0], smt_rg[0][1] * k[0]),
    #             max(smt_rg[0][0] * k[0], smt_rg[0][1] * k[0]))
    #     rg_1 = (min(smt_rg[1][0] * k[1], smt_rg[1][1] * k[1]),
    #             max(smt_rg[1][0] * k[1], smt_rg[1][1] * k[1]))
    #     rg_0 = (rg_0[0] + rg_1[0], rg_0[1] + rg_1[1])
    rg = subtr.getRange(data_rg)
    # if (rg != rg_0):
    #     subtr.exp_draw()
    #     s1.exp_draw()
    #     s2.exp_draw()
    #     print('k', k)
    #     print('smt_rg', smt_rg)
    #     print('rg', rg, rg_0)
    #     raise ValueError
    ancestors = subtr.getAncestors()
    for x in ancestors:
        child = x[0].getChilds()
        if ((x[0].nodeval.name == '/'and child[0].print_exp_subtree() != child[1].print_exp_subtree() and x[1] == 1) or x[0].nodeval.name == 'log') and (
                rg[0] <= 0. <= rg[1] or math.fabs(rg[0]) == 0.0 or math.fabs(rg[1]) == 0.0) :
            return False
        if x[0].nodeval.name == '/' and x[1] == 0:
            child_rg_1 = child[1].getRange(data_rg)
            if (child_rg_1[0] <= 0. <= child_rg_1[1] or math.fabs(child_rg_1[0]) == 0.0 or math.fabs(
                    child_rg_1[1]) == 0.0) and child[0].print_exp_subtree() != child[
                1].print_exp_subtree():  # 修改分子导致本来分子分母相同的除法不再平衡
                return False
        trs = x[0].getChilds().copy()
        trs.pop(x[1])

        rgs = [node.getRange(data_rg) for node in trs]
        rgs.insert(x[1], rg)
        rg = PyGP.rg_compute(x[0], 0, rgs)
    return True

def roth_cdd_slts(rd, res_vals):
    tg = 0

    max_res_vals = np.max(res_vals)
    res_vals = np.array([math.e ** (max_res_vals - res_vals[i]) for i in range(len(res_vals))])
    res_vals /= np.sum(res_vals)
    tg_randval = rd.uniform(0, 1)
    res = 0
    for i in range(len(res_vals)):
        res += res_vals[i]
        if res > tg_randval:
            tg = i
            break
    return tg

def roth(rd, tgs, res_vals):
    tg = 0
    if len(tgs) > 0 and (len(res_vals) == 0):  # or rd.uniform(0, 1) < 0):
        tg = tgs[rd.integers(0, len(tgs))]
    # else:
    #     tg = res_id
    else:
        
        max_res_vals = np.max(res_vals)
        res_vals = np.array([math.e ** (res_vals[i] - max_res_vals) for i in range(len(res_vals))])
        res_vals /= np.sum(res_vals)
        # res_vals /= np.sum(res_vals)
        tg_randval = rd.uniform(0, 1)
        res = 0
        # print(res_vals)
        for i in range(len(res_vals)):
            res += res_vals[i]
            if res > tg_randval:
                tg = tgs[i]
                break
    return tg


import dill


def _crossover(rd, pprogs: [Program], smts: PopSemantic, funcs, depth_limit, fitness):  # [] python回收机制， 这些subtree_node不一定还存在
    # crsover_time = [0, 0, 0]
    progs = []
    idx = 0
    idx_suc = 0
    # print(pprogs[0].CASH_MANAGER, pprogs[0].ID_MANAGER, pprogs[0].ID_MANAGER.idAllocate())
    prog_depth_max = 0
    data_rg = smts.get_datarg()
    idx_best = 0

    num = 0
    if np.random.uniform(0, 1) < 0.9:
        for i in range(len(pprogs)):

            indiv1: Program = pprogs[i]
            child = indiv1
            num += 1
            idx += 1
            id = indiv1.prog_id
            indiv1.seman_sign = -1

            res_tgs = smts.compute_tg(id)
            tg_idx = roth(rd, *res_tgs)

            rlt_posi = smts.get_tgnode_posi(id, tg_idx)
            subtree3 = child.getSubTree(rlt_posi)
            s3_height, s3_size, s3_rlt_depth = subtree3.height(), subtree3.inner_size, subtree3.relative_depth()
            h_limit = PyGP.DEPTH_MAX_SIZE - subtree3.relative_depth() - 1  # subtree3.height()#child.depth - subtree3.relative_depth() - 2#
            h_init = 0  # if PyGP.DEPTH_MAX_SIZE - subtree3.relative_depth() - 1 < PyGP.DEPTH_MAX - depth_limit else PyGP.DEPTH_MAX - depth_limit
            h_limit = 1 if h_limit <= 0 else h_limit

            tgdrvt_origin = smts.get_drvt_d(id, tg_idx)
            smt_size, h_rg = smts.get_smt_size(h_limit, init_height=h_init)

            real_nums = PyGP.SEMANTIC_NUM if smt_size > PyGP.SEMANTIC_NUM else smt_size
            r_idxs = rd.choice(range(smt_size), real_nums, replace=False)

            tgsmt = smts.get_tgsmt_d(id, tg_idx)
            # assert (len(r_idxs) > 0)
            if len(r_idxs) > 1 :
                t0 = time.time()

                # rlt_posi = smts.get_tgnode_posi(id, tg_idx)
                tr_origin = subtree3  # child.getSubTree(rlt_posi)
                cdd_origin = smts.get_snode_tgsmt(id, tg_idx)

                tgdrvt = np.fabs(tgdrvt_origin)  # np.ones(len(tgsmt))#
                tgdrvt_test = PyGP.abs_normalize(tgdrvt)  # np.ones(len(tgsmt))#np.fabs(tgdrvt_origin)#
                # tgdrvt_weight = tgdrvt_origin
                tgdrvt_f_idx = None  # PyGP.cluster(tgdrvt)[0]#过滤后的绝对偏导值
                # if PyGP.DEPTH_MAX_SIZE - (s3_rlt_depth + s3_height) >= 1 and s3_height > 1:
                #     candidate.insert(0, cdd_origin)
                #     smt_rg.insert(0, tr_origin.getRange(data_rg))
                #     trs_size.insert(0, s3_size)

                # depth_new = trs_cdd[0].height() if trs_cdd[0].height() > trs_cdd[1].height() else trs_cdd[
                #     1].height()
                depth_new = depth_limit  # s3_rlt_depth + depth_new + 2 if s3_rlt_depth + depth_new + 2 > child.depth else child.depth
                mask = np.ones(len(tgsmt))  # rd.choice([1], size=len(tgsmt))
                # mask_tmp = np.argsort(tgsmt)
                # mask = np.zeros(len(tgsmt))
                # for i in range(0, len(tgsmt), 4):
                #     mask[mask_tmp[i]] = 1
                candidates = smts.get_smt_trs(h_rg, r_idxs)
                cdd_c = np.ones(len(tgsmt))
                cdd_c0 = np.zeros(len(tgsmt))
                # b_c = np.dot(cdd_c, tgsmt) / np.dot(cdd_c, cdd_c)
                # candidates.append(TR(expr='1', smt= cdd_c, isize=0, rg=[1., 1.], tr=TreeNode(1.)))
                candidates.append(TR(expr='1', smt= cdd_c, isize=0, rg=[1., 1.], tr=TreeNode(1.)))
                # candidates.append(TR(expr='0', smt= cdd_c0, isize=0, rg=[0., 0.], tr=TreeNode(0.)))
                # candidates.append(TR(expr=str(subtree3), smt=cdd_origin, isize=0, rg=subtree3.getRange(data_rg), tr=subtree3.copy()))
                # candidates.append(TR(expr='1', smt= cdd_c, isize=0, rg=[1., 1.], tr=TreeNode(1.)))
                # candidates.append(TR(expr=str(b_c), smt= cdd_c * b_c, isize=0, rg=[b_c, b_c], tr=TreeNode(b_c)))
                if PyGP.CRO_STG == 0:
                    (indiv_idx, indivs, _, k, succeed) = indivSelect_sem_3(tsematic_=tgsmt, candidates=candidates,
                                                                        tgdrvt_=tgdrvt_test,
                                                                        tgdrvt_origin_=tgdrvt_origin,
                                                                        candidate_origin_=cdd_origin,
                                                                        mask=mask, depth_limit=depth_new,
                                                                        s3_size=(s3_size, child.root.inner_size),
                                                                        tr_origin=tr_origin, org=PyGP.DEPTH_MAX_SIZE - (
                                    s3_rlt_depth + s3_height) >= 1 and s3_height > 1)
                elif PyGP.CRO_STG == 1:
                    (indiv_idx, indivs, _, k, succeed) = indivSelect_sem_4(tsematic_=tgsmt, candidates=candidates,
                                                                        tgdrvt_=tgdrvt_test,
                                                                        tgdrvt_origin_=tgdrvt_origin,
                                                                        candidate_origin_=cdd_origin,
                                                                        mask=mask, depth_limit=depth_new,
                                                                        s3_size=(s3_size, child.root.inner_size),
                                                                        tr_origin=tr_origin, org=PyGP.DEPTH_MAX_SIZE - (
                                    s3_rlt_depth + s3_height) >= 1 and s3_height > 1)

                if isinstance(indiv_idx, list):

                    # if PyGP.DEPTH_MAX_SIZE - (s3_rlt_depth + s3_height) >= 1 and s3_height > 1:
                    #     trs_cdd = [smts.get_tr(trs[indiv_idx[0] - 1]) if indiv_idx[0] > 0 else tr_origin,
                    #                smts.get_tr(trs[indiv_idx[1] - 1]) if indiv_idx[1] > 0 else tr_origin]
                    # else:
                    trs_cdd = [candidates[indiv_idx[0]].tree, candidates[indiv_idx[1]].tree]
                    # trs_cdd = [smts.get_tr(trs[indiv_idx[0]]), smts.get_tr(trs[indiv_idx[1]])]
                    if succeed:
                        indivs = [candidates[indiv_idx[0]].semantic, candidates[indiv_idx[1]].semantic]
                        # k = Levenberg_Marquarelt_2(tgdrvt_test, tgdrvt_origin, tgsmt, indivs[0], indivs[1], 50, k)
                        # effect_better=[True]
                        # longer = trs_cdd[0].inner_size + trs_cdd[1].inner_size + 3 >= s3_size
                        # cdd_size = (trs_cdd[0].inner_size, trs_cdd[1].inner_size, s3_size, depth_new)
                        effect_better = [True]  # effect_test(tgsmt, cdd_origin,
                        # indivs[0], indivs[1], k, tgdrvt, cdd_size, True, mask)
                    else:
                        # k = Levenberg_Marquarelt_2(tgdrvt_test, tgdrvt_origin, tgsmt, indivs[0], indivs[1], 0, k)
                        effect_better = [False]

                    # # print('===========+=============', k, k[0], b)
                    # # if PyGP.DEPTH_MAX_SIZE - (s3_rlt_depth + s3_height) >= 1 and s3_height > 1:
                    # #     trs_cdd = [smts.get_tr(trs[indiv_idx[0] - 1]) if indiv_idx[0] > 0 else tr_origin,
                    # #                smts.get_tr(trs[indiv_idx[1] - 1]) if indiv_idx[1] > 0 else tr_origin]
                    # # else:
                    # trs_cdd = [smts.get_tr(trs[indiv_idx[0]]), smts.get_tr(trs[indiv_idx[1]])]
                    # # trs_cdd = [smts.get_tr(trs[indiv_idx[0]]), smts.get_tr(trs[indiv_idx[1]])]
                    # longer = trs_cdd[0].size + trs_cdd[1].size + 3 >= s3_size
                    # cdd_size = (trs_cdd[0].size, trs_cdd[1].size, s3_size, depth_new)
                    # effect_better = effect_test(tgsmt, cdd_origin,
                    #                             indivs[0], indivs[1], k, tgdrvt_test, cdd_size, longer)
                else:

                    # if PyGP.DEPTH_MAX_SIZE - (s3_rlt_depth + s3_height) >= 1 and s3_height > 1:
                    #     trs_cdd = smts.get_tr(trs[indiv_idx - 1]) if indiv_idx > 0 else tr_origin
                    # else:
                    trs_cdd = candidates[indiv_idx].tree
                    effect_better = [True] if succeed else [False]
                    
                if effect_better[0]:
                    if effect_better[0]:
                        idx_suc += 1
                    if not isinstance(trs_cdd, list):
                        subtree1: TreeNode = trs_cdd
                        if not (math.fabs(k - 1) == 0.0):
                            tr1 = TreeNode(funcs.funcSelect_n('mul'))
                            tr1.setChilds([subtree1, TreeNode(k, parent=(tr1, 1))])
                            subtree1.setParent((tr1, 0))
                            tr3 = tr1
                        else:
                            tr3 = subtree1
                    else:
                        subtree1: TreeNode = trs_cdd[0]
                        subtree2: TreeNode = trs_cdd[1]

                        # if (not (not effect_better[0] or (indiv_idx[0] != 0 and indiv_idx[1] != 0)) and a):
                        #     print(subtree3.height(), trs_cdd[0].height(),
                        #           trs_cdd[1].height(), subtree1.height(),
                        #           subtree2.height())
                        # assert(indiv_idx[0] != indiv_idx[1])
                        if (math.fabs(k[1]) == 0.0):
                            if subtree1.dtype == 'Const':
                                tr1 = TreeNode(subtree1.nodeval * k[0])
                            else:
                                tr1 = TreeNode(funcs.funcSelect_n('mul'))
                                tr1.setChilds([subtree1, TreeNode(k[0], parent=(tr1, 1))])
                                subtree1.setParent((tr1, 0))
                            tr3 = tr1
                        elif (math.fabs(k[0]) == 0.0):
                            if subtree2.dtype == 'Const':
                                tr2 = TreeNode(subtree2.nodeval * k[1])
                            else:
                                tr2 = TreeNode(funcs.funcSelect_n('mul'))
                                tr2.setChilds([subtree2, TreeNode(k[1], parent=(tr2, 1))])
                                subtree2.setParent((tr2, 0))
                            tr3 = tr2
                        else:
                            if subtree1.dtype == 'Const':
                                tr1 = TreeNode(subtree1.nodeval * k[0])
                            else:
                                tr1 = TreeNode(funcs.funcSelect_n('mul'))
                                tr1.setChilds([subtree1, TreeNode(k[0], parent=(tr1, 1))])
                                subtree1.setParent((tr1, 0))
                            if subtree2.dtype == 'Const':
                                tr2 = TreeNode(subtree2.nodeval * k[1])
                            else:
                                tr2 = TreeNode(funcs.funcSelect_n('mul'))
                                tr2.setChilds([subtree2, TreeNode(k[1], parent=(tr2, 1))])
                                subtree2.setParent((tr2, 0))
                            if tr1.dtype == 'Const' and tr2.dtype == 'Const':
                                tr3 = TreeNode(tr1.nodeval + tr2.nodeval)
                            else:
                                tr3 = TreeNode(funcs.funcSelect_n('add'))
                                tr3.setChilds([tr1, tr2])
                                tr1.setParent((tr3, 0))
                                tr2.setParent((tr3, 1))

                    if subtree3.parent is not None:
                        tr3.setParent(subtree3.parent)
                    else:
                        child.root = tr3

                    if isinstance(trs_cdd, list):
                        rg0, rg1 = candidates[indiv_idx[0]].range, candidates[indiv_idx[1]].range
                    else:
                        rg0 = candidates[indiv_idx].range

                    if PyGP.INTERVAL_COMPUTE and (
                            (isinstance(trs_cdd, list) and not bounds_check(tr3, (rg0, rg1), k, data_rg))
                            or (not isinstance(trs_cdd, list) and not bounds_check(tr3, rg0, None, data_rg))):
                        if tr3.parent is not None:
                            subtree3.setParent(tr3.parent)
                        else:
                            child.root = subtree3

                    child.sizeUpdate()
                    if child.length > 128:
                        if isinstance(indiv_idx, list):
                            print("S1:", trs_cdd[0].inner_size + trs_cdd[1].inner_size + 3 - s3_size + child.root.inner_size)
                        else:
                            print("S2:", trs_cdd.inner_size - s3_size + child.root.inner_size)
                        raise ValueError("cur_length: {S1}".format(S1=child.length))
                    progs.append(child)
                    if prog_depth_max < child.depth:
                        prog_depth_max = child.depth
                else:
                    progs.append(None)
                # crsover_time[2] += t1 - t0
                t1 = time.time()
            else:
                progs.append(None)
        num += 1
    else:
        for i in range(len(pprogs)):
            indiv1: Program = pprogs[i]
            child = indiv1
            num += 1
            idx += 1
            id = indiv1.prog_id
            indiv1.seman_sign = -1

            res_tgs = smts.compute_tg(id)
            tgdrvt_origins = [smts.get_drvt_d(id, tg) for tg in res_tgs[0]]
            tgsmts = [smts.get_tgsmt_d(id, tg) for tg in res_tgs[0]]

            subtrees = [child.getSubTree(smts.get_tgnode_posi(id, tg)) for tg in res_tgs[0]]
            subtree_innersize = [subtree.inner_size for subtree in subtrees]
            root_size = child.root.inner_size

            smt_size, h_rg = smts.get_smt_size(5, init_height=1)

            real_nums = int(20) if smt_size > int(20) else smt_size
            r_idxs = rd.choice(range(smt_size), real_nums, replace=False)

            candidates = smts.get_smt_trs(h_rg, r_idxs)
            # tgdrvt_norms = [PyGP.abs_normalize(np.fabs(tgdrvt_origins[tg])) for tg in range(len(subtrees))]
            # tree_cdd = random.randint(0, len(candidates) - 1).tree
            dis_cur, tgs, cdds = [], [], []
            for cdd in candidates:
                rsdls_cur = np.subtract(fitness, cdd.semantic)
                res = np.sqrt(np.dot(rsdls_cur, rsdls_cur))
                if not (np.isnan(res) or np.isinf(res)):
                    dis_cur.append(res)
                    cdds.append(cdd)
            # for cdd in candidates:
            #     for tg in range(len(subtrees)):
            #         rsdls_cur = np.subtract(tgsmts[tg], cdd.semantic) * tgdrvt_norms[tg]
            #         res = np.sqrt(np.dot(rsdls_cur, rsdls_cur))
            #         if (root_size - subtree_innersize[tg] + cdd.tree.inner_size < 64 and not (np.isnan(res) or np.isinf(res))):
                        
            #             dis_cur.append(res)
            #             tgs.append(tg)
            #             cdds.append(cdd)
            if len(dis_cur) > 0:
                arg_list = np.argsort(dis_cur)
                dis_cur = [dis_cur[argidx] for argidx in arg_list]
                cdds = [cdds[argidx] for argidx in arg_list]
                tg = roth_cdd_slts(rd, dis_cur)
                tree_cdd = cdds[tg].tree
                child.root = tree_cdd
                child.sizeUpdate()
                if child.length > PyGP.SIZE_LIMIT:
                    raise ValueError("rm, cur_length: {S1}".format(S1=child.length))
                progs.append(child)
                if prog_depth_max < child.depth:
                    prog_depth_max = child.depth
            # if len(dis_cur) > 0:
            #     cdd_list = np.argsort(tgs)[-10:]
            #     dis_cur = [dis_cur[cdd] for cdd in cdd_list]
            #     tgs = [tgs[cdd] for cdd in cdd_list]
            #     cdds = [cdds[cdd] for cdd in cdd_list]
            #     tg = roth_cdd_slts(rd, dis_cur)
            #     tree_cdd = cdds[tg].tree
                    
            #     if subtrees[tgs[tg]].parent is not None:
            #         tree_cdd.setParent(subtrees[tgs[tg]].parent)
            #     else:
            #         child.root = tree_cdd

            #     child.sizeUpdate()
            #     progs.append(child)
            #     if prog_depth_max < child.depth:
            #         prog_depth_max = child.depth
            else:
                progs.append(None)
    # print('crossover: ', idx, idx_suc, idx_best, 'slt_1_time ', crsover_time[0], 'slt_2_time ', crsover_time[1], 'crotime', crsover_time[2])
    return progs


from .base import BaseCrossover


class SMT_Weight_Crossover_LV2(BaseCrossover):
    def __init__(self, pop_size):
        self.pop_size = pop_size

    def run(self, pprogs, funcs, depth_limit, fitness):
        smts = self.semantics
        rd = self.rg
        return _crossover(rd, pprogs, smts, funcs, depth_limit, fitness)