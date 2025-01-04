import PyGP
from .utils import treeid_update
from PyGP import SemanticPerIndiv, BPInfos

def _input_collect(childs, cash_open, output, cnode):
    def set_input(x):
        input_id = x.getCashId() if x.getCashState() == 1 and cash_open else output[x.node_id]
        if x.getCashState() == 2 and cash_open:
            x.setCashState(1)
        return input_id

    return list(map(lambda x: set_input(x), childs))


def _output_collect(prog, cnode, cash_open, output, s_signs):
    if cnode == prog.root:
        return prog.n_terms + 1 + prog.prog_id
    elif cnode.getCashState() == 2 and cash_open:
        return cnode.getCashId()
        # cnode.setCashState(1)
    childs = cnode.getChilds()
    s_inher = True
    if PyGP.SEMANTIC_SIGN:
        s_find = list(filter(lambda child: child.semantic_sign == 0 or child.semantic_save == 0 or s_signs.get(child.node_id) is not None, childs))
        s_inher = False if len(s_find) > 0 else True

    if s_inher:
        for i in range(len(childs)):
            if (childs[i].getCashState() == 0 or not cash_open) and childs[
                i].dtype == "Func":  # ((childs[i].semantic_sign < 0 and childs[i].semantic_save < 0) or not semantic):#很麻烦且浪费性能，语义情况下直接不复用了，不过在树规模很大时内存是否会不足
                return output[childs[i].node_id]
    return -1

def _semantic_collect(childs, cnode, s_infos: SemanticPerIndiv, bpinfos, s_signs, expunit, cash_open, output, cur_node_type:str):

    # 回收节点的semantic_save标识
    if cur_node_type == "Func":
        for i in range(len(childs)):
            if childs[i].semantic_save == 0:
                childs[i].semantic_save = -1
        if cnode.semantic_save == 0:  # 该节点语义需要保存
            assert(expunit[len(expunit) - 1] >= 0)
            s_infos.add_snode(expunit[len(expunit) - 1], cnode.print_exp_subtree(noparent=True))

        s_find = {}#记录子节点的语义id

        for i in range(len(childs)):

            def s_add(key, value):
                bpinfos.add_bfuncs((expunit.copy(), i), key - 1)
                if s_signs.get(cnode.node_id) is None:
                    s_signs[cnode.node_id] = {key: '1'}
                else:
                    s_signs[cnode.node_id][key] = '1'
                s_find[key] = childs[i].node_id

            if childs[i].semantic_sign == 0:  # 为0，遇到新的语义标识
                s_infos.upper()
                s_add(s_infos.count, None)

                # s_infos.set_bf_node(childs[i].zip(), childs[i].rlt_posi(), s_infos.count - 1)
                s_infos.set_bf_node(childs[i].print_exp_subtree(noparent=True), childs[i].rlt_posi(), s_infos.count - 1)
                childs[i].semantic_sign = -1

            if s_signs.get(childs[i].node_id) is not None:  # 非新语义
                list(map(lambda x: s_add(x[0], x[1]), list(s_signs[childs[i].node_id].items())))

        current_opera = expunit[0]

        for key, value in s_find.items():  # 一个个体上的每条语义的收集
            operation = [current_opera]
            locate = -1
            for i in range((len(childs))):  # 语义数据节点收集
                if childs[i].node_id == value:
                    locate = i
                if childs[i].node_id != value or len(childs) == 1:
                    operation.append(childs[i].getCashId() if childs[i].getCashState() == 1 and cash_open and (not PyGP.utils.semanticSearch(childs[i])) else output[childs[i].node_id])#[ ] 有可能缓存由于语义收集被屏蔽
            assert (locate != -1)  # can not find the semantic location
            operation.append(locate)
            bpinfos.add_ffuncs(operation, key - 1)  # 第key - 1个语义上的函数收集

    elif cnode.semantic_save == 0:
        s_infos.add_snode(output[cnode.node_id], cnode.print_exp_subtree(noparent=True))

    if cnode.semantic_sign == 0 and cnode.is_root:#根节点，不需要反向传播，额外标记处理
        s_infos.upper()
        s_infos.set_bf_node(cnode.print_exp_subtree(noparent=True), cnode.rlt_posi(), s_infos.count - 1)
        cnode.semantic_sign=-1
        bpinfos.add_bfuncs(([0, 2, 0, -1, 0], 0), s_infos.count - 1)
        bpinfos.add_ffuncs([0, 0, -1, 0], s_infos.count - 1)  # 第key - 1个语义上的函数收集

from .tree_basic import Program
def data_collects(prog:Program, id_allocator, const_val, cash_open=True, semantic=True):
    stack = [[prog.root, 1]]
    smt_signs = {}#语义标识，用于backpropagation沿途节点标识
    semantic_num = 0
    output = {}
    expunit_collects = [[], []]
    smt_clts = SemanticPerIndiv()
    bpinfos = BPInfos()
    const_val.append(0.0)
    while stack:
        cnode = stack.pop()

        # 第一次访问且非叶子节点；如果该节点已经被缓存，则不再会访问子节点
        if cnode[1] == 1 and cnode[0].dtype == "Func":
            if cnode[0].getCashState() == 1 and cash_open and \
                    (not semantic or not PyGP.semanticSearch(cnode[0])):  # 避免需要获取的语义被缓存截断
                assert (cnode[0].parent is not None)
                continue

            childs = cnode[0].getChilds()
            stack.append([cnode[0], cnode[1] + 1])
            stack.extend(list(map(lambda x: [x, 1], childs)))

        # 第二次访问/叶子节点
        elif cnode[0].dtype == "Func":
            # assert (not (cnode[0] == prog.root and prog.root.dtype != "Func"))  # root can not be an input or const value

            # 自定义节点
            if cnode[0].nodeval.id == -1:  # [] 处理自定义节点？有点忘了
                expunits = treeid_update(cnode[0].nodeval.root,
                                        output, const_val, id_allocator,
                                        cnode[0].getChilds())
                expunit_collects[1].extend(expunits[1])
                expunit_collects[0].extend(expunits[0])
                # [!] 这里直接continue?

            # 输入
            expunit = [cnode[0].nodeval.id]
            childs = cnode[0].getChilds()
            expunit.append(len(childs))  # input size
            expunit.extend(_input_collect(childs, cash_open, output, cnode[0]))
            funcs_num = sum(1 for child in childs if child.dtype == "Func")

            # 输出
            outposi = _output_collect(prog, cnode[0], cash_open, output, smt_signs)
            if outposi >= 0:
                expunit.append(outposi)
            else:
                expunit.append(id_allocator[0])
                id_allocator[0] += 1

            assert (expunit[len(expunit) - 1] >= 0)
            output[cnode[0].node_id] = expunit[len(expunit) - 1]

            # [] 语义收集, 是否可以改进
            if PyGP.SEMANTIC_SIGN and semantic:  # [] 如果删除反向传播，则需要注意语义数据不能被覆盖，可能需要修改输出节点位置覆盖策略
                _semantic_collect(childs, cnode[0], smt_clts, bpinfos, smt_signs, expunit, cash_open, output, "Func")

            # 所有节点是否均为叶子节点，提高L1、L2缓存利用率
            if funcs_num > 0:
                expunit_collects[1].extend(expunit)
            else:
                expunit_collects[0].extend(expunit)

        else:
            if cnode[0].dtype == "Input":
                assert(cnode[0].nodeval >= 0)
                output[cnode[0].node_id] = cnode[0].nodeval
            elif cnode[0].dtype == "Const":
                const_val.append(cnode[0].nodeval)
                output[cnode[0].node_id] = - (len(const_val))  # 不可为len(const_val) - 1，因为为0时无法区分

            if PyGP.SEMANTIC_SIGN and semantic and cnode[0].semantic_save == 0:# 该节点语义需要保存
                _semantic_collect(None, cnode[0], smt_clts, bpinfos, smt_signs, None, cash_open, output, "Terminal")
    exp_reunion = expunit_collects[0] + expunit_collects[1] + [-1]
    assert (len(bpinfos.semantic) == smt_clts.count)
    if PyGP.SEMANTIC_SIGN and semantic:
        return (exp_reunion, smt_clts, bpinfos)
    else:
        return (exp_reunion, smt_clts)
    # print(semprogs_collects, locates_collects)

class DataCollects:
    def __call__(self, prog, id_allocator, const_val, cash_open=True, semantic=True):
        return data_collects(prog, id_allocator, const_val, cash_open, semantic)