import PyGP
from PyGP import SemanticPerIndiv, BPInfos, Program, treeid_update

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
                i].dtype == "Func":  
                return output[childs[i].node_id]
    return -1


def data_collects(prog_id, prog, id_altr, cvals):
    stack = [[prog.root, 1]]
    output = {}
    expunit_collects = [[], []]
    while stack:
        cnode = stack.pop()

        
        if cnode[1] == 1 and cnode[0].dtype == "Func":
            childs = cnode[0].getChilds()
            stack.append([cnode[0], cnode[1] + 1])
            stack.extend(list(map(lambda x: [x, 1], childs)))

        
        elif cnode[0].dtype == "Func":
            # assert (not (cnode[0] == prog.root and prog.root.dtype != "Func"))  # root can not be an input or const value

            
            if cnode[0].nodeval.id == -1:  
                expunits = treeid_update(cnode[0].nodeval.root,
                                         output, cvals, id_altr,
                                         cnode[0].getChilds())
                expunit_collects[1].extend(expunits[1])
                expunit_collects[0].extend(expunits[0])
                

            
            expunit = [cnode[0].nodeval.id]
            childs = cnode[0].getChilds()
            expunit.append(len(childs))  # input size
            expunit.extend(_input_collect(childs, False, output, cnode[0]))
            funcs_num = sum(1 for child in childs if child.dtype == "Func")

            
            if cnode[0] == prog.root:
                expunit.append(prog.n_terms + 1 + prog_id)
            else:
                expunit.append(id_altr[0])
            id_altr[0] += 1

            assert (expunit[len(expunit) - 1] >= 0)
            output[cnode[0].node_id] = expunit[len(expunit) - 1]

            
            if funcs_num > 0:
                expunit_collects[1].extend(expunit)
            else:
                expunit_collects[0].extend(expunit)

        else:
            if cnode[0].dtype == "Input":
                assert (cnode[0].nodeval >= 0)
                output[cnode[0].node_id] = cnode[0].nodeval
            elif cnode[0].dtype == "Const":
                cvals.append(cnode[0].nodeval)
                output[cnode[0].node_id] = - (len(cvals))  

    exp_reunion = expunit_collects[0] + expunit_collects[1] + [-1]
    return (exp_reunion, output)

def encode(progs: Program):
    n_terms = progs[0].n_terms
    prog_size = len(progs)
    id_altr = [n_terms + 1 + prog_size]
    cvals = []
    outputs = []
    e_clts = []
    e_iposi = [0]
    for i in range(prog_size):
        (e_clts_, output_) = data_collects(i, progs[i], id_altr, cvals)
        e_clts.extend(e_clts_)
        outputs.append(output_)
        e_iposi.append(len(e_clts))
    return (e_clts, e_iposi, id_altr, cvals, outputs)


class Encoder:
    def __call__(self, progs):
        return encode(progs)