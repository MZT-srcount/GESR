from .tree_basic import TreeNode, Func

def treeid_update(root: TreeNode, output, const_val, id_allocator, *args):
    stack = [(root, 1)]
    output_sub = {}
    expunits = [[], []]
    counter = 0
    while len(stack) > 0:
        tn = stack.pop()
        if tn[0].dtype == "Func" and tn[1] == 1:
            tn_childs = tn[0].getChilds()
            for i in range(tn[0].getArity()):
                stack.append((tn_childs[i], 0))
            stack.append((tn[0], 2))

        elif tn[1] == 2 and isinstance(tn[0], Func):
            funcs_num = 0
            expunit = [tn[0].nodeval.id, tn[0].getArity()]
            tn_childs = tn[0].getChilds()
            for i in range(tn[0].getArity()):
                if tn_childs[i].dtype != "Int":
                    expunit.append(output_sub[tn_childs[i].node_id])
                else:
                    if args[(tn_childs[i].nodeval)].getCashState() == 1:
                        expunit.append(args[(tn_childs[i].nodeval)].getCashId())
                    else:
                        expunit.append(output[args[tn_childs[i].nodeval]])
            inherit = False

            for i in range(tn[0].getArity()):
                if tn_childs[i].getCashState() == 0 and tn_childs[i].dtype == "Int":
                    if args[tn_childs[i].nodeval].dtype == "Func":
                        expunit.append(output[args[tn_childs[i].nodeval].nodeval.id])
                        inherit = True
                        funcs_num += 1
                        break;
                elif tn_childs[i].dtype == "Func":
                    expunit.append(output_sub[tn_childs[i].node_id])
                    funcs_num += 1
                    inherit = True
                    break;

            if not inherit:
                expunit.append(id_allocator[0])
                id_allocator[0] += 1
            output_sub[tn[0].node_id] = expunit[len(expunit) - 1]

            if counter == 0:# store the output of the subtree root
                output[tn[0].node_id] = expunit(len(expunit) - 1)

            if funcs_num == tn[0].getArity():
                expunits[1].extend(expunit)
            else:
                expunits[0].extend(expunit)

        else:#const、input
            if tn[0].dtype == "Const":
                const_val.append(tn[0].nodeval)
                output_sub[tn[0].node_id] = (-(len(const_val) - 1))
            if tn[0].dtype == "Input":
                
                output_sub[tn[0].node_id] = tn[0].nodeval
        counter += 1

    return expunits

def tr_copy_nc(tr_self):
    tnode_root = TreeNode(tr_self.nodeval)
    tnode_root.dtype_update()
    tr_self.visited = 0

    stack = [tnode_root]
    tr_stack = [tr_self]
    while stack:
        tr1 = stack.pop()  
        tr2 = tr_stack.pop()  
        tr1_childs = []
        if tr2.getCashState() > 0:
            tr1.changeCashState(0)

        if tr2.dtype == "Func":
            tr2_childs = tr2.getChilds()
            for i in range(tr2.getArity()):
                tr1_childs.append(
                    TreeNode(tr2_childs[i].nodeval, parent=(tr1, i),
                             visited=0))

            tr1.setChilds(tr1_childs)

            stack.extend(tr1_childs)
            tr_stack.extend(tr2_childs)
    return tnode_root