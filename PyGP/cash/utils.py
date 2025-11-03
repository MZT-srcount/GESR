
import PyGP
def cashUpdate(prog, c_mngr, id_clts, posi):  
    dfs = [prog.root]
    record = 0
    while dfs:
        cur_node = dfs.pop()
        if PyGP.CASH_OPEN and cur_node.visited == 0:  
            ancestor = cur_node
            while ancestor.parent is not None:  # and ancestor.parent[0].visited == 1:
                ancestor = ancestor.parent[0]
                if ancestor.getCashState() == 1:
                    record += 1
                    c_mngr.releaseNode(ancestor)
                ancestor.visited = 0  
        
        if cur_node.dtype == "Func":
            dfs.extend(cur_node.childs)
        # if id_clts.get(cur_node.node_id) is not None:
        #     print("wrong!!!!! node_id: ", cur_node.node_id)
        #     assert (0 == 1)
        id_clts[cur_node.node_id] = (cur_node, posi)
    return record


def cashGenerate(prog, c_mngr, c_clt, cash_perprog):
    bfs = [prog.root]
    cash_remain = cash_perprog
    record = 0
    while bfs:
        cur_node = bfs.pop(0)
        if cur_node.visited == 0:  
            cur_node.visited = 1
        if cur_node.parent is not None:
            ancestor = cur_node.parent[0]
            while ancestor.parent is not None and ancestor.getCashState() == 0:
                ancestor = ancestor.parent[0]
        else:
            ancestor = cur_node

        prob = (float(cur_node.getChildSize()) / prog.length) * (
                float(ancestor.getChildSize() - cur_node.getChildSize()) / prog.length)
        if cur_node.getCashState() >= 1:  
            c_clt.append(cur_node)
        elif prob >= 100 and cash_remain > 0 and c_mngr.isAvailable() and cur_node.dtype == "Func":
            record += 1
            cash_remain -= 1
            c_mngr.addCash(cur_node)
            c_clt.append(cur_node)

        if cur_node.dtype == "Func":
            for i in range(len(cur_node.getChilds())):
                bfs.append(cur_node.childs[i])
    return record
    # print('cash_remain: ', cash_remain)

def cashClear_prog(prog):
    bfs = [prog.root]
    while bfs:
        cur_node = bfs.pop(0)
        cur_node.visited = 0
        cur_node.changeCashState(0)
        if cur_node.dtype == "Func":
            for i in range(len(cur_node.getChilds())):
                bfs.append(cur_node.childs[i])

def cashClear_tr(trs):
    bfs = [trs]
    while bfs:
        cur_node = bfs.pop(0)
        cur_node.visited = 0
        cur_node.changeCashState(0)
        if cur_node.dtype == "Func":
            for i in range(len(cur_node.getChilds())):
                bfs.append(cur_node.childs[i])

def tr_copy(tr_self, c_mngr):
    from PyGP import TreeNode
    tnode_root = TreeNode(tr_self.nodeval)
    tnode_root.cash = tr_self.cash.copy()
    tnode_root.dtype_update()
    tr_self.visited = 0

    stack = [tnode_root]
    tr_stack = [tr_self]
    count = 0
    while stack:
        tr1 = stack.pop()  
        tr2 = tr_stack.pop()  
        tr1_childs = []
        if len(tr_stack) == 1:
            count += 1
        if tr2.getCashState() == 1:  
            c_mngr.countUp(tr1, tr2)
        elif tr2.getCashState() == 2:
            tr1.changeCashState(0)

        if tr2.dtype == "Func":
            tr2_childs = tr2.getChilds()
            for i in range(tr2.getArity()):
                tr1_childs.append(
                    TreeNode(tr2_childs[i].nodeval, parent=(tr1, i), cash=tr2_childs[i].cash,
                             visited=tr2_childs[i].visited))

            tr1.setChilds(tr1_childs)

            stack.extend(tr1_childs)
            tr_stack.extend(tr2_childs)
    return tnode_root

