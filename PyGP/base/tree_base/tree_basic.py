from PyGP.base.base import Base
from PyGP import Func
import math
import PyGP
import array


class TreeNode(Base):  # [] copy必须init一个treenode才能更新visited从而更新祖上节点的缓存；如果Crossover采用迁移而非复制的方式，如何处理祖上节点缓存的更新
    def __init__(
            self,
            nodeval,
            parent=None,
            cash=None,
            visited=0,
            node_id=None,
    ):
        if node_id is None:
            self.node_id = self.ID_MANAGER.idAllocate()
        else:
            self.node_id = node_id
        self.nodeval = nodeval  # 该节点的值，Func则对应Func类；特征则对应int;常量则对应float
        if parent is None:
            self.parent = None
        else:
            self.parent = parent
        if cash is None:
            self.cash = [0, -1]  # cash_state表明节点cash状态，cash_id用于GPU位置
        else:
            self.cash = cash.copy()  # cash状态

        # else:
        self.child_size = -1
        if visited:
            self.visited = 1
        else:
            self.visited = 0
        self.dtype_update()
        self.semantic_sign = -1
        self.semantic_save = -1

    @property
    def is_root(self):
        if self.parent is None:
            return True
        return False

    def reset_subtree(self, treenode):
        treenode.setParent(self.parent)
        self.parent = None

    def copy(self):
        return PyGP.tr_copy(self, self.CASH_MANAGER)

    def reset(self, nodeval):
        self.nodeval = nodeval
        self.visited = 0
        self.changeCashState(0)

    def setChilds(self, childs):
        if isinstance(self.nodeval, Func):
            if self.nodeval.arity != len(childs):
                raise ValueError("child num is not equal to the function arity {0} {1}".format(self.nodeval.arity, len(childs)))
            self.childs = childs.copy()
        else:
            raise ValueError("assigned childs to a terminal node dtype = %s " % self.dtype)

    @property
    def dtype(self):
        if isinstance(self.nodeval, Func):
            return "Func"
        elif isinstance(self.nodeval, int):
            return "Input"
        elif isinstance(self.nodeval, float):
            return "Const"

    def getCore(self):
        # return {"nodeval": self.nodeval, "node_id":self.node_id, "cash":self.cash, "visited":self.visited}
        if self.dtype == "Func":
            nodeval = [0, self.nodeval.id]
        elif self.dtype == "Input":
            nodeval = [1, self.nodeval]
        elif self.dtype == "Const":
            nodeval = [2, self.nodeval]
        else:
            raise NotImplementedError
        assert (len(self.cash) == 2)
        return [*nodeval, self.node_id, *self.cash, self.visited]

    def exp_draw(self):
        height = self.height()
        posi = (2 ** (height - 1)) * 2
        stack = [(self, posi)]
        next_stack = []
        depth = 0
        strings = []
        next_strings = []
        while stack:
            (node, cur_posi) = stack.pop(0)
            if node.dtype == "Func":
                str_ = str(node.nodeval.name)
            if node.dtype == "Input":
                str_ = str('x' + str(node.nodeval))
            if node.dtype == "Const":
                str_ = str('{:.1e}'.format(node.nodeval))
            strings.append((str_, cur_posi))

            if node.dtype == "Func":
                childs = node.getChilds()
                childs_len = len(childs)
                next_stack.extend(
                    [(childs[i], cur_posi + int((i - (childs_len - 1) / 2) * posi / (2 ** depth))) for i in
                     range(childs_len)])
                next_strings.extend(
                    [(cur_posi + int((i - (childs_len - 1) / 2) * posi / (2 ** (depth + 1))), i - (childs_len - 1) / 2)
                     for i in range(childs_len)])  # 用于画斜杠
            if len(stack) == 0:
                stack = next_stack
                next_stack = []
                strs: str = ""
                strs_: str = ""
                for i in range(len(strings)):
                    strs += str((strings[i][1] - len(strs)) * " ") + strings[i][0]
                if len(next_strings) > 0:
                    for i in range(len(next_strings)):
                        strs_ += str((next_strings[i][0] - len(strs_)) * " ")
                        if next_strings[i][1] < 0:
                            strs_ += str("/")
                        elif next_strings[i][1] > 0:
                            strs_ += str("\\")
                        elif next_strings[i][1] == 0:
                            strs_ += str("|")

                print(strs)
                print(strs_)
                strings = []
                next_strings = []
                depth += 1

    def dtype_update(self):
        if isinstance(self.nodeval, Func):
            self.dtype_ = "Func"
        elif isinstance(self.nodeval, int):
            self.dtype_ = "Input"
        elif isinstance(self.nodeval, float):
            self.dtype_ = "Const"
        else:
            raise NotImplementedError("self.dtype: ", type(self.nodeval))

    def setParent(self, parent):
        if parent is None:
            raise ValueError("parent is empty..")
        self.parent = parent
        self.parent[0].childs[parent[1]] = self

    def setCashId(self, cash_id):
        if not self.cash:
            raise ValueError("visit a cash without initialization")
        else:
            self.cash[1] = cash_id

    def setCashState(self, cash_state):
        if cash_state > 2 or cash_state < 0:
            raise ValueError("cash state is in the range of {0, 1, 2}")
        self.cash[0] = cash_state

    def changeCashState(self, cash_state):
        if not self.cash:
            self.cash = [cash_state, -1]
        else:
            self.cash[0] = cash_state

    def getCashId(self):
        return self.cash[1]

    def getCashState(self):
        return self.cash[0]

    def getArity(self):
        if isinstance(self.nodeval, Func):
            if self.nodeval.arity == 0:
                raise ValueError('getArity wrong.. ', self.nodeval.id)
            return self.nodeval.arity
        else:
            return 0

    def getChilds(self):
        if isinstance(self.nodeval, Func):
            return self.childs
        else:
            raise ValueError("try to get childs from a terminal node")

    # def dtype(self):
    #     return self.dtype

    def print_exp_subtree(self, noparent=False):
        str = ['']
        stack = []
        p = self.parent
        if noparent:
            self.parent = None
        PyGP.inorder_traversal(self, str, stack)
        if noparent:
            self.parent = p
        # print('expression: ', str[0])
        return str[0]

    def getChildSize(self):
        return self.child_size

    def childSize(self):
        size = 0
        for i in range(self.getArity()):
            size += self.childs[i].childSize()
        self.child_size = size + 1
        return self.child_size

    @property
    def size(self):
        size_ = 0
        stack = [self]
        while stack:
            node = stack.pop()
            if node.dtype == "Func":
                stack.extend(node.getChilds())
            size_ += 1
        return size_

    @property
    def inner_size(self):
        size_ = 0
        stack = [self]
        while stack:
            node = stack.pop()
            if node.dtype == "Func":
                stack.extend(node.getChilds())
                size_ += 1
        return size_

    def rlt_posi(self):

        root = self
        depth = 0
        while root.parent is not None:
            depth += 1
            root = root.parent[0]

        stack = [root]
        posi = -1
        while stack:
            pnode = stack.pop(0)
            if pnode.dtype == "Func":
                stack.extend(pnode.getChilds())
            posi += 1
            if pnode == self:
                return posi
        raise ValueError("Can not find the node from the tree")

    def relative_depth(self):
        pnode = self
        depth = 0
        while pnode.parent is not None:
            depth += 1
            pnode = pnode.parent[0]
        return depth

    def height(self):
        stack = [self]
        hgt_stack = []
        height = 0
        while len(stack) > 0:
            pnode = stack.pop(0)
            if pnode.dtype == "Func":
                hgt_stack.extend(pnode.getChilds())
            if len(stack) == 0:
                height += 1
                stack = hgt_stack
                hgt_stack = []
        return height

    def getAncestors(self):
        pnode = self
        ancestors = []
        while pnode.parent is not None:
            ancestors.append((pnode.parent[0], pnode.parent[1]))
            pnode = pnode.parent[0]
        return ancestors

    def zip(self):
        # program
        mainbody = []
        # treenode
        stack = [self]
        while stack:
            pnode = stack.pop(0)
            mainbody.extend(pnode.getCore())
            if pnode.dtype == "Func":
                stack.extend(pnode.getChilds())
        # print(int(array.array('f', [-1.2])[0]))
        # assert (0 == 1)
        return mainbody

    def getRange(self, init_range):
        stack = [self]
        hgt_stack = []
        layer_stack = [stack.copy()]
        while stack:
            pnode = stack.pop(0)
            if pnode.dtype == "Func":
                hgt_stack.extend(pnode.getChilds())
            if len(stack) == 0:
                if len(hgt_stack) > 0:
                    layer_stack.append(hgt_stack.copy())
                stack = hgt_stack
                hgt_stack = []

        data_rg = list(map(lambda x: (init_range[x.nodeval][0], init_range[x.nodeval][1]) if x.dtype == "Input" else (
            x.nodeval, x.nodeval), layer_stack.pop()))
        # self.exp_draw()
        while layer_stack:
            layer = layer_stack.pop()
            idx = 0
            data_rg_tmp = []
            for i, node in enumerate(layer):
                if node.getArity() > 0:
                    (rg_0, rg_1, step) = PyGP.rg_compute(node, idx, data_rg)
                    idx += step
                elif node.dtype == "Const":
                    rg_0 = rg_1 = node.nodeval
                elif node.dtype == "Input":
                    (rg_0, rg_1) = (float(init_range[node.nodeval][0]), float(init_range[node.nodeval][1]))
                data_rg_tmp.append((rg_0, rg_1))
            data_rg = data_rg_tmp
            # print(data_rg)
        return data_rg[0]


from PyGP import FunctionSet


class Program(Base):
    def __init__(
            self,
            pop_id,
            prog_id,
            init_depth=None,
            program=None,
            seman_sign=-1,
            funcs=None,
            root=None,
            const_range = None,
    ):
        self.__dict__.update(self.pop_dict[pop_id])
        if funcs is not None:
            self.funcs = funcs
        if const_range is not None:
            self.const_range = const_range
        self.pop_id = pop_id
        self.prog_id = prog_id
        self.program = program
        self.individual = []
        self.seman_sign = seman_sign
        if init_depth is not None:
            self.init_depth = init_depth
        if self.method == "half-and-half":
            if self.rd_st.uniform(0, 1) < 0.5:
                self.method_ = "grow"
            else:
                self.method_ = "full"
        else:
            self.method_ = self.method
        if self.program is not None:
            if not self.progCheck():
                raise ValueError("Invalid provided program.")
        if root is None:
            self.buildProgram(self.rd_st)
        else:
            self.root = root
            self.sizeUpdate()
            self.childSizeRenew()

    def setId(self, id):
        self.prog_id = id

    def copy(self, prog_id: int = None):

        c_root = PyGP.tr_copy(self.root, self.CASH_MANAGER)
        if prog_id is None:
            prog = Program(self.pop_id, self.prog_id, seman_sign=self.seman_sign, root=c_root)
        else:
            prog = Program(self.pop_id, prog_id, seman_sign=self.seman_sign, root=c_root)

        return prog

    def zip(self):
        # program
        mainbody = [self.pop_id, self.prog_id, self.seman_sign]
        # treenode
        stack = [self.root]
        while stack:
            pnode = stack.pop(0)
            mainbody.extend(pnode.getCore())
            if pnode.dtype == "Func":
                stack.extend(pnode.getChilds())
        # print(int(array.array('f', [-1.2])[0]))
        # assert (0 == 1)
        return mainbody

    def buildProgram(self, rand_state):  # pop是浅拷贝还是深拷贝？
        init_func = self.funcs.funcSelect(rand_state.randint(0, self.funcs.len() - 1))
        len_nterms = self.n_terms
        divide_check = []
        if self.init_depth == 1:
            if self.const_range is None:
                terminal = rand_state.randint(0, self.n_terms)
            else:
                len_nterms = self.n_terms * 10 + 1
                terminal = rand_state.randint(0, len_nterms + 1)
            if terminal == len_nterms:
                terminal = rand_state.uniform(self.const_range[0], self.const_range[1])
            else:
                terminal = len_nterms % self.n_terms
            self.root = TreeNode(terminal)
        else:
            self.root = TreeNode(init_func)
            stack = [self.root]
            tstack = []
            depth = 0
            funcs_size = self.funcs.len()
            while stack:
                node = stack.pop()
                childs = []
                for i in range(node.getArity()):
                    if depth < self.init_depth and (
                            self.method_ == "full" or rand_state.randint(0,
                                                                         funcs_size * 10 + self.n_terms) < funcs_size * 10):
                        r_oper = rand_state.randint(0, funcs_size)
                        tr = TreeNode(self.funcs.funcSelect(r_oper), parent=(node, i))
                        childs.append(tr)
                        if PyGP.INTERVAL_COMPUTE and (
                                self.funcs.funcSelect(r_oper).name == '/' or self.funcs.funcSelect(
                                r_oper).name == 'log'):
                            divide_check.append(tr)
                    else:  # if depth + 1 == self.init_depth:
                        if self.const_range is None:
                            terminal = rand_state.randint(0, self.n_terms)
                        else:
                            terminal = rand_state.randint(0, self.n_terms + 1)
                        if terminal == self.n_terms:
                            terminal = rand_state.uniform(self.const_range[0], self.const_range[1])
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
                smt_rg = childs[1].getRange(self.data_rg)
            if node.nodeval.name == 'log':
                smt_rg = childs[0].getRange(self.data_rg)
            if (smt_rg[0] <= 0. <= smt_rg[1] or math.fabs(smt_rg[0]) == 0 or math.fabs(smt_rg[1]) == 0) \
                    and ((node.nodeval.name == '/' and childs[0].print_exp_subtree() != childs[
                1].print_exp_subtree()) or node.nodeval.name == 'log'):
                if node.nodeval.name == '/':
                    new_tr = TreeNode(1., parent=(node, 1))
                    node.setChilds([childs[0], new_tr])
                elif node.nodeval.name == 'log':
                    new_tr = TreeNode(1., parent=(node, 0))
                    node.setChilds([new_tr])
        self.sizeUpdate()
        self.childSizeRenew()

    def set_root(self, root):
        self.root = root

    def debug_tr(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.dtype == "Func":
                stack.extend(node.getChilds())

            if node.node_id == 307422:
                print("can we get here??", node.getCashState())
            if (node.node_id == 307422 and node.getCashState() == 2):
                raise ValueError("can we get here??", node.getCashState())

    def tr_const_num(tr):
        depth_ = -1
        stack = [tr.root]
        cst_num = 0
        while stack:
            node = stack.pop()
            if node.dtype == "Func":
                stack.extend(node.getChilds())
            if node.dtype == "Const":
                cst_num += 1
        return cst_num

    @property
    def depth(self):
        depth_ = -1
        stack = [self.root]
        tstack = []
        while stack:
            node = stack.pop()
            if node.dtype == "Func":
                tstack.extend(node.getChilds())
            if not stack:
                stack = tstack
                tstack = []
                depth_ += 1
        return depth_

    @property
    def size(self):
        size_ = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.dtype == "Func":
                stack.extend(node.getChilds())
            size_ += 1
        return size_

    def sizeUpdate(self):
        self.length = self.size

    def childSizeRenew(self):
        self.root.childSize()

    def depth_nnum(self, depth):
        stack = [self.root]
        tstack = []
        idx = -1
        depth_ = -1
        while stack:
            tnode = stack.pop(0)
            if tnode.dtype == "Func":
                tstack.extend(tnode.getChilds())
            idx += 1
            if not stack:
                stack = tstack
                tstack = []
                depth_ += 1
            if depth_ == depth:
                return idx
        return idx

    def getSubTree(self, inid):
        if inid >= self.length:
            raise ValueError('the provided index out of tree size', inid, self.length)
        stack = [self.root]
        idx = -1
        while stack:
            tnode = stack.pop(0)
            if tnode.dtype == "Func":
                stack.extend(tnode.getChilds())
            idx += 1
            if idx == inid:
                return tnode
        raise ValueError('can not find the subtree, current idx: %d, inid: %d' % (idx, inid))

    def getSubTree_depbased(self, inid):
        if inid >= self.length:
            raise ValueError('the provided index out of tree size')
        stack = [self.root]
        idx = 0
        while stack:
            tnode = stack.pop(0)
            if tnode.dtype == "Func":
                stack.extend(tnode.getChilds())
            idx += 1
            if idx == inid:
                return tnode

        raise ValueError('can not find the subtree, current idx: %d, inid: %d' % (idx, inid))

    def print_exp(self):
        str = ['']
        stack = []
        PyGP.inorder_traversal(self.root, str, stack)
        # print('expression: ', str[0])
        return str[0]

    def exp_draw(self):
        self.root.exp_draw()

    def progCheck(self):
        raise NotImplementedError
