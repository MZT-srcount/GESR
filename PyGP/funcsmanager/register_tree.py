from .func_init import TermNode, Func
from PyGP.base import TreeNode

def register(func_name, func, arity, used_function_set):
    # some check need to be done here.
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))

    args = [TermNode(i) for i in range(arity)]

    try:
        res = func(*args)
    except(ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.' % (func_name, arity))
    if not isinstance(res, TermNode):
        raise ValueError('supplied function %s with wrong output.' % func_name)

    def generate_child(idx, input):
        if input.op is not None:
            return TreeNode(used_function_set[input.op.op], node_id=-idx)
        else:
            return TreeNode(input.val, node_id=--idx)

    # 如果返回的是一个值或input？
    treenode = TreeNode(used_function_set[res.op.op], node_id=--1)
    stack = [[res], [treenode]]
    while stack.shape[1] > 0:
        unit = stack[0].pop()
        unit_tn = stack[1].pop()
        childs = []

        childs = map(lambda x: generate_child(x), unit.input)# [!]
        stack[1].extend(childs)
        stack[0].extend(unit.input)

        if len(childs) > 0:
            unit_tn.setChilds(childs)

    func_r = Func(-1, arity, func, treenode, fname=func_name, priority=-1)

    return (func_r, arity)

class TreeBasedRegister:
    def __call__(self, func_name, func, arity, used_function_set):
        return register(func_name, func, arity, used_function_set)