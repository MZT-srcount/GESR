'''
Author: your name
Date: 2023-08-03 15:46:09
LastEditTime: 2023-08-07 11:40:51
LastEditors: your name
Description:
FilePath: \PyGP\PyGP\func_init.py
'''
import PyGP
from typing import Optional, List


class Op:
    def __call__(self):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class TensorOp(Op):
    def __call__(self, *args):
        tensor = TermNode.make_from_op(self, args)
        return tensor


class Value:
    """"A value node in computational graph, for cg collection in function"""
    op: Optional[Op]
    input: List["TermNode"]

    def _init(self, op: Optional[Op],
              inputs: list["TermNode"]):
        # print('type op: ', type(op))
        self.op = op
        self.input = inputs


class TermNode(Value):
    def __init__(self,
                 val,
                 dtype=None,
                 **kwargs):
        if dtype == 'int':
            self.termtype = 'const'
        self.val = val
        self._init(None, [])

    def __add__(self, other):
        if isinstance(other, TermNode):
            return PyGP.ops.EWiseAdd()(self, other)
        else:
            return

    def __sub__(self, other):
        if isinstance(other, TermNode):
            return PyGP.ops.EWiseSub()(self, other)

    def __mul__(self, other):
        if isinstance(other, TermNode):
            return PyGP.ops.EWiseMul()(self, other)

    def __div__(self, other):
        if isinstance(other, TermNode):
            return PyGP.ops.EWiseDiv()(self, other)

    def sin(self):
        return PyGP.ops.EWiseSin()(self)

    def cos(self):
        return PyGP.ops.EWiseSin()(self)

    def sin(self):
        return PyGP.ops.EWiseSin()(self)

    def where(self, *args):
        return PyGP.ops.EWiseWhere()(self, args)

    @staticmethod
    def make_from_op(op: Op, inputs: list["TermNode"]):
        termnode = TermNode.__new__(TermNode)
        termnode._init(op, inputs)
        return termnode


class Func(object):
    def __init__(self, idx, arity, function=None, root=None, fname=None, priority=0):
        if function is not None:
            self.function = function
        self.id = idx
        self.arity = arity
        self.name = fname
        self.priority = priority
        if root is not None:
            self.root = root

    def transform(self, input: []):
        output = self.function(input)
        # transform the computational graph to a tree expression
        raise NotImplementedError

    def arity(self):
        return self.arity


class FunctionSet(object):
    def __init__(self, type):

        self.init_function_set = {}
        self.used_function_set = {}
        self.init_function_set['add'] = Func(idx=0, arity=2, fname='+', priority=0)
        self.init_function_set['sub'] = Func(idx=1, arity=2, fname='-', priority=1)
        self.init_function_set['mul'] = Func(idx=2, arity=2, fname='*', priority=2)
        self.init_function_set['div'] = Func(idx=3, arity=2, fname='/', priority=3)
        self.init_function_set['sin'] = Func(idx=4, arity=1, fname='sin', priority=4)
        self.init_function_set['cos'] = Func(idx=5, arity=1, fname='cos', priority=4)
        self.init_function_set['log'] = Func(idx=6, arity=1, fname='log', priority=4)
        self.init_function_set['exp'] = Func(idx=7, arity=1, fname='exp', priority=4)
        self.init_function_set['sqrt'] = Func(idx=8, arity=1, fname='sqrt', priority=4)
        self.init_function_set['fabs'] = Func(idx=9, arity=1, fname='fabs', priority=4)
        self.init_funcs_name = list(self.init_function_set.keys())
        self.arity_function_set = {}
        self.reg_idx = 4
        self.max_arity_ = 0
        self.type_=type

    def init(self, funcs):
        if self.used_function_set:  # to avoid some unexpected error, init function can be call only once
            raise ValueError("init function has already been called")
        # it should be put into other initialization way
        if isinstance(funcs, list):
            for i in range(len(funcs)):
                assert(self.init_function_set.get(funcs[i])) #raise ValueError( "find an undefined function symbol '%s' in the function list, it should be registered first" %funcs[i])
                if self.init_function_set.get(funcs[i]):
                    self.used_function_set[funcs[i]] = self.init_function_set[funcs[i]]

        elif funcs is None or len(funcs) == 0:
            self.used_function_set = self.init_function_set

        for key, value in self.used_function_set.items():
            if not self.arity_function_set.get(value.arity):
                self.arity_function_set[value.arity] = [value]
            else:
                self.arity_function_set[value.arity].append(value)
        self.funcs_name = funcs

    @classmethod
    def funcid_search(cls, str):
        return cls.init_function_set[str]

    def append(self, func):
        self.init_function_set.append(func)

    def len(self):
        return len(self.used_function_set)

    def funcSelect(self, idx):
        if idx >= len(self.funcs_name):
            raise ValueError("idx out of funcs range, where idx = '%d' , and funcs size = '%d'", idx,
                             len(self.funcs_name))
        return self.used_function_set[self.funcs_name[idx]]

    def funcSelect_oid(self, idx):
        if idx >= len(self.init_funcs_name):
            raise ValueError("idx out of funcs range, where idx = '%d' , and funcs size = '%d'", idx,
                             len(self.funcs_name))
        return self.used_function_set[self.init_funcs_name[idx]]

    def funcSelect_n(self, name):
        if self.used_function_set.get(name) is None:
            raise ValueError("can not find the operator name: ", name)
        return self.used_function_set[name]

    def register(self, func_name, func, arity):
        # some check need to be done here.
        if isinstance(self.type_, PyGP.base.TreeNode):
            from .register_tree import TreeBasedRegister
            (func_, arity_) = TreeBasedRegister(func_name, func, arity, self.used_function_set)
            self.used_function_set[func_name] = func_
            self.funcs_name.append(func_name)

            if not self.arity_function_set.get(arity_):
                self.arity_function_set[arity] = [func_]
            else:
                self.arity_function_set[arity].append(func_)
        else:
            raise NotImplementedError("Can not recognize the basic func type: ", self.type_)

    def max_arity(self):
        if self.max_arity_ == 0:
            for i in range(len(self.used_function_set)):
                if self.used_function_set[self.funcs_name[i]].arity > self.max_arity_:
                    self.max_arity_ = self.used_function_set[self.funcs_name[i]].arity
        return self.max_arity_
