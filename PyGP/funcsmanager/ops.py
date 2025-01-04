from .func_init import TensorOp, FunctionSet
class EWiseAdd(TensorOp):
    op = 'add'
    def compute(self, a, b):
        return a + b
    def get_id(self):
        id = FunctionSet.funcid_search('add')
        if id is not None:
            return id
        raise ValueError("add function not implement yet")

class EWiseMul(TensorOp):
    op = 'mul'
    def compute(self, a, b):
        return a * b

    def get_id(self):
        id = FunctionSet.funcid_search('mul')
        if id is not None:
            return id
        raise ValueError("mul function not implement yet")
class EWiseSub(TensorOp):
    op = 'sub'
    def compute(self, a, b):
        return a * b

    def get_id(self):
        id = FunctionSet.funcid_search('sub')
        if id is not None:
            return id
        raise ValueError("sub function not implement yet")
class EWiseDiv(TensorOp):
    op = 'div'
    def compute(self, a, b):
        return a * b

    def get_id(self):
        id = FunctionSet.funcid_search('div')
        if id is not None:
            return id
        raise ValueError("div function not implement yet")
class EWiseWhere(TensorOp):
    def compute(self, a, b):
        return a * b

    def get_id(self):
        id = FunctionSet.funcid_search('where')
        if id is not None:
            return id
        raise ValueError("where function not implement yet")
class EWiseSin(TensorOp):
    def compute(self, a):
        raise NotImplementedError()

    def get_id(self):
        id = FunctionSet.funcid_search('where')
        if id is not None:
            return id
        raise ValueError("where function not implement yet")

class EWiseCos(TensorOp):
    def compute(self, a):
        raise NotImplementedError()

    def get_id(self):
        id = FunctionSet.funcid_search('where')
        if id is not None:
            return id
        raise ValueError("where function not implement yet")