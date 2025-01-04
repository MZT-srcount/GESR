from .semantic_collects import PopSemantic

class TR:
    def __init__(self, mngr = None, idx = None, expr = None, smt = None, **kwargs):
        if mngr is not None:
            self.mngr = mngr
        if idx is not None:
            self.seg = idx
        if expr is not None:
            self.expr = expr
        if smt is not None:
            self.smt = smt
        self.__dict__.update(kwargs)
        # for key, value in kwargs.items:
            
        # self.expr, self.smt, self.tr = None, None, None

    @property
    def get(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            raise ValueError("can not find the corresponding values")
        # self.__dict__.get(str)
    @property
    def print(self):
        if not hasattr(self, 'expr'):
            # ht, idx = self.seg[0], self.seg[1]
            self.expr = self.mngr.get_expr(self.seg)#self.mngr.lbr_keys[ht][idx]
        return self.expr


    @property
    def semantic(self):
        if hasattr(self, 'smt'):
            return self.smt
        expr = self.print
        self.smt = self.mngr.get_semantic(expr)
        return self.smt

    @property
    def tree(self):
        if hasattr(self, 'tr'):
            return self.tr
        # expr = self.print
        # ht, idx = self.seg[0], self.seg[1]
        self.tr = self.mngr.get_tr(self.seg)
        return self.tr

    @property
    def range(self):
        if not hasattr(self, 'rg'):
            expr = self.print
            self.rg = self.mngr.get_range(expr)
        return self.rg
    
    @property
    def inner_size(self):
        if not hasattr(self, 'isize'):
            self.isize = self.mngr.get_inner_size(self.seg)
        return self.isize

class SManager():
    # def __init__(self, proc_manager, **kwargs):
    #     self.proc_manager = proc_manager
    #     # self._init(**kwargs)
    #     self.base_manager = self.proc_manager.PopSemantic(**kwargs)
    
    def __init__(self, pop_semantic):
        # self._init(**kwargs)
        self.base_manager = pop_semantic
    def reset(self):
        self.base_manager.reset()
    def set_library(self, *args):
        self.base_manager.set_library(*args)
    def select(self, *args):
        self.base_manager.select(*args)
    # def renew(self, **kwargs):
    #     self._init(**kwargs)
    #     self.base_manager = self.proc_manager.PopSemantic(**kwargs)
    def extend(self, *args):
        self.base_manager.extend(*args)
    def snode_merge(self):
        return self.base_manager.snode_merge()
    def data_load(self, *args):
        self.base_manager.data_load(*args)
    def smt_clts(self):
        self.base_manager.smt_clts()
    def get_snode_tgsmt(self, *args):
        return self.base_manager.get_snode_tgsmt(*args)
    def get_tgsmt_d(self, *args):
        return self.base_manager.get_tgsmt_d(*args)
    def get_drvt_d(self, *args):
        return self.base_manager.get_drvt_d(*args)
    def get_smt_size(self, *args, **kwargs):
        return self.base_manager.get_smt_size(*args, **kwargs)
    def get_datarg(self):
        return self.base_manager.get_datarg()
    def compute_tg(self, *args):
        return self.base_manager.compute_tg(*args)
    def get_tgnode_posi(self, *args):
        return self.base_manager.get_tgnode_posi(*args)
    def get_smt_trs(self, *args):
        trs_clt = []
        (segs, exprs, smts) = self.base_manager.get_smt_trs(*args)
        for i in range(len(segs)):
            trs_clt.append(TR(self.base_manager, segs[i], expr=exprs[i], smt=smts[i]))
        return trs_clt



