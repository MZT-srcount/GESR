from multiprocessing import Pool
from PyGP import IdAllocator, CashManager
import dill
class Base:
    POP_SIZE = 0
    CASH_MANAGER = None
    ID_MANAGER = None
    PROC_MANAGER = None
    pop_dict = []
    time = 0
    # pool = None

    def _init(self, seed):
        from PyGP import SharedManager, PopSemantic
        # if Base.PROC_MANAGER is None:
        SharedManager.register("PopSemantic", PopSemantic)
        Base.PROC_MANAGER = SharedManager()
        Base.PROC_MANAGER._init(seed)
        Base.PROC_MANAGER.start()
        # if Base.ID_MANAGER is None:
        Base.ID_MANAGER = IdAllocator()
        # if Base.pool is None:
        #     Base.pool = Pool()

    def update(self, c_mngr, id_mngr, pop_size, pop_dict):
        Base.CASH_MANAGER = c_mngr
        Base.ID_MANAGER = id_mngr
        Base.ID_MANAGER.poolClear()
        Base.pop_dict = pop_dict
        Base.POP_SIZE = pop_size

    def parallel_prepare(self):
        self._cash = self.CASH_MANAGER
        Base.CASH_MANAGER = Base.CASH_MANAGER_fparallel
        Base.CASH_MANAGER_fparallel.update(Base.CASH_MANAGER.get_dict())

    def parallel_restore(self):
        Base.CASH_MANAGER = self._cash
        Base.CASH_MANAGER.update(Base.CASH_MANAGER_fparallel.get_dict())
    def getBasecore(self, forparallel=True):
        if forparallel:
            return (Base.CASH_MANAGER_fparallel, Base.ID_MANAGER, Base.POP_SIZE, Base.pop_dict)
        else:
            raise NotImplementedError
    def property_add(self, dict):
        self.pop_dict.append(dict)

    def _cash_set(self, init_posi, size, seed):
        if self.CASH_MANAGER is None:
            Base.CASH_MANAGER = CashManager(init_posi, size)
            Base.CASH_MANAGER_fparallel = Base.PROC_MANAGER.CashManager(init_posi, size)
        else:
            Base.CASH_MANAGER.reset(init_posi, size)
            Base.CASH_MANAGER_fparallel.reset(init_posi, size)
        # Base.PROC_MANAGER._init(self.getBasecore(), seed)
        Base.PROC_MANAGER._init(seed)