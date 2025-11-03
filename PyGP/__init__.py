SEMANTIC_SIGN = True
INT_MAX = 2e20
BATCH_NUM = 1
CASHNUM_PERBLOCK = 1
SEMANTIC_NUM = 100
SEMANTIC_CDD = 3
LIBRARY_SIZE = 10000
PARALLEL = 3
CASH_SZIE_MULTI = 10
LIBRARY_SUPPLEMENT_INTERVAL = 5
LIBRARY_SUPPLEMENT_NUM = 5
DATA_TYPE = 8
INTERVAL_COMPUTE = False
CASH_OPEN = False
DEPTH_MAX = 8
POP_SIZE = 500
DEPTH_LIMIT = 8
DEPTH_MAX_SIZE = 8
INT_V = 8
MAX_SIZE = 2000
OPT = True
CRO_STG = 0
NEW_BVAL = False

SIZE_LIMIT = 100

from .data_funcs import *
from .funcsmanager import Func, FunctionSet
from .Semantic import *
from .cash import *
from .base import Base, TreeNode, Program, DataCollects, tr_copy_nc, treeid_update
from .population import Population
from . import test_module
from .Results import Curve, DCurve, curves
from .util import exp_simplify
from .utils import *
from .operators import *