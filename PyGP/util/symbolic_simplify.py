from sympy import parse_expr, simplify, pprint, Symbol, preorder_traversal
def sympy_complexity(expr):
    print('expr: ', expr)
    try:
        exp = parse_expr(expr)
    except:
        exp = expr
    c = 0
    for arg in preorder_traversal(exp):
        c+=1
    return (exp, c)
def exp_simplify(expr):
    exp = parse_expr(expr)
    # pprint(simplify(exp))
    return sympy_complexity(simplify(exp))
