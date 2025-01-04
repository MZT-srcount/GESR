import os
import pickle
import re
import pandas as pd
from sympy import parse_expr, preorder_traversal, simplify, Integer, Float
import signal
TIMEOUT = 60
def handler(signum, frame):
    raise Exception('Simplify timed out')




def round_floats(ex1, precision=12):
    ex2 = ex1

    for a in preorder_traversal(ex1):
        if isinstance(a, Float):
            if abs(a) < 0.1 ** precision:# or (len(str(ex1)) <= 5 and abs(a) < 0.1 ** precision):
                ex2 = ex2.subs(a,Integer(0))
            else:
                ex2 = ex2.subs(a, Float(round(a, 3),3))
    return ex2

        


def sympy_complexity(expr):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(10)
    try:
        exp = simplify(round_floats(parse_expr(expr)), ratio=1)
        if str(exp) == '0' or str(exp) == 'zoo' or str(exp) == 'nan':
            exp = simplify(parse_expr(expr), ratio=1)
            
    except:
        print("can not simplify")
        exp = parse_expr(expr)
    c = 0
    for arg in preorder_traversal(exp):
        c+=1
    return (exp, c)

if __name__ == '__main__':


    
    dataset_dir='./Dataset/hard/true_eq/'
    read_file='./result/Feynman_hard_dummy_0.txt'
    write_file = './result/sym_exp_clts_hard_0.txt'

    dataset_list = sorted(os.listdir(dataset_dir))

    
    real_eqs = {}
    for i in range(len(dataset_list)):
        with open(dataset_dir + dataset_list[i], 'rb') as f:
            str_ = str(pickle.load(f))
            d = parse_expr(str_.replace('pi','3.1415926535'))
            d1 = round_floats(d)
            if str(d1) != '0' and str(d1) != 'zoo' and str(d1) != 'nan':
                d = d1
            real_eqs[dataset_list[i][:-4]] = d
            
    dataset = []
    datas = {}
    exps = {}

    match_num = 0
    avg_complexity = 0
    sym_exps = []
    names = []
    complexs = []
    times = []
    with open(read_file, 'r') as files:
        pat = ', '
        cur_dset = ''
        lines = files.readlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if re.findall(pat, line):
                # print(data)
                words = line.replace('\n', '').split(', ')
                cur_dset = words
                print('words: ', words)
                names.append(words[1])
            elif line.strip() and not re.findall('Sth wrong', line) and not re.findall('==========', line):
                data = line.replace('\n', '').split(' ')
                data.append(cur_dset)
                idx += 1
                exp = lines[idx].replace('fabs', '')
                # print('exp: ', exp)
                sc = sympy_complexity(exp)
                sym_exp = sc[0]
                complexs.append(sc[1])
                times.append(float(data[2]))


                signal.signal(signal.SIGALRM, handler)
                signal.alarm(10)
                is_constant = False
                try:
                    
                    sym_diff = sym_exp - real_eqs[cur_dset[1][:-4]]
                    sym_frac = sym_exp / real_eqs[cur_dset[1][:-4]]
                    match = False
                    print("sym_diff: ", sym_diff, '. real: ', real_eqs[cur_dset[1][:-4]])
                    is_constant = (str(sym_diff) == '0' or sym_frac.is_constant() or sym_diff.is_constant()) 
                except Exception as e:
                    print(e)
                finally:
                    # 取消定时器
                    signal.alarm(0)
                print('exps: ', str(sym_exp), len(str(sym_exp)), sc[1])
                print('origin exps: ', exp)
                if is_constant and not str(sym_exp) == 'nan' and (not str(sym_diff) == 'nan' or not str(sym_frac) == 'nan') and not (str(sym_exp) == '0' and str(real_eqs[cur_dset[1][:-4]]) != '0'):
                    match = True
                    print('simplified sym_diff:',sym_diff, sym_frac)
                    print('simplified exp: ', sym_exp)
                    match_num += 1
                    print('match_num: ', match_num)

                # print('sym_exp:', sym_exp)
                sym_exps.append([str(data[0]), str(match), str(real_eqs[words[1][:-4]]), str(sym_exp), exp])
            idx += 1
    print('match num: ', match_num / len(sym_exps))
    # print(times)
    print(complexs)
    assert 0==1
    with open(write_file, '+a') as file:
        for word, exp in zip(names, sym_exps):
            # print(exp)
            file.write(word + '\n')
            file.write(', '.join(exp) + '\n')

