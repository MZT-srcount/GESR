import re
import pandas as pd
from sympy import parse_expr, preorder_traversal

def get_middle_r2(data):
    sorted_data = sorted(data, key=lambda x: x[0])
    return sorted_data[int(len(sorted_data) / 2)]

def sympy_complexity(expr):
    print('expr: ', expr)
    try:
        exp = parse_expr(expr)
    except:
        assert (0==1)
    c = 0
    for arg in preorder_traversal(exp):
        c+=1
    return (exp, c)

if __name__ == '__main__':
    dataset = []
    datas = {}
    exps = {}

    read_file='PMLB_res_new_fri.txt'
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
                if datas.get(words[1]) is None:
                    datas[words[1]] = []
                    exps[words[1]] = []
                cur_dset = words
                print('words: ', words)
            elif line.strip() and not re.findall('Sth wrong', line) and not re.findall('==========', line):
                data = line.replace('\n', '').split(' ')
                data.append(cur_dset)
                idx += 1
                exp = lines[idx]
                print('idx: ', idx)
                sym_exp = sympy_complexity(exp)
                # print(len(exp.replace(' ', '')), exp)
                # print( sym_exp[1], sym_exp[0])
                exps[cur_dset[1]].append(sym_exp[0])
                data.append(str(sym_exp[1]))
                datas[cur_dset[1]].append(data)
            idx += 1
    R2_, RMSE, exp_, train_time_, train_R2, dset = [], [], [], [], [], []
    for key in datas:
        data = datas[key]
        if len(data) > 0:
            print(data)
            res = get_middle_r2(data)
            R2_.append(res[0])
            exp_.append(res[len(res) - 1])
            train_time_.append(res[2])
            train_R2.append(res[3])
            RMSE.append(res[1])
            dset.append(res[len(res) - 2][1])
    data = {'Dataset': dset, 'R2': R2_, 'RMSE': RMSE, 'TrainTime': train_time_, 'ModelComplexity': exp_}
    df = pd.DataFrame(data)
    df.to_csv('pmlb_res_fri.csv', index=False)
