import math
from pycuda.compiler import SourceModule
INT_MAX = 2e20
def execution_test(exe_progs, init_posi, dataset, fitness, com_fitness):
    fitnesses = []
    for i in range(len(init_posi) - 1):
        # print('=================', i, init_posi[i],  '=======================')
        fits = []
        end = 0
        start = init_posi[i]
        if i == len(init_posi) - 1:
            end = len(exe_progs) - 1
        else:
            end = init_posi[i + 1] - 1
        for j in range(len(dataset[0])):
            # print(j, start, init_posi[i + 1])
            start_tmp = start
            inval = {}
            res = 0
            idx = 0
            while(start_tmp < end):
                if(exe_progs[start_tmp] == 0):
                    if exe_progs[start_tmp + 2] < len(dataset):
                        data_1 = dataset[exe_progs[start_tmp + 2]][j]
                    else:
                        data_1 = inval[exe_progs[start_tmp + 2]]
                    if exe_progs[start_tmp + 3] < len(dataset):
                        data_2 = dataset[exe_progs[start_tmp + 3]][j]
                    else:
                        data_2 = inval[exe_progs[start_tmp + 3]]

                    # if i == 74 and j == 8:
                    #     print('+', data_1, data_2, data_1 + data_2)
                    res = data_1 + data_2

                if (exe_progs[start_tmp] == 1):
                    if exe_progs[start_tmp + 2] < len(dataset):
                        data_1 = dataset[exe_progs[start_tmp + 2]][j]
                    else:
                        data_1 = inval[exe_progs[start_tmp + 2]]
                    if exe_progs[start_tmp + 3] < len(dataset):
                        data_2 = dataset[exe_progs[start_tmp + 3]][j]
                    else:
                        data_2 = inval[exe_progs[start_tmp + 3]]

                    # if i == 74 and j == 8:
                    #     print('-', data_1, data_2, data_1 - data_2)
                    res = data_1 - data_2

                if (exe_progs[start_tmp] == 2):
                    if exe_progs[start_tmp + 2] < len(dataset):
                        data_1 = dataset[exe_progs[start_tmp + 2]][j]
                    else:
                        data_1 = inval[exe_progs[start_tmp + 2]]
                    if exe_progs[start_tmp + 3] < len(dataset):
                        data_2 = dataset[exe_progs[start_tmp + 3]][j]
                    else:
                        data_2 = inval[exe_progs[start_tmp + 3]]

                    # if i == 74 and j == 8:
                    #     print('*', idx, data_1, data_2, data_1 * data_2)
                    res = data_1 * data_2

                if (exe_progs[start_tmp] == 3):
                    if exe_progs[start_tmp + 2] < len(dataset):
                        data_1 = dataset[exe_progs[start_tmp + 2]][j]
                    else:
                        data_1 = inval[exe_progs[start_tmp + 2]]
                    if exe_progs[start_tmp + 3] < len(dataset):
                        data_2 = dataset[exe_progs[start_tmp + 3]][j]
                    else:
                        data_2 = inval[exe_progs[start_tmp + 3]]

                    # if i == 74 and j == 8:
                    #     print('/', idx, data_1, data_2, data_1 / data_2, INT_MAX)
                    if math.fabs(data_2) < 1e-6:
                        res = INT_MAX
                    else:
                        res = data_1 / data_2
                if start_tmp + 1 >= len(exe_progs):
                    print("lens out", len(exe_progs), start_tmp + 1)
                inval[exe_progs[start_tmp + 2 + exe_progs[start_tmp + 1]]] = res
                start_tmp += 3 + exe_progs[start_tmp + 1]
                idx += 1
            # if(j == 5):
            #     print(exe_progs[start: end], start, end)
            #     for i in range(len(dataset)):
            #         print(dataset[i][5])
            #     print(res)
            fits.append(res)
        fit = 0
        if i == 74:
            print(fits)
            # print("{0}:{1}".format(k, fits[k]) for k in range(len(fits)))
        # print(fitness)
        for j in range(len(fits)):
            fit += math.fabs(fits[j] - fitness[j])
        fit /= len(fits)
        fitnesses.append(fit)
    for i in range(len(fitnesses)):
        print(i, fitnesses[i], com_fitness[i], len(init_posi))

def backpragation_test(exe_progs, init_posi, locate, locate_posi, dataset, fitness, com_fitness, const_vals):
    fitnesses = []
    for i in range(len(init_posi) - 1):
        # print('=================', i, init_posi[i],  '=======================')
        fits = []
        end = 0
        start = init_posi[i]
        if i == len(init_posi) - 1:
            end = len(exe_progs) - 1
        else:
            end = init_posi[i + 1] - 1
        for j in range(len(dataset[0])):
            # print(j, start, init_posi[i + 1])
            start_tmp = start
            inval = {}
            res = 0
            idx = 0
            while(start_tmp < end):
                if(exe_progs[start_tmp] == 0):
                    if  exe_progs[start_tmp + 2] < 0:
                        data_1 = const_vals[-exe_progs[start_tmp + 2]]
                    elif  exe_progs[start_tmp + 2] < len(dataset):
                        data_1 = dataset[exe_progs[start_tmp + 2]][j]
                    else:
                        data_1 = inval[exe_progs[start_tmp + 2]]
                    if  exe_progs[start_tmp + 3] < 0:
                        data_2 = const_vals[-exe_progs[start_tmp + 3]]
                    elif exe_progs[start_tmp + 3] < len(dataset):
                        data_2 = dataset[exe_progs[start_tmp + 3]][j]
                    else:
                        data_2 = inval[exe_progs[start_tmp + 3]]


                    # if i == 74 and j == 8:
                    #     print('+', data_1, data_2, data_1 + data_2)
                    if locate[locate_posi[i] + idx] == 0:
                        res -= data_2
                    else:
                        res -= data_1

                if (exe_progs[start_tmp] == 1):
                    if  exe_progs[start_tmp + 2] < 0:
                        data_1 = const_vals[-exe_progs[start_tmp + 2]]
                    elif  exe_progs[start_tmp + 2] < len(dataset):
                        data_1 = dataset[exe_progs[start_tmp + 2]][j]
                    else:
                        data_1 = inval[exe_progs[start_tmp + 2]]
                    if  exe_progs[start_tmp + 3] < 0:
                        data_2 = const_vals[-exe_progs[start_tmp + 3]]
                    elif exe_progs[start_tmp + 3] < len(dataset):
                        data_2 = dataset[exe_progs[start_tmp + 3]][j]
                    else:
                        data_2 = inval[exe_progs[start_tmp + 3]]

                    # if i == 74 and j == 8:
                    #     print('-', data_1, data_2, data_1 - data_2)
                    if locate[locate_posi[i] + idx] == 0:
                        res += data_2
                    else:
                        res = data_1 - res

                if (exe_progs[start_tmp] == 2):
                    if  exe_progs[start_tmp + 2] < 0:
                        data_1 = const_vals[-exe_progs[start_tmp + 2]]
                    elif  exe_progs[start_tmp + 2] < len(dataset):
                        data_1 = dataset[exe_progs[start_tmp + 2]][j]
                    else:
                        data_1 = inval[exe_progs[start_tmp + 2]]
                    if  exe_progs[start_tmp + 3] < 0:
                        data_2 = const_vals[-exe_progs[start_tmp + 3]]
                    elif exe_progs[start_tmp + 3] < len(dataset):
                        data_2 = dataset[exe_progs[start_tmp + 3]][j]
                    else:
                        data_2 = inval[exe_progs[start_tmp + 3]]
                    # if i == 74 and j == 8:
                    #     print('*', idx, data_1, data_2, data_1 * data_2)
                    if locate[locate_posi[i] + idx] == 0:
                        if math.fabs(data_2) < 1e-6:
                            res /= data_2
                        else:
                            res = INT_MAX
                    else:
                        if math.fabs(data_1) < 1e-6:
                            res /= data_1
                        else:
                            res = INT_MAX

                if (exe_progs[start_tmp] == 3):
                    if  exe_progs[start_tmp + 2] < 0:
                        data_1 = const_vals[-exe_progs[start_tmp + 2]]
                    elif  exe_progs[start_tmp + 2] < len(dataset):
                        data_1 = dataset[exe_progs[start_tmp + 2]][j]
                    else:
                        data_1 = inval[exe_progs[start_tmp + 2]]
                    if  exe_progs[start_tmp + 3] < 0:
                        data_2 = const_vals[-exe_progs[start_tmp + 3]]
                    elif exe_progs[start_tmp + 3] < len(dataset):
                        data_2 = dataset[exe_progs[start_tmp + 3]][j]
                    else:
                        data_2 = inval[exe_progs[start_tmp + 3]]

                    # if i == 74 and j == 8:
                    #     print('/', idx, data_1, data_2, data_1 / data_2, INT_MAX)
                    if locate[locate_posi[i] + idx] == 0:
                        res *= data_2
                    else:
                        if math.fabs(res) < 1e-6:
                            res = data_1 / res
                        else:
                            res = INT_MAX
                if start_tmp + 1 >= len(exe_progs):
                    print("lens out", len(exe_progs), start_tmp + 1)
                inval[exe_progs[start_tmp + 2 + exe_progs[start_tmp + 1]]] = res
                start_tmp += 3 + exe_progs[start_tmp + 1]
                idx += 1
            # if(j == 5):
            #     print(exe_progs[start: end], start, end)
            #     for i in range(len(dataset)):
            #         print(dataset[i][5])
            #     print(res)
            fits.append(res)
        fit = 0
        if i == 74:
            print(fits)
            # print("{0}:{1}".format(k, fits[k]) for k in range(len(fits)))
        # print(fitness)
        for j in range(len(fits)):
            fit += math.fabs(fits[j] - fitness[j])
        fit /= len(fits)
        fitnesses.append(fit)
    for i in range(len(fitnesses)):
        print(i, fitnesses[i], com_fitness[i], len(init_posi))

test_mod = SourceModule("""
__global__ void printf_(double* data, int offset, int len, size_t pitch){
    if(threadIdx.x == 0){
        printf("---+--test_printf:%d---+---\\n",pitch);
        int prog_posi = pitch * 74;
        double* data_p = (double*)((char*)data + offset);
        for(int i = 0; i < len; ++i){
            printf("%d:%f ",i, data_p[i]);
        }
        printf("\\n\\n");
    }
}
""")