
import os
import argparse

DATASET_SPECIAL = [i for i in range(120)]
if __name__ == '__main__':
    RUN = 1

    parser = argparse.ArgumentParser(
        description="Autorun a method.", add_help=False)
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-job_id', dest='job_id', type=int, default=None,
                        help='the index of dataset')
    parser.add_argument('-save_file', dest='save_file', type=str, default="./result/livermore_result.txt",
                        help='the index of dataset')
    parser.add_argument('-train_dataset', dest='train_set', type=str, default="./Dataset/livermore/train/",
                        help='the index of dataset')
    parser.add_argument('-test_dataset', dest='test_set', type=str, default="./Dataset/livermore/test/",
                        help='the index of dataset')
    parser.add_argument('-seed', dest='seed', type=int, default=1111,
                        help='seed for model')
    args = parser.parse_args()

    save_file = args.save_file
    dataset_dir=args.train_set
    dataset_list = sorted(os.listdir(dataset_dir))

    for dset in range(len(dataset_list)):
        train_set = dataset_dir + '/' + dataset_list[dset]
        test_set = args.test_set + '/' + dataset_list[dset]
        with open(save_file, 'a+') as f:
            f.write("\n{0}, {1}\n\n".format(dset, dataset_list[dset]))
        for i in range(RUN):
            command = 'python {SCRIPT}.py -run_id {RUNID} -job_id {JOBID} -file_name {FILE} -dataset_name {DNAME} -train_set {TRSET} -test_set {TESET} -seed {SEED}'.format(
                SCRIPT='train', RUNID=i % RUN,JOBID=dset, FILE=save_file, DNAME=dataset_list[dset], TRSET=train_set, TESET=test_set, SEED=args.seed)
            os.system(command)
        