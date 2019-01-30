import argparse
from bootstrap.compare import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--nb_epochs', default=-1, type=int)
    parser.add_argument('-d', '--dir_logs', default='', type=str, nargs='*')
    parser.add_argument('-m', '--metrics', type=str, action='append', nargs=3,
        metavar=('json', 'name', 'order'), default=[
            ['logs_test', 'eval_epoch.accuracy_top1', 'max'],
            ['logs_test', 'eval_epoch.tdiuc.acc_mpt_a', 'max'],
            ['logs_test', 'eval_epoch.tdiuc.acc_mpt_h', 'max'],
            ['logs_test', 'eval_epoch.tdiuc.acc_mpt_a_norm', 'max'],
            ['logs_test', 'eval_epoch.tdiuc.acc_mpt_h_norm', 'max']
        ])
    parser.add_argument('-b', '--best', type=str, nargs=3)
    args = parser.parse_args()
    main(args)
