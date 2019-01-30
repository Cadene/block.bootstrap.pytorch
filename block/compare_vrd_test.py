import argparse
from bootstrap.compare import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--nb_epochs', default=-1, type=int)
    parser.add_argument('-d', '--dir_logs', default='', type=str, nargs='*')
    parser.add_argument('-m', '--metrics', type=str, action='append', nargs=3,
        metavar=('json', 'name', 'order'),
        default=[
            ['logs_predicate', 'eval_epoch.predicate.R_50', 'max'],
            ['logs_predicate', 'eval_epoch.predicate.R_100', 'max'],
            ['logs_rel_phrase', 'eval_epoch.phrase.R_50', 'max'],
            ['logs_rel_phrase', 'eval_epoch.phrase.R_100', 'max'],
            ['logs_rel_phrase', 'eval_epoch.rel.R_50', 'max'],
            ['logs_rel_phrase', 'eval_epoch.rel.R_100', 'max']
        ])
    parser.add_argument('-b', '--best', type=str)
    args = parser.parse_args()
    main(args)