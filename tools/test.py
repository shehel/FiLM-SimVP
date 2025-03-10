# Copyright (c) CAIRI AI Lab. All rights reserved

import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)

from clearml import Task
try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    assert args.config_file is not None, "Config file is required for testing"
    config = update_config(config, load_config(args.config_file),
                           exclude_keys=['method', 'batch_size', 'val_batch_size'])
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]
    if not config['inference'] and not config['test']:
        config['test'] = True

    results_csv_path = "./results/p2_e1_q1.csv"
    task = Task.init(project_name='simvp/p2/e1/q1/tests', task_name=config['ex_name']+"_test")
    task_name = Task.get_task(task_id=task.name.split('_')[0]).name
    task.connect_configuration(config)
    # set multi-process settings
    setup_multi_processes(config)

    print('>'*35 + ' testing  ' + '<'*35)
    exp = BaseExperiment(args, task)
    rank, _ = get_dist_info()

    if config['inference'] and not config['test']:
        mse = exp.inference(task_name, results_csv_path)
    else:
        mse = exp.test(task_name, results_csv_path)
    if rank == 0 and has_nni and mse is not None:
        nni.report_final_result(mse)
