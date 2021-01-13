from multiprocessing import Process
from multi_instance_learner import main as mil_main
from config import get_config_str
import time


def make_config(args):
    arg_str = ""
    for arg_name, arg_val in args.items():
        arg_str += f" --{arg_name} {arg_val}"
    config = get_config_str(arg_str)
    return config


if __name__ == "__main__":
    procs_per_gpu = 1
    experiment_name = 'mil_fine_tune'

    available_gpus = [0, 1, 2]
    random_seeds = [0]
    mil_info = [
        {
            'fold': 0,
            'model_id': '3fdw8njf'
        },
        {
            'fold': 1,
            'model_id': '1l33ib88'
        },
        {
            'fold': 2,
            'model_id': '1d3hpvs0'
        },
        {
            'fold': 3,
            'model_id': 'jaqln5pf'
        },
        {
            'fold': 4,
            'model_id': '1wo44ahr'
        },
        {
            'fold': 5,
            'model_id': 'h4zoe30l'
        },
        {
            'fold': 0,
            'model_id': '1fd629ti'
        },
        {
            'fold': 1,
            'model_id': '1dfzl3zj'
        },
        {
            'fold': 2,
            'model_id': '1riklbmy'
        },
        {
            'fold': 3,
            'model_id': '36ck8xim'
        },
        {
            'fold': 4,
            'model_id': 'jup6btte'
        },
        {
            'fold': 5,
            'model_id': '2qyssv3y'
        }

    ]
    args = {
        'control_weight': 1.0,
        'epochs': 150,
        'lr_schedule': 'cyclic'
    }
    all_iteration_args = []
    for random_seed in random_seeds:
        for conf in mil_info:
            run_args = args.copy()
            run_args.update(conf)
            run_args['random_seed'] = random_seed
            run_args['experiment_name'] = experiment_name
            all_iteration_args.append(run_args)

    # now we load balance all the runs across all available GPUs
    current_load = [0] * len(available_gpus)
    current_processes = []
    while len(all_iteration_args) > 0:
        started_proc = False
        # check if there is available compute
        for gpu_index, gpu_number in enumerate(available_gpus):
            if current_load[gpu_index] < procs_per_gpu:
                started_proc = True
                proc_args = all_iteration_args[0]
                del all_iteration_args[0]
                proc_args['gpu_number'] = gpu_number
                current_load[gpu_index] += 1
                p = Process(target=mil_main, args=[make_config(proc_args)])
                p.start()
                # store the process handle the gpu used to run it
                current_processes.append((p, gpu_index))
                break
        # if nothing is started check if any of the processes are done
        if not started_proc:
            closed_proc = None
            for process_index, (process, gpu_index) in enumerate(current_processes):
                if not process.is_alive():
                    process.close()
                    current_load[gpu_index] -= 1
                    closed_proc = process_index
                    break
            if closed_proc is not None:
                del current_processes[closed_proc]
            # sleep to prevent just looping at max speed
            time.sleep(10)
    # waiting for all of them to be done
    while len(current_processes) > 0:
        closed_proc = None
        for process_index, (process, gpu_index) in enumerate(current_processes):
            if not process.is_alive():
                process.close()
                closed_proc = process_index
                break
        if closed_proc is not None:
            del current_processes[closed_proc]