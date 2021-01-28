import json
import os
import errno
import time

import numpy as np
import torch

import pdb

if os.name == "posix":
    import fcntl
else:
    fcntl = None


def create_model_name(args):
    """
    Creates name of a model based on the training arguments in the format:
    for iFlow:
    <network>_<lambda_implementation>_<flow_network>_<dataset_size>
    for iVAE:
        <network>_<dataset_size>
    """
    mname = f"{args.i_what}"
    if args.i_what == "iFlow":
        mname += f"_{args.nat_param_method}_{args.flow_type}"
    dataset_size = args.data_args.split("_")[0]
    mname += f"_{dataset_size}"
    return mname

def save_model(model, args, save_dir, seed, perf, loss):
    """
    Saves the model in the format: 
    for iFlow:
    <network>_<lambda_implementation>_<flow_network>_<dataset_size>_<network_seed>_<data_seed>.pt
    for iVAE:
        <network>_<dataset_size>_<network_seed>_<data_seed>.pt
    
    Input: 
        -model: trained model.
        -args: network training arguments
        -save_dir: location to save network
    """
    # Creating model save name
    f_name = create_model_name(args)
    f_name += f"_{args.seed}_{seed}.pt"
    torch.save({'model_state_dict': model.state_dict(),
            'args': args,
            'loss': loss,
            'perf': perf},
            save_dir + f_name)

def save_results(fname, args, perf, seed):
    """
    Saves the MCC results in a json file indexed on configuration.
    Input:
            -fname: string with file name
            -args: network training arguments
            -perf: float performance scores
            -seed: integer seed used to train model
    """
    # Create new dictionary if file does not exist.
    try:
        with open(fname) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    # Create key    
    key = create_model_name(args)
    # Create configuration entry
    if not key in results.keys():
        results[key] = []
    # Change results list to required length.    
    if len(results[key]) < seed:
        tmp = np.zeros(seed)
        tmp[:len(results[key])] = results[key]
        results[key] = tmp
    # Add result and dump JSON.    
    results[key][seed - 1] = np.round(perf, 4)
    try:
        results[key] = results[key].tolist()
    except AttributeError:
        pass
    with open(fname, 'w') as f:
        json.dump(results, f)


def make_dir(dir_name):
    if dir_name[-1] != '/':
        dir_name += '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def make_file(file_name):
    if not os.path.exists(file_name):
        open(file_name, 'a').close()
    return file_name


def get_exp_id(log_folder):
    log_folder = make_dir(log_folder)
    helper_id_file = log_folder + '.expid'
    if not os.path.exists(helper_id_file):
        with open(helper_id_file, 'w') as f:
            f.writelines('0')
    # helper_id_file = make_file(helper_id_file)
    with open(helper_id_file, 'r+') as file:
        st = time.time()
        while time.time() - st < 30:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # a = input()
                break
            except IOError as e:
                # raise on unrelated IOErrors
                if e.errno != errno.EAGAIN:
                    raise
                else:
                    print('sleeping')
                    time.sleep(0.1)
            except AttributeError:
                break
        else:
            raise TimeoutError('Timeout on accessing log helper file {}'.format(helper_id_file))
        prev_id = int(file.readline())
        curr_id = prev_id + 1

        file.seek(0)
        file.writelines(str(curr_id))
        try:
            fcntl.flock(file, fcntl.LOCK_UN)
        except AttributeError:
            pass
    return curr_id


def from_log(args, argv, logpath):
    """
    read from log, and allow change of arguments
    assumes that arguments are assigned using an = sign
    assumes that the first argument is --from-log. so argv[1] is of the form --from-log=id
    everything that comes after --from-log in sys.argv will be resolved and its value substituted for the one in the log
    """
    i = args.from_log
    d = {}
    new_d = vars(args).copy()
    args_not_from_log = []
    add_to_log = False
    if len(argv) > 2:
        add_to_log = True
    for a in argv[1:]:  # start from 2 if the from-log value is to be overwritten by the one in the log
        sp = a.split('=')
        args_not_from_log.append(sp[0][2:].replace('-', '_'))
    file = open(logpath)
    for line in file:
        d = json.loads(line)
        if d['id'] == i:
            break
    file.close()
    for a in args_not_from_log:
        d.pop(a)
    del d['id'], d['train_perf'], d['test_perf']
    new_d.update(d)
    return new_d, add_to_log


def checkpoint(path, exp_id, iteration, model, optimizer, loss, perf):
    sub_path = make_dir(path + str(exp_id) + '/')
    weights_path = sub_path + str(exp_id) + '_ckpt_' + str(iteration) + '.pth'
    print('.. checkpoint at iteration {} ..'.format(iteration))
    torch.save({'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'perf': perf},
               weights_path)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.97):
        self.momentum = momentum
        self.val = None
        self.avg = 0

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class Averager:
    def __init__(self):
        self.val = 0
        self.count = 0
        self.avg = 0
        self.sum = 0

    def reset(self):
        self.val = 0
        self.count = 0
        self.avg = 0
        self.sum = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


class Logger:
    """A logging helper that tracks training loss and metrics."""

    def __init__(self, logdir='log//', **metadata):
        self.logdir = make_dir(logdir)
        exp_id = get_exp_id(logdir)

        self.reset()

        self.metadata = metadata
        self.exp_id = exp_id

        self.log_dict = {}
        self.running_means = {}

    def get_id(self):
        return self.exp_id

    def add(self, key):
        self.running_means.update({key: Averager()})
        self.log_dict.update({key: []})

    def update(self, key, val):
        self.running_means[key].update(val)

    def _reset_means(self):
        for key in self.keys():
            self.running_means[key].reset()

    def reset(self):
        self.log_dict = {}
        self.running_means = {}

    def log(self):
        for key in self.keys():
            self.log_dict[key].append(self.running_means[key].avg)
        self._reset_means()

    def get_last(self, key):
        return self.log_dict[key][-1]

    def save_to_npz(self, path=None):
        if path is None:
            data_path = make_dir(self.logdir + 'data/')
            path = data_path + str(self.exp_id) + '.npz'
        else:
            if path[-4:] != '.npz':
                path += '.npz'
        for k, v in self.log_dict.items():
            self.log_dict[k] = np.array(v)
        np.savez_compressed(path, **self.log_dict)
        print('Log data saved to {}'.format(path))

    def save_to_json(self, path=None, method='last'):
        if path is None:
            path = make_file(self.logdir + 'log.json')
        with open(path, 'a') as file:
            log = {'id': self.exp_id}

            for k in self.keys():
                if method == 'last':
                    log.update({k: self.get_last(k)})
                elif method == 'full':
                    log.update({k: self.log_dict[k]})
                else:
                    raise ValueError('Incorrect method {}'.format(method))

            if 'device' in self.metadata:
                self.metadata.pop('device')
            log.update({'metadata': self.metadata})

            json.dump(log, file)
            file.write('\n')
        print('Log saved to {}'.format(path))

    def add_metadata(self, **metadata):
        self.metadata.update(metadata)

    def __len__(self):
        return len(self.log_dict)

    def __get__(self, key):
        self.get_last(key)

    def keys(self):
        return self.log_dict.keys()
