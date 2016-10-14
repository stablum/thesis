import glob
import os
import sys
import re
import pandas as pd
pd.set_option('display.max_colwidth', -1)

eventual_timestamp_regex = "[0-9 :]*"
parser = re.compile(eventual_timestamp_regex+"(.*):[ ]*(.*)")
epoch_parser = re.compile(eventual_timestamp_regex+"epoch[: ]*([0-9]*)")
testing_rmse_parser = re.compile(eventual_timestamp_regex+"testing RMSE[: ]*(.*)")
harvest_parser = re.compile("[\./]*harvest_([a-z_]*[a-z]).*")
shorten_dict = {
    "learning rate":"lr",
    "learning_rate":"lr",
    "lr annealing T":"T",
    "update algorithm":"upd",
}

not_shortened = [
    "K",
    "hid_dim",
    "chan_out_dim",
    "optimizer",
    "n_epochs",
    "g_in",
    "g_rij",
    "dropout_p",
    "regularization_lambda",
]

defaults = {
    'K':10,
    'hid_dim':100,
    'chan_out_dim':10
}

def shorten(name):
    if name in shorten_dict.keys():
        return shorten_dict[name]
    else:
        return name

def allowed(name):
    if name in shorten_dict.keys():
        return True
    if name in not_shortened:
        return True
    return False

def process_line(line):
    match = parser.match(line)
    if match is None:
        return None
    name,val = match.groups()[-2:]
    if not allowed(name):
        return None
    return shorten(name),val

def complete_default(params):
    for k,v in defaults.items():
        if k not in params.keys():
            params[k] = v
    return params

def check_match(parser,line):
    match = parser.match(line)
    if match is not None:
        ret = match.groups()[0]
        return ret
    return None

def process_file(filename):
    params = {}
    max_epoch = None
    last_testing_rmse = 999
    best_testing_rmse = 999
    with open(filename,'r') as f:
        for line in f:
            param = process_line(line)
            if param is not None and param[0] not in params.keys():
                params.update((param,))
            else:
                tmp = check_match(epoch_parser,line)
                if tmp is not None:
                    max_epoch = tmp
                tmp = check_match(testing_rmse_parser,line)
                if tmp is not None:
                    last_testing_rmse = float(tmp)
                if last_testing_rmse < best_testing_rmse:
                    best_testing_rmse = last_testing_rmse

    if max_epoch is not None:
        params['max_epoch'] = max_epoch
    if last_testing_rmse != 999:
        params['last_testing_rmse'] = last_testing_rmse
    if best_testing_rmse != 999:
        params['best_testing_rmse'] = best_testing_rmse
    return params

process_notes_file = process_log_file = process_file

def merge_params(params1,params2):
    intersection = set(params1.keys()).intersection(set(params2.keys()))
    if len(intersection) > 0:
        raise Exception("params found both in params1/2: {}".format(
            {
                param: (
                    params1[param],
                    params2[param]
                )
                for param
                in intersection
            }
        ))
    ret = {}
    ret.update(params1)
    ret.update(params2)
    return ret

def params_to_str(params):
    ret = ""
    for k,v in params.items():
        ret+=k
        ret+=":"
        ret+=str(v)
        ret+=","
    return ret

def process_single_harvest(harvest_dir):
    """
    a harvest_dir will contain a *.log file and an optional notes.txt file
    """

    log_filenames = glob.glob(os.path.join(harvest_dir,"*.log"))
    if len(log_filenames) == 0:
        return None
    elif len(log_filenames) != 1:
        raise Exception("cannot process log file of harvest_dir={}, because there are {} : {}".format(
            harvest_dir,
            len(log_filenames),
            str(log_filenames)
        ))
    log_filename = log_filenames[0]
    params = process_log_file(log_filename)
    notes_filename = os.path.join(harvest_dir,"notes.txt")
    if os.path.isfile(notes_filename):
        notes_params = process_notes_file(notes_filename)
        params = merge_params(params,notes_params)
    params['harvest_dir'] = harvest_dir

    return params

def create_table(paramss):
    df = pd.DataFrame(paramss)
    return df
def process_multiple(args):
    paramss = []
    for arg in args:
        curr = process_single_arg(arg)
        if curr is not None:
            paramss.append(curr)
    df = create_table(paramss)
    print(df)

def process_single_arg(arg):
    if os.path.isfile(arg):
        params = process_file(arg)
    elif os.path.isdir(arg):
        params = process_single_harvest(arg)
    if params is None or len(params.keys()) == 0:
        return
    params = complete_default(params)
    return params

def main():
    if len(sys.argv) == 1: # 0 args
        tmp = glob.glob("./harvest_*")
    elif len(sys.argv) >= 2: # 1 arg
        tmp = sys.argv[1:]

    process_multiple(tmp)

if __name__ == "__main__":
    main()
