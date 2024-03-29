import glob
import os
import sys
import re
import pandas as pd
import argparse
from tqdm import tqdm
tqdm.monitor_interval = 0

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision',15)

eventual_timestamp_regex = "[0-9 :]*"
parser = re.compile(eventual_timestamp_regex+"(.*):[ ]*(.*)")
epoch_parser = re.compile(eventual_timestamp_regex+"epoch[: ]*([0-9]*)")
testing_rmse_parser = re.compile(eventual_timestamp_regex+"testing RMSE[: ]*(.*)")
training_rmse_parser = re.compile(eventual_timestamp_regex+"training RMSE[: ]*(.*)")
harvest_parser = re.compile("[\./]*harvest_([a-z_]*[a-z]).*")
shorten_dict = {
    "learning rate":("lr",float),
    "learning_rate":("lr",float),
    "lr annealing T":("T",int),
    "update algorithm":("upd",str),
    "regularization_latent_kl":("r_KL",float),
    #"best_testing_rmse":("test_rmse",float),
    #"best_training_rmse":("train_rmse",float),
    #"minibatch_size":("mb_sz",int),
    #"n_hid_layers":("n_hid",int),
    #"regularization_lambda":("reg_la",float),
}

not_shortened_dict = dict([
    ("K",int),
    ("TK",int),
    ("hid_dim",int),
    ("chan_out_dim",int),
    ("optimizer",str),
    ("n_epochs",int),
    ("g_in",str),
    ("g_rij",str),
    ("g_hid",str),
    ("g_latent",str),
    ("dropout_p",float),
    ("input_dropout_p",float),
    ("regularization_lambda",float),
    ("regularization_latent_kl",float),
    ("minibatch_size",int),
    ("n_hid_layers",int),
    ("stochastic_prediction",str),
    ("regression_error_coef",str),
    ("regression_type",str),
    ("preprocessing_type",str),
    ("regularization_type",str),
    ("output_ratings_regularizer_amount",str),
])

additional_column_types = {
    'max_epoch':int,
    'best_testing_rmse':float,
    'best_training_rmse':float,
    'last_testing_rmse':float,
    'last_training_rmse':float,
}

all_column_types = {}
all_column_types.update(additional_column_types)
all_column_types.update(not_shortened_dict)
all_column_types.update(dict(shorten_dict.values()))

defaults = {
    'K':10,
    'hid_dim':100,
    'chan_out_dim':10,
}

def to_type(val,t):
    if t is int:
        return int(float(val))
    else:
        return t(val)

def shorten(name):
    if name in shorten_dict.keys():
        shortened,t = shorten_dict[name]
        return shortened,t
    else:
        t = not_shortened_dict[name]
        return name,t

def allowed(name):
    if name in shorten_dict.keys():
        return True
    if name in not_shortened_dict.keys():
        return True
    return False

def process_line(line):
    match = parser.match(line)
    if match is None:
        return None
    name,val = match.groups()[-2:]
    if not allowed(name):
        return None
    short_name,t = shorten(name)
    casted_val = to_type(val,t)
    return short_name,casted_val

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
    last_testing_rmse = 99999
    best_testing_rmse = 99999
    last_training_rmse = 99999
    best_training_rmse = 99999
    nan = False
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
                    if tmp == "nan":
                        nan = True
                if last_testing_rmse < best_testing_rmse:
                    best_testing_rmse = last_testing_rmse
                tmp = check_match(training_rmse_parser,line)
                if tmp is not None:
                    last_training_rmse = float(tmp)
                if last_training_rmse < best_training_rmse:
                    best_training_rmse = last_training_rmse

    if max_epoch is not None:
        params['max_epoch'] = max_epoch
    if last_testing_rmse != 99999:
        params['last_testing_rmse'] = last_testing_rmse
    if best_testing_rmse != 99999:
        params['best_testing_rmse'] = best_testing_rmse
    if last_training_rmse != 99999:
        params['last_training_rmse'] = last_training_rmse
    if best_training_rmse != 99999:
        params['best_training_rmse'] = best_training_rmse
    params['nan'] = str(nan)
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

    log_filenames = glob.glob(os.path.join(harvest_dir,"*rec*.log"))
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

def cast_column(df,name,t):
    if name in df.keys():
        if t in (float,int):
            df[name] = df[name].fillna(-1)
        df[name] = df[name].astype(t)
    return df

def cast_table(df):
    for name, t in all_column_types.items():
        df = cast_column(df,name,t)
    return df

def create_table(paramss,sortby=None,filterby=None,top=None):
    df = pd.DataFrame(paramss)
    df = cast_table(df)
    if sortby is not None:
        df = df.sort_values(by=[sortby])
    if filterby is not None:
        selector = df.harvest_dir.str.contains(filterby)
        df = df[selector]
    if top is not None:
        df = df.head(n=int(top))
    return df

def process_multiple(args):
    paramss = []
    for arg in tqdm(args):
        curr = process_single_arg(arg)
        if curr is not None:
            paramss.append(curr)
    return paramss

def process_single_arg(arg):
    if os.path.isfile(arg):
        params = process_file(arg)
    elif os.path.isdir(arg):
        params = process_single_harvest(arg)
    else:
        print("invalid arg in process_single_arg:",arg)
        import ipdb; ipdb.set_trace()
    if params is None or len(params.keys()) == 0:
        return
    params = complete_default(params)
    return params

def print_twod(df,first_name,second_name,quantity_name):
    first_list = getattr(df,first_name).tolist()
    second_list = getattr(df,second_name).tolist()
    twodf = pd.DataFrame(
        index=map(str,sorted(list(set(second_list)))), # pandas does not like float names
        columns=map(str,sorted(list(set(first_list)))) # index as string
    )
    for junk,row in df.iterrows():
        quantity = row[quantity_name]
        row_val = str(row[second_name])
        col_val = str(row[first_name])
        twodf.loc[row_val][col_val] = quantity
    print(twodf)

def select_columns(df,args):
    max_len=6
    #pd.set_option('display.max_colwidth',7)
    pd.set_option('expand_frame_repr', False)

    columns = args.c.split(',')
    # select only chosen columns
    df = df[columns]
    for c in columns:
        if c=="harvest_dir":
            # cannot truncate this information
            continue
        #import ipdb; ipdb.set_trace()
        # truncate column content
        col = df[c].astype('str').str[:max_len]
        # truncate column name
        del df[c]
        trunc = c[:max_len]
        if trunc in df.columns:
            # already there, let's use the last charachters
            trunc = c[-max_len:]
        df[trunc] = col
    return df

def main():
    parser = argparse.ArgumentParser(description='Logs summarizer.')
    parser.add_argument(
        'logs_or_dirs',
        type=str,
        nargs='*',
        help='log files or harvest dirs'
    )
    parser.add_argument(
        '-s',
        help='sort by'
    )
    parser.add_argument(
        '-f',
        help='filter filename'
    )
    parser.add_argument(
        '-t',
        help='top n entries'
    )
    parser.add_argument(
        '-T',
        action="store_true",
        help='top n entries'
    )
    parser.add_argument(
        '--twod',
        help="2 factor table"
    )
    parser.add_argument(
        '-c',
        help="select which columns to display, separated by commas"
    )
    args = parser.parse_args()
    if len(args.logs_or_dirs) == 0:
        tmp = glob.glob("./harvest_*")
    elif len(args.logs_or_dirs) >= 1:
        tmp = args.logs_or_dirs
    tmp2 = filter(lambda curr: args.f in curr, tmp)
    paramss = process_multiple(tmp2)
    df = create_table(paramss,sortby=args.s,filterby=args.f,top=args.t)
    #import ipdb; ipdb.set_trace()
    if args.c is not None:
        df = select_columns(df,args)
    if args.twod:
        factors = args.twod.split(',')
        print_twod(df,factors[0],factors[1],args.s)
    elif args.T:
        print(df.T)
    else:
        print(df)

if __name__ == "__main__":
    main()
