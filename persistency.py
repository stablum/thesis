import os
import pickle
import shutil
import lzma
import glob
import re

rename_pattern = "(.*)\.new"
rename_prog = re.compile(rename_pattern)

def _xzopen(path,*args,**kwargs):
    xzpath = path+".xz"
    if os.path.isfile(xzpath):
        open_func = lzma.open
        path = xzpath
    else:
        open_func = open
    return open_func(path,*args,**kwargs)

def save(model,lr,epoch, epoch_dir=False):
    print("saving state...")

    if epoch_dir:
        dir_name = "state_epoch_{0:05d}".format(epoch)
        assert not os.path.isdir(dir_name), "state at epoch {} already exists".format(epoch)
    else:
        dir_name = "state"

    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    def gen_path(filename):
        return os.path.join(dir_name,filename)
    def _open(filename,*args,**kwargs):
        if '.pickle.xz' in filename:
            open_func = lambda *args,**kwargs : lzma.open(*args,"wb",**kwargs)
        else:
            open_func = lambda *args,**kwargs : open(*args,"w+",**kwargs)
        path = gen_path(filename)
        if os.path.isfile(path):
            os.remove(path)
        # don't write on the actual path, copy new into actual paths later
        path = path+".new"
        print("opening file {}".format(path))
        return open_func(path)

    print('writing model parameters..')
    with _open("params.pickle.xz") as f:
        pickle.dump(model.params_for_persistency,f)
    print('writing parameters update symbols and algorithm metainfo..')
    with _open("params_updates_values.pickle.xz") as f:
        pickle.dump(model.params_updates_values,f)
    print('writing learning rate..')
    with _open("lr") as f:
        f.write(str(lr))
    print('writing epoch..')
    with _open("epoch") as f:
        f.write(str(epoch))
    print("state saved. Now renaming files..")
    for filename in glob.glob(os.path.join(dir_name,"*.new")):
        m = rename_prog.match(filename)
        dst = m.groups()[0]
        print('renaming {} into {}'.format(filename,dst))
        os.rename(filename,dst)
    print("persistency save completed.")

def load(dir_name,model):
    def gen_path(filename):
        return os.path.join(dir_name,filename)
    def _open(path):
        return open(path,"r")

    print("put params into model")
    with _xzopen(gen_path("params.pickle"),"rb") as f:
        model.params_for_persistency = pickle.load(f)
    print('loading parameters update symbols and algorithm metainfo..')
    with _xzopen(gen_path("params_updates_values.pickle"),"rb") as f:
        model.params_updates_values = pickle.load(f)
    print('loading learning rate..')
    with _open(gen_path("lr")) as f:
        lr = float(f.read())
    print('loading epoch number..')
    with _open(gen_path("epoch")) as f:
        epoch = int(f.read())

    return lr, epoch
