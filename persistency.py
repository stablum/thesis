import os
import pickle
import shutil
import lzma

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

    os.mkdir(dir_name)
    def gen_path(filename):
        return os.path.join(dir_name,filename)
    def _open(path):
        return open(path,"w+")
    print('writing model parameters..')
    filename = gen_path("params.pickle.xz")
    os.remove(filename)
    with lzma.open(filename,"wb") as f:
        pickle.dump(model.params_for_persistency,f)
    print('writing parameters update symbols and algorithm metainfo..')
    filename = gen_path("params_updates_values.pickle.xz")
    os.remove(filename)
    with lzma.open(filename,"wb") as f:
        pickle.dump(model.params_updates_values,f)
    print('writing learning rate..')
    filename = gen_path("lr")
    os.remove(filename)
    with _open(filename) as f:
        f.write(str(lr))
    print('writing epoch..')
    filename = gen_path("epoch")
    os.remove(filename)
    with _open(filename) as f:
        f.write(str(epoch))
    print("state saved.")

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
