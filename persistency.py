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
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)

    os.mkdir(dir_name)
    def gen_path(filename):
        return os.path.join(dir_name,filename)
    def _open(path):
        return open(path,"w+")
    print('writing model parameters..')
    with lzma.open(gen_path("params.pickle.xz"),"wb") as f:
        pickle.dump(model.params_for_persistency,f)
    print('writing parameters update symbols and algorithm metainfo..')
    with lzma.open(gen_path("params_updates_values.pickle.xz"),"wb") as f:
        pickle.dump(model.params_updates_values,f)
    print('writing learning rate..')
    with _open(gen_path("lr")) as f:
        f.write(str(lr))
    print('writing epoch..')
    with _open(gen_path("epoch")) as f:
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
