#!/usr/bin/env python3

import numpy as np
import theano
import theano.tensor as T

# local imports

import config

class AugmentedType(object):

    def __finalize__(self,*args,**kwargs):
        level_up("__finalize__ AugmentedType")
        self.augm_data = {}
        level_down()

    def _typecheck_attrname(self,attrname):
        assert type(attrname) is str, \
            "attrname '%s' not str but %s"%(str(attrname),str(type(attrname)))
        assert len(attrname) > 0, \
            "length of attrname string is %d but should be > 0"%len(attrname)

    def augment(self, attrname, value):
        self._typecheck_attrname(attrname)
        self.augm_data[attrname] = value

    def get_augmented(self, attrname):
        self._typecheck_attrname(attrname)
        assert attrname in self.augm_data.keys(),\
            "attrname '%s' not in augm_data %s"%(
                str(attrname),
                str(self.augm_data)
            )
        return self.augm_data[attrname]

class AdamType(AugmentedType):

    def __finalize__(self,*args,**kwargs):
        level_up("__finalize__ AdamType")
        super().__finalize__(*args,**kwargs)
        level_down()

    @property
    def m(self):
        m = self.get_augmented('m')
        assert m.shape == self.shape, \
            "AdamType: m (%s) has different shape than self (%s)"%(
                m.shape,
                self.shape
            )
        return m

    @m.setter
    def m(self, value):
        assert value.shape == self.shape, \
            "AdamType: given value for m (%s) has different shape than self (%s)"%(
                value.shape,
                self.shape
            )
        return self.augment('m', value)

    @property
    def v(self):
        v = self.get_augmented('v')
        assert v.shape == self.shape, \
            "AdamType: v (%s) has different shape than self (%s)"%(
                v.shape,
                self.shape
            )
        return v

    @v.setter
    def v(self, value):
        assert value.shape == self.shape, \
            "AdamType: given value for v (%s) has different shape than self (%s)"%(
                value.shape,
                self.shape
            )
        return self.augment('v', value)

    @property
    def t(self):
        t = self.get_augmented('t')
        return t

    @t.setter
    def t(self, value):
        return self.augment('t', value)

    @property
    def beta1_t(self):
        beta1_t = self.get_augmented('beta1_t')
        return beta1_t

    @beta1_t.setter
    def beta1_t(self, value):
        return self.augment('beta1_t', value)

    @property
    def beta2_t(self):
        beta2_t = self.get_augmented('beta2_t')
        return beta2_t

    @beta2_t.setter
    def beta2_t(self, value):
        return self.augment('beta2_t', value)

class NumpyArray(np.ndarray):
    pass

class Int64Adam(np.int64, AdamType):
    def __finalize__(self,*args,**kwargs):
        AdamType.__finalize__(self,*args,**kwargs)

class Float64Adam(np.float64, AdamType):
    def __finalize__(self,*args,**kwargs):
        level_up("__finalize__ Float64Adam")
        AdamType.__finalize__(self,*args,**kwargs)
        level_down()

level = 0
def log(*args):
    return # <--!!!!
    s = " ".join(map(str,args))
    for x in range(level):
        print("    ",end="",flush=True)
    print(s,flush=True)

def level_up(s):
    global level
    log(s)
    level +=1

def level_down():
    global level
    level -=1
    log("end")

class NumpyArrayAdam(NumpyArray, AdamType):

    @staticmethod
    def random(shape):
        rr = np.random.random(shape)
        return rr.view(NumpyArrayAdam)

    def __array_finalize__(self, obj):
        level_up("__array_finalize__")
        AdamType.__finalize__(self)
        self.m = np.zeros(self.shape)
        self.v = np.zeros(self.shape)
        self.t = 1
        self.beta1_t = config.adam_beta1
        self.beta2_t = config.adam_beta2
        level_down()
        return self

    def __getitem__(self, sel): # WARNING: very slow!
        level_up("__getitem__")
        log("halllooooo! %s"%str(sel))
        _slice = NumpyArray.__getitem__(self,sel)
        log("sliced:%s type:%s"%(_slice,type(_slice)))
        if type(_slice) is np.int64:
            log("casting to Int64Adam")
            ret = Int64Adam(_slice)
            ret.__finalize__()
        elif type(_slice) is np.float64:
            log("casting to Float64Adam")
            ret = Float64Adam(_slice)
            ret.__finalize__()
        else:
            log("this is a proper numpy slice")
            ret = _slice.copy()
        log("after casting ret:%s type:%s"%(ret,type(ret)))
        #log(".m: %s type:%s"%(ret.m,type(ret.m)))
        # extract just the right slice of the 'm' and 'v' arrays
        #log("self.m %s"%self.m)
        log("self address:",self.__array_interface__['data'])
        log("ret address:",ret.__array_interface__['data'])
        #log("self.m address:",self.m.__array_interface__['data'])
        #log("ret.m address:",ret.m.__array_interface__['data'])
        #ret.__array_finalize__(None)
        ret.m = self.m.__getitem__(sel)
        ret.v = self.v.__getitem__(sel)
        log("self.m address BIS:",self.m.__array_interface__['data'])
        log("ret.m address BIS:",ret.m.__array_interface__['data'])
        log("ret address BIS:",ret.__array_interface__['data'])
        log("self.m BIS %s"%self.m)
        level_down()
        return ret

class TheanoVariableAdam(theano.Variable, AdamType):
    pass

class TheanoSharedVariableAdam(T.sharedvar.TensorSharedVariable, AdamType):
    def ctor(self, *args, **kwargs):
        self.__finalize__(*args,**kwargs)

    @staticmethod
    def random(name,shape):
        rr = np.random.random(shape)
        return TheanoSharedTensorAdam(name,rr,TheanoVariableAdam,True)
