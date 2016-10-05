#!/usr/bin/env python3

import numpy as np
import augmented_types as at
import config
M = 4
V = at.NumpyArrayAdam.random((config.K,M)) * 0.05
print(V)
print(V.m)
print(V.v)
#foo = np.array([[1.1,2],[4,5]]).view(at.NumpyArrayAdam)
#foo.m *= np.array([[42,43],[44,45]])
#at.log("should be 42 %s"%foo.m)
#at.log("prima: %s"%str(foo.m.shape))
#zuppa = foo[1,:]
#at.log("dopo: %s"%str(foo.m.shape))
#at.log("zuppa:")
#at.log(zuppa,type(zuppa))
#at.log("dio:")
#at.log(foo.dtype)
#at.log("cane:")
#at.log(foo[1,:],type(foo[1,:]))
#print("foo[1,1].m: "+str(foo[1,:].m))
#print("foo[1,1].v: "+str(foo[1,:].v))
#
