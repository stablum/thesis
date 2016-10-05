#!/usr/bin/env python

import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('A1Ratings.csv')
    dfm = df.mean()
    dfm.sort()
    dfg4 = (df >= 4).sum()
    df_isin = df.isin([1,2,3,4,5])
    df_fill = df_isin.sum()
    dfp = (dfg4/df_fill)
    dfp.sort()
    df_fill_sort = df_fill.copy()
    df_fill_sort.sort()

    sw = df.icol(1)>0
    df_im = df_isin.as_matrix()
    sw_m = sw.as_matrix()
    sw_m2 = np.expand_dims(sw_m,1)
    sw_and = df_im & sw_m2
    sw_and_sum = np.sum(sw_and,0) # #ratings from users that also rated sw
    #sw_and_perc = (sw_and_sum/df_fill)*100
    df_sw_sum = df_fill.copy()
    df_sw_sum[0:] = sum(sw)
    sw_and_perc = (sw_and_sum/df_sw_sum)*100
    sw_and_perc.sort()
    import ipdb;
    ipdb.set_trace()

if __name__=="__main__":
    main()
