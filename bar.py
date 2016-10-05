#!/usr/bin/env python

import pandas as pd
import numpy as np

def main():
    df_whole = pd.read_excel('a2.xls').fillna(0)
    df_m = df_whole.icol(range(10)).irow(range(20))
    m = df_m.as_matrix()
    u = []
    uv = []
    tf = []
    for i in range(2):
        xls_i = 13 + i
        curr = df_whole.icol( xls_i ).irow(range(20))
        u.append(curr)
    um = np.array(u).T

    def ps(_m):
        profiles = np.dot(_m.T,um)
        scores = np.dot(_m,profiles)
        scores_df = pd.DataFrame(scores)
        scores_df = scores_df.set_index(df_m.index)
        print(scores_df.sort(0))
        print(scores_df.sort(1))
        disliking = (scores_df < 0).sum()
        print(disliking)
    print("part 1")
    ps(m)

    print("part 2")
    m2 = m.copy()
    sums = np.expand_dims(m2.sum(1),1)
    m_norm = m2 / np.sqrt(sums)

    ps(m_norm)

    print("part 3")
    idf = np.expand_dims(1/m.sum(0),0)
    profiles = np.dot(m_norm.T, um)
    scores = np.dot( m_norm * idf , profiles)

    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.set_index(df_m.index)
    print(scores_df.sort(0))
    print(scores_df.sort(1))
    disliking = (scores_df < 0).sum()
    print(disliking)
    import ipdb;
    ipdb.set_trace()

if __name__=="__main__":
    main()
