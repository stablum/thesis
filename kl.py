from theano import tensor as T
import theano
import numpy as np

def kl_normal_diagonal(mu1,sigma_diag1,mu2,sigma_diag2,dim):
    log_det1 = T.sum(T.log(sigma_diag1))
    log_det2 = T.sum(T.log(sigma_diag2))
    log_sigmas_term = log_det2 - log_det1
    inv_sigma_diag2 = 1/sigma_diag2
    mu_diff = mu2-mu1
    mu_diff_sq = mu_diff ** 2
    ret = 0.5 * (
        log_sigmas_term
        - dim
        + T.sum(inv_sigma_diag2 * sigma_diag1)
        + T.sum(inv_sigma_diag2 * mu_diff_sq)
    )
    return ret

def kl_normal_diagonal_vs_unit(self,mu1,sigma_diag1,dim):
    # KL divergence of a multivariate normal over a normal with 0 mean and I cov
    log_det1 = T.sum(T.log(sigma_diag1)) #sum log is better than log prod
    mu_diff = -mu1
    ret = 0.5 * (
        - log_det1
        #- dim
        + T.sum(sigma_diag1) # trace
        + T.sum(mu_diff**2) # mu^T mu
    )
    return ret
