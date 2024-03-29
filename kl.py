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

def kl_normal_diagonal_vs_unit(mu1,sigma_diag1,dim):
    # KL divergence of a multivariate normal over a normal with 0 mean and I cov
    logs = T.log(sigma_diag1)
    logs.name = "logs"
    log_det1 = T.sum(logs) #sum log is better than log prod
    log_det1.name = "log_det1"
    mu_diff = -mu1
    mu_diff.name = "mu_diff"
    trace = T.sum(sigma_diag1) # trace
    trace.name = "trace"
    muTmu = T.sum(mu_diff**2) # mu^T mu
    muTmu.name = "muTmu"
    ret = 0.5 * (
        - log_det1
        - dim
        + trace
        + muTmu
    )
    ret.name="kl_normal_diagonal_vs_unit ret"
    return ret
