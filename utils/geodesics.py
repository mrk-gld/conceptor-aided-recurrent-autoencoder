import jax.numpy as np
from scipy.linalg import expm, logm, sqrtm

def np_logm(M):
    return np.from_numpy(logm(M.detach().numpy())).float()

def np_expm(M):
    return np.from_numpy(expm(M.detach().numpy())).float()

def np_sqrtm(M):
    return np.from_numpy(sqrtm(M.detach().numpy())).float()

def square_rootm(M):
    Q, S, _ = np.linalg.svd(M)
    return Q @ np.diag(np.sqrt(S)) @ Q.T


def naive_linear_interpolation(T1,T2,L):
    return (1-L)*T1 + L*T2

def shape_and_orientation_rotation_metric(T1,T2,L):

    Q1, lamda1, _ = np.linalg.svd(T1)
    Q2, lamda2, _ = np.linalg.svd(T2)

    lamda1 = np.diag(lamda1)
    lamda2 = np.diag(lamda2)

    gamma_r_L = Q1 @ expm(L * logm(np.linalg.pinv(Q1)@Q2))
    gamma_s_L = lamda1 @ expm(L * logm(np.linalg.pinv(lamda1)@lamda2))

    
    Gamma_L = gamma_r_L @ gamma_s_L @ np.linalg.pinv(gamma_r_L)

    return np.real(Gamma_L)

def log_eucledian_metric(T1,T2,L):

    Gamma_L = expm((1-L) * logm(T1) + (L) * logm(T2))

    return np.real(Gamma_L)

def affine_invariant_metric(T1,T2,L):
    arg = np.linalg.pinv(T1) @ T2
    Gamma_AI = T1 @ expm(L * logm(arg))

    return np.real(Gamma_AI)

def difference_SAO_matrix(C1,C2):
    try:
        Q1, lamda1, _ = np.linalg.svd(C1)
        Q2, lamda2, _ = np.linalg.svd(C2)

        lamda1 = np.diag(lamda1)
        lamda2 = np.diag(lamda2)
        
        def dSO_3(Q1,Q2):
            return 2**(-0.5) * logm(Q2@Q1.T)
        
        def R_p_3(lamda1,lamda2):
            return 2**(-0.5) * logm(lamda2 @ np.inverse(lamda1))
        
        return np.sqrt(dSO_3(Q1,Q2)**2 + R_p_3(lamda1,lamda2)**2)
    except:
        return np.array([0])