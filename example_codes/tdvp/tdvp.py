import numpy as np
from numpy import linalg
from setting import *
import cytnx
from cytnx import cytnx_extension as cyx
"""
References:https://www.tensors.net
"""
# path = ''

def one_site_H_psi(psi, L, W,R):
    ''' psi is Tensor, while L,M1,M2,R are CyTensor.
    Return: h|psi> (Tensor)'''
    psi = cytnx.from_numpy(psi)
    # print(psi.shape())
    # print(L.shape(), W.shape(), R.shape())
    psi = psi.reshape(L.shape()[1], W.shape()[2], R.shape()[1])
    psi = cyx.CyTensor(psi,2)
    anet = cyx.Network("Network/psi_L_W_R.net")
    anet.PutCyTensor("psi", psi);
    anet.PutCyTensor("L", L);
    anet.PutCyTensor("W", W);
    anet.PutCyTensor('R', R);
    H_psi = anet.Launch(optimal=True).get_block().reshape(-1).numpy()
    return H_psi

def zero_site_H_psi(psi,L,R):
    psi = cytnx.from_numpy(psi)
    psi = psi.reshape(L.shape()[1], R.shape()[1])
    psi = cyx.CyTensor(psi, 1)
    anet = cyx.Network("Network/C_L_R.net")
    anet.PutCyTensor("C", psi);
    anet.PutCyTensor("L", L);
    anet.PutCyTensor('R', R);
    H_psi = anet.Launch(optimal=True).get_block().reshape(-1).numpy()
    return H_psi

def gs_Arnoldi_numpy(psivec, A_funct, functArgs, maxit=2, krydim=10):
    q = np.zeros([len(psivec), krydim+1], dtype=complex)
    h_ar = np.zeros([krydim+1, krydim], dtype=complex)
    for it in range(maxit):
        q[:,0] = psivec/linalg.norm(psivec)
        for j in range(krydim):
            u = A_funct(q[:,j], *functArgs)
            for i in range(j+1):
                h_ar[i, j] = np.conj(q[:, i]) @ u
                u = u - h_ar[i,j]*q[:,i]
            h_ar[j+1,j]  = linalg.norm(u)
            eps = 1e-12
            if linalg.norm(u) > eps:
                q[:,j+1] = u/linalg.norm(u)
        ## Arnoldi done
        [energy, psi_basis] = linalg.eigh(h_ar[:krydim,:])
        # print('h_ar = ', h_ar)
        psivec = q[:, :krydim] @ psi_basis[:, 0] ## basis transformation
    psivec = psivec / linalg.norm(psivec)
    gs_energy = energy[0]
    return psivec, gs_energy

def gs_Lanczos_numpy(psivec, A_funct, functArgs, maxit=2, krydim=10):
    q = np.zeros([len(psivec), krydim + 1], dtype=complex)
    h_la = np.zeros([krydim , krydim], dtype=complex)
    for it in range(maxit):
        q[:, 0] = psivec/linalg.norm(psivec)
        for j in range(krydim):
            u = A_funct(q[:,j], *functArgs)
            h_la[j,j] = np.conj(q[:,j])@ u
            if j == 0:
                u = u - h_la[j,j]*q[:, j]
            else:
                u = u - h_la[j,j]*q[:, j]-h_la[j-1,j]*q[:,j-1]
            if j < krydim-1:
                h_la[j,j+1] = h_la[j+1,j] = linalg.norm(u)
                eps = 1e-12
                if linalg.norm(u) > eps:
                    q[:, j + 1] = u/linalg.norm(u)
                else:
                    krydim = j + 1
                    print('Krylov space = %spaced is large enough!'%krydim)
                    break
        # print('h_la = ', h_la)
        [energy, psi_basis] = linalg.eigh(h_la[:krydim,:krydim])
        psivec = q[:,:krydim] @ psi_basis[:,0]

    psivec = psivec / linalg.norm(psivec)
    gs_energy = energy[0]
    return psivec, gs_energy

def exp_Lanczos_numpy(psivec, A_funct, functArgs, dt,maxit=2, krydim=10):
    q = np.zeros([len(psivec), krydim + 1], dtype=complex)
    h_la = np.zeros([krydim, krydim], dtype=complex)
    for it in range(maxit):
        q[:, 0] = psivec / linalg.norm(psivec)
        for j in range(krydim):
            u = A_funct(q[:, j], *functArgs)
            h_la[j, j] = np.conj(q[:, j]) @ u
            if j == 0:
                u = u - h_la[j, j] * q[:, j]
            else:
                u = u - h_la[j, j] * q[:, j] - h_la[j - 1, j] * q[:, j - 1]
            if j < krydim - 1:
                h_la[j, j + 1] = h_la[j + 1, j] = linalg.norm(u)
                eps = 1e-12
                if linalg.norm(u) > eps:
                    q[:, j + 1] = u / linalg.norm(u)
                else:
                    krydim = j + 1
                    # print('Krylov space = %d is large enough!' % krydim)
                    break
        # print('h_la = ', h_la)
        [energy, psi_basis] = linalg.eigh(h_la[:krydim, :krydim])
        psivec = q[:,:krydim] @ psi_basis[:,0]

    V = q[:, :krydim] @ psi_basis
    expE = np.diag(np.exp(-1j * dt / 2 * energy))
    psi_new = (V@expE@np.conj(V.T)) @ q[:,0]
    psi_new = psi_new/ linalg.norm(psi_new)
    return psi_new, energy


def get_new_L(L, A, W, Aconj):
    anet = cyx.Network("Network/L_A_W_Aconj.net")
    ## L[p+1] = L[p+1]*A[p]*A[p].Conj()*M
    anet.PutCyTensor("L", L);
    anet.PutCyTensor("A", A);
    anet.PutCyTensor("W", W);
    anet.PutCyTensor('A_Conj', Aconj);
    L_1 = anet.Launch(optimal=True)
    return L_1


def get_new_R(R, B, W, B_Conj):
    anet = cyx.Network("Network/R_B_W_Bconj.net")
    ## R[P] = R[p+1]*B[p+1]*B[p+1].conj()*M
    anet.PutCyTensor("R", R);
    anet.PutCyTensor("B", B);
    anet.PutCyTensor("W", W);
    anet.PutCyTensor('B_Conj', B_Conj);
    R = anet.Launch(optimal=True)
    return R


def absorb_remain(M, utemp, stemp):
    anet = cyx.Network("Network/M_u_s.net")
    anet.PutCyTensor('M', M)
    anet.PutCyTensor('u', utemp)
    anet.PutCyTensor('s', stemp)
    M_1 = anet.Launch(optimal=True)
    return M_1

'''
########################################################################################################################
                                        ########## Main Function ##########
########################################################################################################################
'''
def tdvp_one_site(M, WL, W, WR, dt, numsweeps=10, dispon=2, updateon=True, maxit=1, krydim=10):
    d = M[0].shape()[2]  # physical dimension
    Nsites = len(M) # M is a list
    A = [None] * Nsites # left orthogonal tensor
    B = [None] * Nsites # right orthogonal tensor
    # EE = [[None]*Nsites]*(2*numsweeps+1)
    # EE = np.zeros([Nsites-2, 2*numsweeps+1])
    EE_all_time = []
    EE = np.zeros([Nsites-1])
    L = [None]*Nsites; # Left boundary for each MPS
    L[0] = WL
    R = [None]*Nsites; # Right boundary for each MPS
    R[Nsites - 1] = WR
    '''
        ########## Warm up: Put M into right orthogonal form ##########
    '''
    for p in range(Nsites - 1, 0, -1):
        stemp, utemp, B[p] = cyx.xlinalg.Svd(M[p])
        M[p-1] = absorb_remain(M[p-1], utemp, stemp)
        R[p-1] = get_new_R(R[p],B[p],W,B[p].Conj())
    dim_l = 1; dim_r = M[0].shape()[2]
    Mtemp = M[0].get_block().reshape(dim_l, d*dim_r)
    _,_,B[0] = cytnx.linalg.Svd(Mtemp)
    B[0] = B[0].reshape(dim_l, d, dim_r)
    B[0] = cyx.CyTensor(B[0], 1)
    Ac = B[0] ## One-site center to do update
    '''
            ########## TDVP sweep begin ##########
    '''
    for k in range(1, numsweeps + 2):
        ##### final sweep is only for orthogonalization (disable updates)
        if k == numsweeps + 1:
            updateon = False
            dispon = 0
        print('-'*50,k)
        '''
        ########## Optimization sweep: left-to-right ##########
        '''
        for p in range(Nsites - 1):
            ##### two-site update
            # print('p = ', p)
            if updateon:
                ## put psi_gs to Tensor for Lanczos algorithm
                dim_l = Ac.shape()[0]; # Used to reshape Ac_new later
                dim_r = Ac.shape()[2]
                ############ Numpy Begin: Update one site
                Ac_old = Ac.get_block().reshape(-1).numpy()
                Ac_new, E = exp_Lanczos_numpy(Ac_old, one_site_H_psi, (L[p],W,R[p]),dt, maxit=maxit, krydim=krydim)
                Ac_new = cytnx.from_numpy(Ac_new)
                ############ Numpy End
                Ac_new = cyx.CyTensor(Ac_new.reshape(dim_l,d, dim_r), 2)
                stemp, A[p], vTtemp = cyx.xlinalg.Svd(Ac_new)
                ############ Entanglement entropy
                s_np = stemp.get_block().numpy()
                s_np[s_np <1.e-20] = 0.
                assert abs(np.linalg.norm(s_np) - 1.) < 1.e-14
                S2 = s_np**2
                EE[p] = -np.sum(S2*np.log(S2))
                # print(EE[2*k-2,p])

                C_old = cyx.Contract(stemp,vTtemp)
                dim_l = C_old.shape()[0]; dim_r = C_old.shape()[1]
                L[p+1] = get_new_L(L[p], A[p], W,A[p].Conj())
                ############ Numpy Begin: Update zero site
                C_old = C_old.get_block().reshape(-1).numpy()
                C_new, E = exp_Lanczos_numpy(C_old, zero_site_H_psi, (L[p+1],R[p]) ,-dt)
                C_new = cytnx.from_numpy(C_new)
                ############ Numpy End
                C_new = C_new.reshape(dim_l, dim_r)
                C_new = cyx.CyTensor(C_new, 1)
                C_new.set_labels([0,-2])
                if p == Nsites-2:
                    B[Nsites-1].set_labels([-2,2,3])
                Ac = cyx.Contract(C_new, B[p+1])
            ##### display energy
            if dispon == 2:
                print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, E[0]))
        EE_all_time.append(EE.copy())
        # print(EE_all_time)
        # C_new.print_diagram()
        # B[p+1].print_diagram()
        # Ac.print_diagram()
        dim_l = Ac.shape()[0];
        dim_r = Ac.shape()[2]
        # print(Ac.shape())
        Ac_old = Ac.get_block().reshape(-1).numpy()
        Ac_new, E = exp_Lanczos_numpy(Ac_old, one_site_H_psi, (L[Nsites-1], W, R[Nsites-1]),\
                                      0, maxit=maxit, krydim=krydim)
        Ac_new = cytnx.from_numpy(Ac_new)
        Ac = cyx.CyTensor(Ac_new.reshape(dim_l, d, dim_r), 1)

        for p in range(Nsites-2,-1,-1):
            if updateon:
                stemp, utemp, B[p+1] = cyx.xlinalg.Svd(Ac)
                ############ Entanglement entropy
                s_np = stemp.get_block().numpy()
                s_np[s_np < 1.e-20] = 0.
                assert abs(np.linalg.norm(s_np) - 1.) < 1.e-14
                S2 = s_np ** 2
                EE[p] = -np.sum(S2 * np.log(S2))
                C_old = cyx.Contract(stemp, utemp)
                dim_l = C_old.shape()[0];
                dim_r = C_old.shape()[1]

                R[p] = get_new_R(R[p+1],B[p+1],W,B[p+1].Conj())
                ############ Numpy Begin: Update zero site
                C_old = C_old.get_block().reshape(-1).numpy()
                C_new, E = exp_Lanczos_numpy(C_old, zero_site_H_psi, (L[p + 1], R[p]), -dt)
                C_new = cytnx.from_numpy(C_new)
                ############ Numpy Enf
                C_new = C_new.reshape(dim_l, dim_r)
                C_new = cyx.CyTensor(C_new, 1)

                C_new.set_labels([-1, 2])

                Ac_old = cyx.Contract(A[p],C_new).get_block()
                dim_l = Ac_old.shape()[0];
                dim_r = Ac_old.shape()[2]
                ############ Numpy Begin: Update one site
                Ac_old = Ac_old.reshape(-1).numpy()
                Ac_new, E = exp_Lanczos_numpy(Ac_old, one_site_H_psi, (L[p], W, R[p]),\
                                              dt, maxit=maxit, krydim=krydim)
                Ac_new = cytnx.from_numpy(Ac_new)
                ############ Numpy End
                Ac_new = cyx.CyTensor(Ac_new.reshape(dim_l, d, dim_r), 1)
                Ac =  Ac_new
        if dispon == 1:
            print('Sweep: %d of %d, Energy: %.8f, Bond dim: %d' % (k, numsweeps, E[0], chi))
        EE_all_time.append(EE.copy())
        # EE = 0
        # EE_all_time.append(EE)
        # print(EE_all_time)
        # exit()

    return EE_all_time, A, B

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../')
    from dmrg.dmrg_two_sites import dmrg_two_sites
    ##### XX model
    ##### Set bond dimensions and simulation options
    chi = 20;
    Nsites = 10;
    model = 'Ising'

    OPTS_numsweeps = 3 # number of DMRG sweeps
    OPTS_dispon = 1 # level of output display
    OPTS_updateon = True # level of output display
    OPTS_maxit = 1 # iterations of Lanczos method
    OPTS_krydim = 5 # dimension of Krylov subspace

    '''
    ########## Initialiaze MPO (quantum XX model) and random MPS ##########
    '''
    d = 2
    s = 0.5
    sx = cytnx.physics.spin(0.5, 'x')
    sy = cytnx.physics.spin(0.5, 'y')
    sz = cytnx.physics.spin(0.5, 'y')
    sp = sx + 1j * sy
    sm = sx - 1j * sy
    eye = cytnx.eye(d).astype(cytnx.Type.ComplexDouble)
    if model == 'XX':
        W = cytnx.zeros([4, 4, d, d]).astype(cytnx.Type.ComplexDouble)
        W[0, 0, :, :] = W[3, 3, :, :] = eye
        W[0, 1, :, :] = W[2, 3, :, :] = 2 ** 0.5 * sp
        W[0, 2, :, :] = W[1, 3, :, :] = 2 ** 0.5 * sm
        WL = cytnx.zeros([4, 1, 1]).astype(cytnx.Type.ComplexDouble)
        WR = cytnx.zeros([4, 1, 1]).astype(cytnx.Type.ComplexDouble)
        WL[0, 0, 0] = 1.;
        WR[3, 0, 0] = 1.
    if model == 'Ising':
        W = cytnx.zeros([3, 3, d, d]).astype(cytnx.Type.ComplexDouble)
        W[0, 0, :, :] = W[2, 2, :, :] = eye
        W[0, 1, :, :] = sx * 2
        W[0, 2, :, :] = -1.0 * (sz * 2)  ## g
        W[1, 2, :, :] = sx * 2
        WL = cytnx.zeros([3, 1, 1]).astype(cytnx.Type.ComplexDouble)
        WR = cytnx.zeros([3, 1, 1]).astype(cytnx.Type.ComplexDouble)
        WL[0, 0, 0] = 1.;
        WR[2, 0, 0] = 1.
    W = cyx.CyTensor(W, 0);
    WR = cyx.CyTensor(WR, 0);
    WL = cyx.CyTensor(WL, 0)
    M = [0] * Nsites
    M[0] = cytnx.random.normal([1, d, min(chi, d)], 0., 1.)
    for k in range(1,Nsites):
        dim1 = M[k - 1].shape()[2]; dim2 = d;
        dim3 = min(min(chi, M[k - 1].shape()[2] * d), d ** (Nsites - k - 1));
        M[k] = cytnx.random.normal([dim1, dim2, dim3], 0., 1.)
    ## Transform M to CyTensor
    # M = [cyx.CyTensor(M[i], 1) for i in range(len(M))]
    M = [cyx.CyTensor(M[i], 2) for i in range(len(M))]

    En1, A, sWeight, B = dmrg_two_sites(M, WL, W, WR, chi, numsweeps=OPTS_numsweeps, dispon=OPTS_dispon,
                                        updateon=OPTS_updateon, maxit=OPTS_maxit, krydim=OPTS_krydim)
    # print('-'*50)
    print('TDVP begin!')
    # print('-' * 50)
    EE, A,B = tdvp_one_site(B, WL, W, WR, 0.01, numsweeps = 10, dispon = OPTS_dispon,
                                       updateon = OPTS_updateon, maxit = OPTS_maxit, krydim = 30)
    # print(EE)
    # for i in range(len(EE)):
    #     print(EE[i])
    exit()





'''
def gs_Arnoldi(psivec, A_funct, functArgs, maxit=2, krydim=4):
    col_vec = cytnx.zeros([len(psivec), krydim +1]).astype(cytnx.Type.ComplexDouble)
    h_ar = cytnx.zeros([krydim+1, krydim]).astype(cytnx.Type.ComplexDouble)
    for it in range(maxit):
        norm = max(psivec.reshape(-1).Norm().item(), 1e-16)
        q = psivec/norm
        col_vec[:, 0] = q
        for j in range(krydim):
            u = A_funct(q, *functArgs)
            for i in range(j+1):
                h_ar[i,j] = cytnx.linalg.Dot(col_vec[:,i].Conj(), u)
                u = u - h_ar[i,j].item()*col_vec[:,i]
            h_ar[j+1,j]  = u.Norm()
            eps = 1e-12
            if h_ar[j+1,j].Norm().item() > eps:
                q = u/h_ar[j+1,j].item()
                col_vec[:, j+1] = q
        # print(h_ar[:krydim,:])
        [energy, psi_columns_basis] = cytnx.linalg.Eigh(h_ar[:krydim,:])
        psivec = cytnx.linalg.Matmul(col_vec[:, :krydim], psi_columns_basis[:, 0].reshape(krydim, 1)).reshape(-1)

    norm = psivec.reshape(-1).Norm().item()
    psivec = psivec / norm
    gs_energy = energy[0].item()
    return psivec, gs_energy
    
    
def gs_Lanczos(psivec, A_funct, functArgs, maxit=2, krydim=10):
    col_vec = cytnx.zeros([len(psivec), krydim +1]).astype(cytnx.Type.ComplexDouble)
    h_la = cytnx.zeros([krydim, krydim]).astype(cytnx.Type.ComplexDouble)
    for it in range(maxit):
        norm = max(psivec.reshape(-1).Norm().item(), 1e-16)
        q = psivec/norm
        col_vec[:, 0] = q
        for j in range(krydim):
            u = A_funct(q, *functArgs)
            h_la[j,j] = cytnx.linalg.Dot(col_vec[:,j].Conj(), u)
            if j == 0:
                u = u - h_la[j, j].item()*col_vec[:, j]
            else:
                u = u - h_la[j,j].item()*col_vec[:, j]-h_la[j-1,j].item()*col_vec[:,j-1]
            if j < krydim-1:
                h_la[j,j+1] = h_la[j+1,j] = u.Norm()
                eps = 1e-12
                if h_la[j+1,j].Norm().item() > eps:
                    col_vec[:, j + 1] = u/h_la[j+1,j].item()
                else:
                    krydim = j + 1
                    print('Krylov space = %d is large enough!'%krydim)
                    # print(krydim)
                    break
            q = col_vec[:, j + 1]

        [energy, psi_columns_basis] = cytnx.linalg.Eigh(h_la[:krydim,:krydim].reshape(krydim,krydim))

        psivec = cytnx.linalg.Matmul(col_vec[:, :krydim].reshape(len(psivec),krydim),\
                                     psi_columns_basis[:, 0].reshape(krydim, 1)).reshape(-1)

    norm = psivec.reshape(-1).Norm().item()
    psivec = psivec / norm
    gs_energy = energy[0].item()
    return psivec, gs_energy
    
def eig_Lanczos(psivec, A_funct, functArgs, dt, maxit=2,krydim=10):
    col_vec = cytnx.zeros([len(psivec), krydim +1]).astype(cytnx.Type.ComplexDouble)
    h_la = cytnx.zeros([krydim, krydim]).astype(cytnx.Type.ComplexDouble)
    for it in range(maxit):
        norm = max(psivec.reshape(-1).Norm().item(), 1e-16)
        q = psivec/norm
        col_vec[:, 0] = q
        for j in range(krydim):
            u = A_funct(q, *functArgs)
            h_la[j,j] = cytnx.linalg.Dot(col_vec[:,j].Conj(), u)
            if j == 0:
                u = u - h_la[j, j].item()*col_vec[:, j]
            else:
                u = u - h_la[j,j].item()*col_vec[:, j]-h_la[j-1,j].item()*col_vec[:,j-1]
            if j < krydim-1:
                h_la[j,j+1] = h_la[j+1,j] = u.Norm()
                eps = 1e-12
                if h_la[j+1,j].Norm().item() > eps:
                    col_vec[:, j + 1] = u/h_la[j+1,j].item()
                else:
                    krydim = j + 1
                    print('Krylov space = %d is large enough!'%krydim)
                    break
            q = col_vec[:, j + 1]
        [energy, psi_columns_basis] = cytnx.linalg.Eigh(h_la[:krydim,:krydim].reshape(krydim,krydim))
        psivec = cytnx.linalg.Matmul(col_vec[:, :krydim].reshape(len(psivec),krydim),\
                                     psi_columns_basis[:, 0].reshape(krydim, 1)).reshape(-1)
    [E, V] = cytnx.linalg.Eigh(h_la[:krydim, :krydim].reshape(krydim,krydim))
    V = cytnx.linalg.Matmul(col_vec[:, :krydim].reshape(len(psivec),krydim), V)
    expE = cytnx.linalg.Diag(cytnx.linalg.Exp(-1j * dt / 2 * E))
    evolve = cytnx.linalg.Matmul(V, expE)
    evolve = cytnx.linalg.Matmul(evolve, V.permute(1, 0).contiguous().Conj())
    psi_new = cytnx.linalg.Matmul(evolve, col_vec[:, 0].reshape(len(psivec), 1))
    norm = psi_new.reshape(-1).Norm().item()
    psi_new = psi_new / norm
    return psi_new, E, energy[0].item()
'''