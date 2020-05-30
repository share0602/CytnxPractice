from setting import *
import cytnx
from cytnx import cytnx_extension as cyx
import numpy as np
from numpy import linalg
import copy
# import os
path = '/Users/chenyuxue/PycharmProjects/kitaev_cytnx/examples/dmrg/'
# path = str(os.path.abspath(path)) +'/'
# print(os.path.abspath(path))
"""
References:https://www.tensors.net
"""

def get_H_psi(psi, L, M1, M2, R):
    ''' psi is Tensor, while L,M1,M2,R are CyTensor.
    Return: h|psi> (Tensor)'''
    psi = cytnx.from_numpy(psi)
    psi = cyx.CyTensor(psi,0)
    psi = psi.reshape(L.shape()[1], M1.shape()[2], M2.shape()[2], R.shape()[1])
    anet = cyx.Network(path+"Network/psi_L_M1_M2_R.net")
    anet.PutCyTensor("psi", psi);
    anet.PutCyTensor("L", L);
    anet.PutCyTensor("M1", M1);
    anet.PutCyTensor('M2', M2);
    anet.PutCyTensor('R', R);
    H_psi = anet.Launch(optimal=True).reshape(-1).get_block().numpy()
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
                    print('Krylov space = %d is large enough!'%krydim)
                    break
        # print('h_la = ', h_la)
        [energy, psi_basis] = linalg.eigh(h_la[:krydim,:krydim])
        psivec = q[:,:krydim] @ psi_basis[:,0]

    psivec = psivec / linalg.norm(psivec)
    gs_energy = energy[0]
    return psivec, gs_energy


def absorb_right(s, vt, A):
    anet = cyx.Network(path + "Network/s_vt_A.net")
    ## A[p+1] = s@vt@A[p+1]
    anet.PutCyTensor("s_diag", s);
    anet.PutCyTensor("vt", vt);
    anet.PutCyTensor("A", A);
    A_1 = anet.Launch(optimal=True)
    return A_1


def get_new_L(L, A, M, Aconj):
    anet = cyx.Network(path + "Network/L_A_M_Aconj.net")
    ## L[p+1] = L[p+1]*A[p]*A[p].Conj()*M
    anet.PutCyTensor("L", L);
    anet.PutCyTensor("A", A);
    anet.PutCyTensor("M", M);
    anet.PutCyTensor('A_Conj', Aconj);
    L_1 = anet.Launch(optimal=True)
    return L_1


def get_psi_from_left(A1, A2, s):
    anet = cyx.Network(path + "Network/A_A_s.net")
    ## psi = A[p-1]*A[p]*s[p+1]
    anet.PutCyTensor("A1", A1);
    anet.PutCyTensor("A2", A2);
    anet.PutCyTensor("s",s);
    psi_gs = anet.Launch(optimal=True)
    return psi_gs


def get_new_R(R, B, M, Bconj):
    anet = cyx.Network(path + "Network/R_B_M_Bconj.net")
    ## R[P] = R[p+1]*B[p+1]*B[p+1].conj()*M
    anet.PutCyTensor("R", R);
    anet.PutCyTensor("B", B);
    anet.PutCyTensor("M", M);
    anet.PutCyTensor('B_Conj', Bconj);
    R_1 = anet.Launch(optimal=True)
    return R_1


def get_psi_from_right(s, B1, B2):
    anet = cyx.Network(path + "Network/s_B_B.net")
    ## psi = s[p]*B[p]*B[p+1]
    anet.PutCyTensor("s", s);
    anet.PutCyTensor("B1", B1);
    anet.PutCyTensor("B2", B2);
    psi_gs = anet.Launch(optimal=True)
    return psi_gs
'''
########################################################################################################################
                                    ########## Main Function ##########
########################################################################################################################
'''
def dmrg_two_sites(A, ML, M, MR, dim_cut, numsweeps=10, dispon=2, updateon=True, maxit=2, krydim=4):
    '''
    :param A: list of initial CyTensor
    :param ML: Left boundary
    :param M: MPO, M.shape() = (D,D,d,d)
    :param MR: Right boundary
    :return: Ekeep, A, s_weight, B
    '''
    d = M.shape()[2]  # physical dimension
    Nsites = len(A) # A is a list
    L = [0]*Nsites; # Left boundary for each MPS
    L[0] = ML
    R = [0]*Nsites; # Right boundary for each MPS
    R[Nsites - 1] = MR
    '''
    ########## Warm up: Put A into left orthogonal form ##########
    '''
    for p in range(Nsites - 1):
        s_diag, A[p], vt = cyx.xlinalg.Svd(A[p])

        A[p+1] = absorb_right(s_diag, vt, A[p + 1])

        L[p+1] = get_new_L(L[p], A[p], M, A[p].Conj())
    ## Initialiaze s_weight
    s_weight = [0] * (Nsites + 1)
    ## Specify A[final] and s[final]
    dim_l = A[Nsites -1].shape()[0];
    dim_r = A[Nsites-1].shape()[2]; # =1
    A[Nsites-1] = A[Nsites-1].get_block().reshape(dim_l*d,dim_r) ## CyTensor -> Tensor
    # This is because A[Nsites-1].shape() = [dim_l*d,1]
    _, A[Nsites-1], _ = cytnx.linalg.Svd(A[Nsites-1])
    ##[1], [4,1], [1,1] = svd([4,1)
    ## Just to make A[final] left orthogonal and renorm s to 1
    A[Nsites-1] = cyx.CyTensor(A[Nsites-1].reshape(dim_l,d,dim_r), 2)

    s_weight[Nsites] = cyx.CyTensor([cyx.Bond(1),cyx.Bond(1)],rowrank = 1, is_diag=True)
    s_weight[Nsites].put_block(cytnx.ones(1));
    Ekeep = [] # Store the energy of each two sites
    B = [0]*Nsites
    '''
            ########## DMRG sweep begin ##########
    '''
    for k in range(1, numsweeps + 2):
        ##### final sweep is only for orthogonalization (disable updates)
        if k == numsweeps + 1:
            updateon = False
            dispon = 0
        print('-'*50,k)
        '''
        ########## Optimization sweep: right-to-left ##########
        '''
        for p in range(Nsites - 2, -1, -1):
            ##### two-site update
            dim_l = A[p].shape()[0];
            dim_r = A[p + 1].shape()[2];
            psi_gs = get_psi_from_left(A[p], A[p+1], s_weight[p+2])
            if updateon:
                ## put psi_gs to Tensor for Lanczos algorithm

                psi_gs = psi_gs.reshape(-1).get_block().numpy()
                # psi_gs2 = copy.deepcopy(psi_gs)
                # psi_gs2, Entemp2 = gs_Arnoldi_numpy(psi_gs2, get_H_psi, (L[p], M, M, R[p + 1]), maxit=maxit,
                #                               krydim=krydim)
                # print(psi_gs.shape)
                psi_gs, Entemp = gs_Lanczos_numpy(psi_gs, get_H_psi, (L[p], M, M, R[p + 1]), maxit=maxit,
                                               krydim=krydim)
                # print(Entemp2 - Entemp)
                Ekeep.append(Entemp)
                psi_gs = cytnx.from_numpy(psi_gs)
                psi_gs = cyx.CyTensor(psi_gs.reshape(dim_l, d, d, dim_r), 2)

            dim_new = min(dim_l*d, dim_r*d, dim_cut)
            s_weight[p+1], A[p], B[p+1] = cyx.xlinalg.Svd_truncate(psi_gs, dim_new)
            norm = s_weight[p+1].get_block().Norm().item()
            s_weight[p+1] = s_weight[p+1]/norm

            # ##### new block Hamiltonian

            R[p] = get_new_R(R[p+1], B[p+1], M, B[p+1].Conj())
            if dispon == 2:
                print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))

        ###### left boundary tensor
        Btemp = cyx.Contract(A[0],s_weight[1])
        dim_l = A[0].shape()[0]; ## dim_l = 1
        dim_r = A[0].shape()[2];
        Btemp = Btemp.get_block().reshape(dim_l, d*dim_r)

        _, _, B[0] = cytnx.linalg.Svd(Btemp)
        ##[1], [1,1], [1,4] = svd([1,4)
        ## Just to make A[final] left orthogonal and renorm s to 1
        B[0] = B[0].reshape(1,d,dim_r)
        B[0] = cyx.CyTensor(B[0], 1)
        s_weight[0] = cyx.CyTensor([cyx.Bond(1), cyx.Bond(1)], rowrank=1, is_diag=True)
        s_weight[0].put_block(cytnx.ones(1));

        '''
        ########## Optimization sweep: left-to-right ##########
        '''
        for p in range(Nsites - 1):
            ##### two-site update
            dim_l = s_weight[p].shape()[0];
            dim_r = B[p+1].shape()[2]

            psi_gs = get_psi_from_right(s_weight[p], B[p], B[p+1])
            if updateon:
                ## put psi_gs to Tensor for Lanczos algorithm
                psi_gs = psi_gs.reshape(-1).get_block().numpy()
                psi_gs, Entemp = gs_Lanczos_numpy(psi_gs, get_H_psi, (L[p], M, M, R[p + 1]), maxit=maxit,
                                             krydim=krydim)
                Ekeep.append(Entemp)
                psi_gs = cytnx.from_numpy(psi_gs)
                psi_gs = cyx.CyTensor(psi_gs.reshape(dim_l, d, d, dim_r), 2)

            dim_new = min(dim_l * d, dim_r * d, dim_cut)
            s_weight[p + 1], A[p], B[p + 1] = cyx.xlinalg.Svd_truncate(psi_gs, dim_new)
            norm = s_weight[p + 1].get_block().Norm().item()
            s_weight[p + 1] = s_weight[p + 1] / norm

            ##### new block Hamiltonian
            L[p+1] = get_new_L(L[p], A[p], M, A[p].Conj())

            ##### display energy
            if dispon == 2:
                print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))
        ###### right boundary tensor
        Atemp = cyx.Contract(B[Nsites-1], s_weight[Nsites-1])
        # Atemp.print_diagram()
        dim_l = B[Nsites-1].shape()[0];
        dim_r = B[Nsites-1].shape()[2]; # dim_r = 1
        Atemp = Atemp.get_block().reshape(dim_l*d, dim_r)
        _, A[Nsites-1], _ = cytnx.linalg.Svd(Atemp)
        ##[1], [4,1], [1,1] = svd([4,1)
        # print(A[Nsites-1].shape())
        A[Nsites-1] = A[Nsites-1].reshape(dim_l, d,1)
        A[Nsites-1] = cyx.CyTensor(A[Nsites-1], 2)
        s_weight[Nsites] = cyx.CyTensor([cyx.Bond(1), cyx.Bond(1)], rowrank=1, is_diag=True)
        s_weight[Nsites].put_block(cytnx.ones(1));

        if dispon == 1:
            print('Sweep: %d of %d, Energy: %.8f, Bond dim: %d' % (k, numsweeps, Ekeep[-1], dim_cut))

    return Ekeep, A, s_weight, B

if __name__ == '__main__':
    ##### Set bond dimensions and simulation options
    chi = 20;
    Nsites = 10;
    model = 'Ising' # Ising or XX
    OPTS_numsweeps = 6 # number of DMRG sweeps
    OPTS_dispon = 1 # level of output display
    OPTS_updateon = True # level of output display
    OPTS_maxit = 2 # iterations of Lanczos method
    OPTS_krydim = 4 # dimension of Krylov subspace

    '''
    ########## Initialiaze MPO (quantum XX model) and random MPS ##########
    '''
    d = 2
    s = 0.5
    sx = cytnx.physics.spin(0.5,'x')
    sy = cytnx.physics.spin(0.5,'y')
    sz = cytnx.physics.spin(0.5, 'y')
    sp = sx+1j*sy
    sm = sx-1j*sy
    eye = cytnx.eye(d).astype(cytnx.Type.ComplexDouble)
    if model == 'XX':
        W = cytnx.zeros([4, 4, d, d]).astype(cytnx.Type.ComplexDouble)
        W[0,0,:,:] = W[3,3,:,:] = eye
        W[0,1,:,:] = W[2,3,:,:] = 2**0.5*sp
        W[0,2,:,:] = W[1,3,:,:] = 2**0.5*sm
        WL = cytnx.zeros([4, 1, 1]).astype(cytnx.Type.ComplexDouble)
        WR = cytnx.zeros([4, 1, 1]).astype(cytnx.Type.ComplexDouble)
        WL[0, 0, 0] = 1.;
        WR[3, 0, 0] = 1.
    if model == 'Ising':
        W = cytnx.zeros([3, 3, d, d]).astype(cytnx.Type.ComplexDouble)
        W[0,0, :,:] = W[2,2,:,:] = eye
        W[0,1,:,:] = sx*2
        W[0,2,:,:] = -1.0*(sz*2) ## g
        W[1,2,:,:] = sx*2
        WL = cytnx.zeros([3, 1, 1]).astype(cytnx.Type.ComplexDouble)
        WR = cytnx.zeros([3, 1, 1]).astype(cytnx.Type.ComplexDouble)
        WL[0, 0, 0] = 1.;
        WR[2, 0, 0] = 1.


    W = cyx.CyTensor(W,0); WR = cyx.CyTensor(WR, 0); WL = cyx.CyTensor(WL, 0)
    M = [0]*Nsites
    M[0] = cytnx.random.normal([1, d, min(chi, d)], 0., 1.)
    # A[0] = np.random.rand(1,chid,min(chi,chid))

    for k in range(1,Nsites):
        dim1 = M[k-1].shape()[2]; dim2 = d;
        dim3 = min(min(chi, M[k-1].shape()[2] * d), d ** (Nsites - k - 1));
        M[k] = cytnx.random.normal([dim1, dim2, dim3],0.,1.)
    ## Transform A to
    M = [cyx.CyTensor(M[i], 2) for i in range(len(M))]
    En1, A, sWeight, B = dmrg_two_sites(M, WL, W, WR, chi, numsweeps = OPTS_numsweeps, dispon = OPTS_dispon,
                                        updateon = OPTS_updateon, maxit = OPTS_maxit, krydim = OPTS_krydim)
    exit()
    #### Increase bond dim and reconverge
    chi = 32;
    En2, A, sWeight, B = dmrg_two_sites(A, WL, M, WR, chi, numsweeps = OPTS_numsweeps, dispon = OPTS_dispon,
                                        updateon = OPTS_updateon, maxit = OPTS_maxit, krydim = OPTS_krydim)

    #### Compare with exact results (computed from free fermions)
    from numpy import linalg as LA
    import matplotlib.pyplot as plt
    H = np.diag(np.ones(Nsites-1),k=1) + np.diag(np.ones(Nsites-1),k=-1)
    D = LA.eigvalsh(H)
    EnExact = 2*sum(D[D < 0])

    ##### Plot results
    plt.figure(1)
    plt.yscale('log')
    plt.plot(range(len(En1)), En1 - EnExact, 'b', label="chi = 16")
    plt.plot(range(len(En2)), En2 - EnExact, 'r', label="chi = 32")
    plt.legend()
    plt.title('DMRG for XX model')
    plt.xlabel('Update Step')
    plt.ylabel('Ground Energy Error')
    plt.show()




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

def gs_Lanczos(psivec, A_funct, functArgs, maxit=2, krydim=4):
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
            q = col_vec[:, j + 1]
        # print(h_la)
        [energy, psi_columns_basis] = cytnx.linalg.Eigh(h_la)
        psivec = cytnx.linalg.Matmul(col_vec[:, :krydim], psi_columns_basis[:, 0].reshape(krydim, 1)).reshape(-1)

    norm = psivec.reshape(-1).Norm().item()
    psivec = psivec / norm
    gs_energy = energy[0].item()
    return psivec, gs_energy
'''

