import numpy as np
from setting import *
import cytnx
from cytnx import cytnx_extension as cyx
"""
References:https://www.tensors.net
"""

def get_H_psi(psi, L, M1, M2, R):
    ''' psi is Tensor, while L,M1,M2,R are CyTensor.
    Return: h|psi> (Tensor)'''
    psi = cyx.CyTensor(psi,0)
    psi = psi.reshape(L.shape()[1], M1.shape()[2], M2.shape()[2], R.shape()[1])
    anet = cyx.Network("Network/psi_L_M1_M2_R.net")
    anet.PutCyTensor("psi", psi);
    anet.PutCyTensor("L", L);
    anet.PutCyTensor("M1", M1);
    anet.PutCyTensor('M2', M2);
    anet.PutCyTensor('R', R);
    H_psi = anet.Launch(optimal=True).reshape(-1).get_block()
    return H_psi

def eig_Lanczos(psivec, linFunct, functArgs, maxit=2, krydim=4):
    """ Lanczos method for finding smallest algebraic eigenvector of linear \
    operator defined as a function"""
    psi_columns = cytnx.zeros([len(psivec), krydim + 1]).astype(cytnx.Type.ComplexDouble)
    krylov_matrix = cytnx.zeros([krydim, krydim]).astype(cytnx.Type.ComplexDouble)
    for ik in range(maxit):
        norm = max(psivec.reshape(-1).Norm().item(), 1e-16)
        # print(psi_columns[:, 0].shape())
        psi_columns[:, 0] = psivec / norm
        for ip in range(1, krydim + 1):

            psi_columns[:, ip] = linFunct(psi_columns[:, ip - 1], *functArgs)
            for ig in range(ip):
                krylov_matrix[ip - 1, ig] = cytnx.linalg.Dot(psi_columns[:, ip], psi_columns[:, ig])
                krylov_matrix[ig, ip - 1] = krylov_matrix[ip - 1, ig].Conj()

            for ig in range(ip):
                vp = psi_columns[:, ip];
                vg = psi_columns[:, ig]
                vp = vp - cytnx.linalg.Dot(vg, vp).item() * vg;
                norm = max(vp.Norm().item(), 1e-16)
                psi_columns[:, ip] = vp / norm  ## only access set() once!!

        [energy, psi_columns_basis] = cytnx.linalg.Eigh(krylov_matrix)
        psivec = cytnx.linalg.Matmul(psi_columns[:, :krydim],psi_columns_basis[:, 0].reshape(krydim,1)).reshape(-1)

    norm = psivec.reshape(-1).Norm().item()
    psivec = psivec / norm
    gs_energy = energy[0].item()
    return psivec, gs_energy

'''
########## Main Function ##########
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
        anet = cyx.Network("Network/s_vt_A.net")
        ## A[p+1] = s@vt@A[p+1]
        anet.PutCyTensor("s_diag", s_diag);
        anet.PutCyTensor("vt", vt);
        anet.PutCyTensor("A", A[p+1]);
        A[p+1] = anet.Launch(optimal = True)
        anet = cyx.Network("Network/L_A_M_Aconj.net")
        ## L[p+1] = L[p+1]*A[p]*A[p].Conj()*M
        anet.PutCyTensor("L", L[p]);
        anet.PutCyTensor("A", A[p]);
        anet.PutCyTensor("M", M);
        anet.PutCyTensor('A_Conj', A[p].Conj());
        L[p + 1] = anet.Launch(optimal=True)

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
            anet = cyx.Network("Network/A_A_s.net")
            ## psi = A[p-1]*A[p]*s[p+1]
            anet.PutCyTensor("A1", A[p]);
            anet.PutCyTensor("A2", A[p+1]);
            anet.PutCyTensor("s", s_weight[p+2]);
            psi_gs = anet.Launch(optimal=True)

            if updateon:
                ## put psi_gs to Tensor for Lanczos algorithm
                psi_gs = psi_gs.reshape(-1).get_block()
                psi_gs, Entemp = eig_Lanczos(psi_gs, get_H_psi, (L[p], M, M, R[p + 1]), maxit=maxit,
                                               krydim=krydim)
                Ekeep.append(Entemp)
                psi_gs = cyx.CyTensor(psi_gs.reshape(dim_l, d, d, dim_r), 2)

            dim_new = min(dim_l*d, dim_r*d, dim_cut)
            s_weight[p+1], A[p], B[p+1] = cyx.xlinalg.Svd_truncate(psi_gs, dim_new)
            norm = s_weight[p+1].get_block().Norm().item()
            s_weight[p+1] = s_weight[p+1]/norm

            # ##### new block Hamiltonian
            anet = cyx.Network("Network/R_B_M_Bconj.net")
            ## R[P] = R[p+1]*B[p+1]*B[p+1].conj()*M
            anet.PutCyTensor("R", R[p+1]);
            anet.PutCyTensor("B", B[p+1]);
            anet.PutCyTensor("M", M);
            anet.PutCyTensor('B_Conj', B[p+1].Conj());
            R[p] = anet.Launch(optimal=True)
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
            anet = cyx.Network("Network/s_B_B.net")
            ## psi = s[p]*B[p]*B[p+1]
            anet.PutCyTensor("s", s_weight[p]);
            anet.PutCyTensor("B1", B[p]);
            anet.PutCyTensor("B2", B[p+1]);
            psi_gs = anet.Launch(optimal=True)

            if updateon:
                ## put psi_gs to Tensor for Lanczos algorithm
                psi_gs = psi_gs.reshape(-1).get_block()
                psi_gs, Entemp = eig_Lanczos(psi_gs, get_H_psi, (L[p], M, M, R[p + 1]), maxit=maxit,
                                             krydim=krydim)
                Ekeep.append(Entemp)
                psi_gs = cyx.CyTensor(psi_gs.reshape(dim_l, d, d, dim_r), 2)

            dim_new = min(dim_l * d, dim_r * d, dim_cut)
            s_weight[p + 1], A[p], B[p + 1] = cyx.xlinalg.Svd_truncate(psi_gs, dim_new)
            norm = s_weight[p + 1].get_block().Norm().item()
            s_weight[p + 1] = s_weight[p + 1] / norm

            ##### new block Hamiltonian
            anet = cyx.Network("Network/L_A_M_Aconj.net")
            ## L[p+1] = L[p+1]*A[p]*A[p].Conj()*M
            anet.PutCyTensor("L", L[p]);
            anet.PutCyTensor("A", A[p]);
            anet.PutCyTensor("M", M);
            anet.PutCyTensor('A_Conj', A[p].Conj());
            L[p + 1] = anet.Launch(optimal=True)

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

        # if dispon == 1:
        #     print('Sweep: %d of %d, Energy: %12.12d, Bond dim: %d' % (k, numsweeps, Ekeep[-1], chi))

    return Ekeep, A, s_weight, B

if __name__ == '__main__':
    ##### XX model
    ##### Set bond dimensions and simulation options
    chi = 16;
    Nsites = 20;

    OPTS_numsweeps = 4 # number of DMRG sweeps
    OPTS_dispon = 2 # level of output display
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
    sp = sx+1j*sy
    sm = sx-1j*sy
    eye = cytnx.eye(d).astype(cytnx.Type.ComplexDouble)
    M = cytnx.zeros([4, 4, d, d]).astype(cytnx.Type.ComplexDouble)
    M[0,0,:,:] = M[3,3,:,:] = eye
    M[0,1,:,:] = M[2,3,:,:] = 2**0.5*sp
    M[0,2,:,:] = M[1,3,:,:] = 2**0.5*sm
    ML = cytnx.zeros([4,1,1]).astype(cytnx.Type.ComplexDouble)
    MR = cytnx.zeros([4,1,1]).astype(cytnx.Type.ComplexDouble)
    ML[0,0,0] = 1.; MR[3,0,0] = 1.
    M = cyx.CyTensor(M,0); MR = cyx.CyTensor(MR,0); ML = cyx.CyTensor(ML,0)
    A = [0]*Nsites
    A[0] = cytnx.random.normal([1, d, min(chi, d)], 0., 1.)
    # A[0] = np.random.rand(1,chid,min(chi,chid))

    for k in range(1,Nsites):
        dim1 = A[k-1].shape()[2]; dim2 = d;
        dim3 = min(min(chi, A[k-1].shape()[2] * d), d ** (Nsites - k - 1));
        A[k] = cytnx.random.normal([dim1, dim2, dim3],0.,1.)
    ## Transform A to
    A = [cyx.CyTensor(A[i], 2) for i in range(len(A))]
    En1, A, sWeight, B = dmrg_two_sites(A,ML,M,MR,chi, numsweeps = OPTS_numsweeps, dispon = OPTS_dispon,
                                    updateon = OPTS_updateon, maxit = OPTS_maxit, krydim = OPTS_krydim)
    # print(sWeight[0])
    # print(sWeight[1])
    # print(sWeight[Nsites - 1])
    # print(sWeight[Nsites])
    # exit()

    # for i in range(len(sWeight)):
    #     print(sWeight[i].shape())
    # exit()
    #### Increase bond dim and reconverge
    chi = 32;
    En2, A, sWeight, B = dmrg_two_sites(A,ML,M,MR,chi, numsweeps = OPTS_numsweeps, dispon = OPTS_dispon,
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
    plt.plot(range(len(En1)), En1 - EnExact, 'b', label="chi = 16", marker = 'o')
    plt.plot(range(len(En2)), En2 - EnExact, 'r', label="chi = 32", marker = 'o')
    plt.legend()
    plt.title('DMRG for XX model')
    plt.xlabel('Update Step')
    plt.ylabel('Ground Energy Error')
    plt.show()




