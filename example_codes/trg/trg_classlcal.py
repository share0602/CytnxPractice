from setting import *
import cytnx
from cytnx import cytnx_extension as cyx
from cytnx import linalg as La
import numpy as np

#### Set paramaters
beta = 0.4
Tval = 1/beta
chi = 20
RGstep = 20

Q = cytnx.Tensor([2, 2])
p_beta = np.exp(beta)
m_beta = np.exp(-beta)
Q[0, 0] = Q[1, 1] = p_beta
Q[1, 0] = Q[0, 1] = m_beta
w, v = La.Eigh(Q)
Q_sqrt_tmp = v @ La.Diag(w)**0.5 @ La.Inv(v)
Q_sqrt = cyx.CyTensor(Q_sqrt_tmp,0)

delta_tmp = cytnx.zeros([2,2,2,2])
delta_tmp[0,0,0,0] = delta_tmp[1,1,1,1] = 1
delta = cyx.CyTensor(delta_tmp,0)
anet = cyx.Network('Network/Q4_delta.net')
anet.PutCyTensors(["Q1","Q2","Q3","Q4","delta"],[Q_sqrt]*4+[delta]);
T = anet.Launch(optimal = True)


lnz = 0.0
for k in range(RGstep):
    print('RGstep = ', k+1, 'T.shape() = ', T.shape())
    Tmax = La.Max(La.Abs(T.get_block())).item()
    T = T / Tmax
    lnz += 2 ** (-k) * np.log(Tmax)
    chiT = T.shape()[0]
    chitemp = min(chiT**2, chi)
    ## Construct U1, V1
    stmp, utmp, vtmp = cyx.xlinalg.Svd_truncate(T,chitemp)
    s_sqrt = stmp**0.5
    U1 = cyx.Contract(utmp, s_sqrt)
    V1 = cyx.Contract(s_sqrt, vtmp)
    V1.permute_([1,2,0])
    ## Construct U2, V2
    T.permute_([3,0,1,2])
    stmp, utmp, vtmp = cyx.xlinalg.Svd_truncate(T,chitemp)
    s_sqrt = stmp**0.5
    U2 = cyx.Contract(utmp, s_sqrt)
    V2 = cyx.Contract(s_sqrt, vtmp)
    V2.permute_([1, 2, 0])
    anet = cyx.Network('Network/vuvu.net')
    anet.PutCyTensors(['U1','V1','U2','V2'], [U1,V1,U2,V2])
    T = anet.Launch(optimal = True)

lnz += T.Trace(0,2).Trace(0,1).item() / 2 ** (RGstep)

free_energy = -Tval*lnz
# exit()

##### Compare with exact results (thermodynamic limit)
maglambda = 1 / (np.sinh(2 / Tval) ** 2)
N = 1000000;
x = np.linspace(0, np.pi, N + 1)
y = np.log(np.cosh(2 * beta) * np.cosh(2 * beta) + (1 / maglambda) * np.sqrt(
        1 + maglambda ** 2 - 2 * maglambda * np.cos(2 * x)))
free_exact = -Tval * ((np.log(2) / 2) + 0.25 * sum(y[1:(N + 1)] + y[:N]) / N)
Error = abs(free_energy-free_exact)/free_exact
print('Free Energy = ', free_energy)
print('Exact = ', free_exact)
print('Error = ', Error)
