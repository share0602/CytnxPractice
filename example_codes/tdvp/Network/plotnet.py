from setting import *
import cytnx as cy
from cytnx import cytnx_extension as cyx


# anet = cyx.Network("extend_corner_corboz.net")
# anet = cyx.Network("s_vt_A.net")
anet = cyx.Network("L_A_W_Aconj.net")
anet = cyx.Network("C_L_R.net")

# anet = cyx.Network("psi_L_M1_M2_R.net")
# anet = cyx.Network("A_A_s.net")
# anet = cyx.Network("R_B_M_Bconj.net")
# anet = cyx.Network("M_u_s.net")
# anet = cyx.Network("s_B.net")
# anet = cyx.Network("psi_L_W_R.net")

anet.Diagram(figsize=[6,5])

