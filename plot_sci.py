from pyscf import gto, scf, ao2mo, fci, cc, mp, hci
from pyscf.hci import SelectedCI 
from pyscf.fci import cistring
import numpy as np
import itertools
from collections import Counter
import operator
import matplotlib.pyplot as plt
import pyscf
import h5py

filename = 'H2O_631g_diff_sci_sel_r15.hdf5'
hf1 = h5py.File(filename, 'r')
for name in hf1:
    print(name)
    # print(hf1[name])

r = hf1['r'][:]
e_fci = hf1['e_fci'][:]
e_sci = hf1['e_sci'][:]
e_ccsd = hf1['e_ccsd'][:]
err_sci_wrt_fci = hf1['sci_err_wrt_fci'][:]
err_wrt_fci_sci_sel = hf1['err_wrt_fci_sci_sel'][:]
e_ccsd_mp2_sel = hf1['e_ccsd_mp2_sel'][:]
e_ccsd_sci_sel = hf1['e_ccsd_sci_sel'][:]
err_ccsd_wrt_fci = hf1['e_ccsd'][:] - e_fci
n_det_sci = hf1['n_det_sci'][:]
n_det_fci = hf1['n_det_fci'][:]
n_t2_sci_sel = hf1['n_t2_amp_sci_sel'][:]
n_t2_mp2_sel = hf1['n_t2_amp_mp2_sel'][:]
n_t2 = hf1['n_t2_amp'][:]
sci_select_cutoff = hf1['sci_select_cutoff'][:]

err_wrt_fci_mp2_sel = e_ccsd_mp2_sel - e_fci

font = {'size'   : 16}

plt.rc('font', **font)

# # Plotting PES with preselected amplitudes
# plt.plot(r, e_ccsd, label='CCSD', linewidth=3)
# plt.plot(r, e_ccsd_sci_sel, label ='CCSD SCI screen', linewidth=3, linestyle='--')
# plt.plot(r, e_ccsd_mp2_sel, label='CCSD MP2 screen', linewidth=3, linestyle='-.')
# plt.plot(r, e_fci, label='FCI', linewidth=3)
# plt.plot(r, e_sci, label='SCI', linewidth=3, linestyle='--')

# plt.title("PES with $\mathregular{t_2}$ prescreening")
# plt.xlabel("R, $\AA$")
# plt.ylabel("E, Ha")
# plt.legend()
# plt.tight_layout()
# plt.savefig("PES.png", dpi=300)
# plt.show()

# # Plotting error
# plt.plot(r, err_ccsd_wrt_fci*1000, label='CCSD', linewidth=3)
# plt.plot(r, err_wrt_fci_sci_sel*1000, label ='CCSD SCI screen', linewidth=3, linestyle='--')
# plt.plot(r, err_wrt_fci_mp2_sel*1000, label='CCSD MP2 screen', linewidth=3, linestyle='-.')
# plt.plot(r, err_sci_wrt_fci*1000, label='SCI', linewidth=3)
# plt.title("error WRT FCI with $\mathregular{t_2}$ prescreening")
# plt.xlabel("R, $\AA$")
# plt.ylabel("E - $E\_{FCI}$, mHa")
# plt.legend()
# plt.tight_layout()
# plt.savefig("error.png", dpi=300)
# plt.show()

# # Plotting n t2 amplitudes
# plt.plot(r, n_t2_mp2_sel, label ='MP2 screen', linewidth=3)
# plt.plot(r, n_t2_sci_sel, label='SCI screen', linewidth=3)
# plt.plot(r, n_t2, label='CCSD', linewidth=3)
# plt.title("Number of $\mathregular{t_2}$ amplitudes")
# plt.xlabel("R, $\AA$")
# plt.ylabel("N")
# plt.legend()
# plt.tight_layout()
# plt.savefig("amps.png", dpi=300)
# plt.show()

# # Plotting number of filtered determinants
# plt.plot(r, n_det_sci, label ='SCI', linewidth=3)
# plt.plot(r, n_det_fci, label='FCI', linewidth=3)
# plt.title("Number of determinants")
# plt.xlabel("R, $\AA$")
# plt.ylabel("N")
# plt.legend()
# plt.tight_layout()
# plt.savefig("dets.png", dpi=300)
# plt.show()


print(n_t2)




# Plot error vs ci_sel_coeff
plt.plot(sci_select_cutoff, err_wrt_fci_sci_sel, linewidth=3)
plt.yscale('log')
plt.xscale('log')
plt.xlabel("SCI select cutoff")
plt.ylabel("E - $E\_{FCI}$, mHa")
plt.title("Error WRT FCI vs cutoff")
plt.tight_layout()
plt.savefig("error_vs_cutoff.png", dpi=300)
plt.show()

# Plot number amps vs threshold
plt.plot(sci_select_cutoff, n_t2_sci_sel, linewidth=3)
plt.xscale('log')
plt.xlabel("SCI select cutoff")
plt.ylabel("N")
plt.title("Number of $\mathregular{t_2}$ amplitudes vs cutoff ")
plt.tight_layout()
plt.savefig("n_amps_vs_cutoff.png", dpi=300)
plt.show()

# Plot error vs number amps 
plt.plot(n_t2_sci_sel, err_wrt_fci_sci_sel, linewidth=3)
# plt.xscale('log')
plt.xlabel("N")
plt.ylabel("E - $E\_{FCI}$, mHa")
plt.title("Error WRT FCI vs number of $\mathregular{t_2}$ amplitudes")
plt.tight_layout()
plt.savefig("err_vs_n_amps.png", dpi=300)
plt.show()