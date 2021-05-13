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

def convert_t1_to_blocks(t1_ind_list, nelec, norb):
    t1_list = []
    for t1_amp_ind in t1_ind_list:
        t1_list.append([t1_amp_ind[0], t1_amp_ind[1] + int(nelec/2)])
        t1_list.append([t1_amp_ind[0]+norb, t1_amp_ind[1] + norb + int(nelec/2)])
    return t1_list

def convert_t2_to_blocks(t2_ind_list, nelec, norb):
    t2_list = []
    for t2_amp_ind in t2_ind_list:
        t2_list.append([t2_amp_ind[0], t2_amp_ind[2] + int(nelec/2), t2_amp_ind[1]+norb, t2_amp_ind[3] + norb + int(nelec/2)])
    return t2_list

def convert_t3_to_blocks(t3_ind_list, nelec, norb):
    t3_list = []
    for t3_amp_ind in t3_ind_list:
        t3_list.append([t3_amp_ind[0], t3_amp_ind[3] + int(nelec/2),
                        t3_amp_ind[1]+norb, t3_amp_ind[4] + norb + int(nelec/2),
                        t3_amp_ind[2]+norb, t3_amp_ind[5] + norb + int(nelec/2)
                       ])
    return t3_list

def print_sorted_dict(dict_to_print):
    sorted_dict = dict( sorted(dict_to_print.items(), key=lambda kv: np.max(np.abs(kv[1])), reverse=True))
    for key in sorted_dict.keys():
        print(key, sorted_dict[key])


def filter_dets_from_dict(d, thresh=1e-08):
    filtered_dict = dict()
    for key in d.keys():
        if np.max(np.abs(d[key])) > thresh:

            filtered_dict[key] = d[key]
        # else:
        #     print(key, np.max(np.abs(d[key])))
    # print("filtered", filtered_dict)
    return filtered_dict  

def filter_out_excitations_from_det_dict(dict_to_filter, order_of_exc, nelec):
    filtered_dict = dict()
    nocc = int(nelec/2)
    for key in dict_to_filter.keys():
        n_el_occ = 0
        for x in key[0:nocc]:
            n_el_occ += int(x)
        # print("n_el_occ", n_el_occ, key)
        if nelec - n_el_occ == order_of_exc:
            filtered_dict[key] = dict_to_filter[key]
    return filtered_dict

def generate_triple_exc_dets(nelec, norb, occ_array):

    det_vs_amp_ind_dict = dict()

    for exc_from_i in range(int(nelec/2)):
        for exc_from_j in range(int(nelec/2)):
            for exc_from_k in range(int(nelec/2)):
                for exc_to_a in range(norb-int(nelec/2)):
                    for exc_to_b in range(norb-int(nelec/2)):
                        for exc_to_c in range(norb-int(nelec/2)):

                            ini_occ = occ_array.copy()
                            ini_occ[exc_from_i] = ini_occ[exc_from_i] - 1
                            ini_occ[exc_from_j] = ini_occ[exc_from_j] - 1
                            ini_occ[exc_from_k] = ini_occ[exc_from_k] - 1
                            
                            ini_occ[exc_to_a + int(nelec/2)] = ini_occ[exc_to_a + int(nelec/2)] + 1
                            ini_occ[exc_to_b + int(nelec/2)] = ini_occ[exc_to_b + int(nelec/2)] + 1
                            ini_occ[exc_to_c + int(nelec/2)] = ini_occ[exc_to_c + int(nelec/2)] + 1

                            if ini_occ[exc_from_i] in range(0,3) and ini_occ[exc_from_j] in range(0,3) \
                                and ini_occ[exc_from_k] in range(0,3) and ini_occ[exc_to_a + int(nelec/2)] in range(0,3)\
                                    and ini_occ[exc_to_b + int(nelec/2)] in range(0,3) and ini_occ[exc_to_c + int(nelec/2)] in range(0,3):
                                        # valid triple excitation
                                        string = ''
                                        for l in range(np.shape(ini_occ)[0]):
                                            string += str(int(ini_occ[l]))
                                        if string in det_vs_amp_ind_dict.keys():
                                            det_vs_amp_ind_dict[string].append([exc_from_i, exc_from_j, exc_from_k,
                                                                                exc_to_a, exc_to_b, exc_to_c])
                                        else:
                                            det_vs_amp_ind_dict[string] = [[exc_from_i, exc_from_j, exc_from_k,
                                                                            exc_to_a, exc_to_b, exc_to_c]]

    return det_vs_amp_ind_dict

def run_calcs(r, mol, basis, sci_select_cutoff, sci_ci_coeff_cutoff, det_thresh, discarded_amp_thresh, mp2_thresh):
   
    print("r=", r)
    basis = basis
    sci_select_cutoff = sci_select_cutoff
    sci_ci_coeff_cutoff = sci_ci_coeff_cutoff
    det_thresh = det_thresh # filters SCI selected dets (in FCI format there's a lot of zeros)
    discarded_amp_thresh = discarded_amp_thresh # for counting amplitudes filtered out by SCI
    mp2_thresh = mp2_thresh # throw out mp amps smaller than this 

    mol.verbose = 0

    # RHF
    mf = scf.RHF(mol)
    e_hf = mf.kernel()
    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    eri = ao2mo.kernel(mol, mf.mo_coeff)
    

    # Full CI
    print("\nFCI")
    cisolver = fci.FCI(mol)
    e_fci, ci = cisolver.kernel(h1, eri, h1.shape[1], mol.nelec, ecore=mol.energy_nuc())
    print("FCI energy", e_fci)
    norb = h1.shape[1]
    nelec = mol.nelectron
    print("nelec", nelec)
    print("norb", norb)
    print("Num Strings: ", cistring.num_strings(norb,nelec/2))

    coeff = list()
    spin_orb_strings = list()
    det_dict = dict()

    for alpha_addr in range(ci.shape[0]):
        for beta_addr in range(ci.shape[0]):
            # In pyscf the lower energy orbitals are to the RIGHT!!! so we have to reverse the strings

            a_str = bin(cistring.addrs2str(norb, int(nelec/2),[alpha_addr])[0])[2:].rjust(norb, '0')[::-1] # converting to strings with occ numbers
            b_str = bin(cistring.addrs2str(norb, int(nelec/2),[beta_addr])[0])[2:].rjust(norb, '0')[::-1]
            
            spin_orb_str = ''
            spatial_orb_str = ''
            for n in range(len(a_str)):
                spatial_orb_str += str(int(a_str[n]) + int(b_str[n]))
                spin_orb_str += a_str[n]
                spin_orb_str += b_str[n]

            if spatial_orb_str in det_dict.keys():
                det_dict[spatial_orb_str].append(ci[alpha_addr, beta_addr])
            else:
                det_dict[spatial_orb_str] = [ci[alpha_addr, beta_addr]]
            spin_orb_strings.append(spin_orb_str)
            coeff.append(ci[alpha_addr, beta_addr])
    print("Number of FCI determinants", len(det_dict.keys()))

    # Running CCSD
    print("\nCCSD")
    mycc = cc.CCSD(mf)
    result = mycc.kernel()
    t2 = result[2]
    t1 = result[1]
    ccsd_conv = result[0]


    # Spin Orbital HF configuration
    # hf_conf = '' 
    # for n in range(nelec):
    #     hf_conf += '1'

    # for m in range(norb*2-nelec):
    #     hf_conf += '0'
    # print('hf_conf', hf_conf)

    # Spatial Orbital HF configuration
    hf_conf = '' 

    for n in range(int(nelec/2)):
        hf_conf += '2'

    for m in range(norb - int(nelec/2)):
        hf_conf += '0'
    print('HF occupation number (spatial orbitals):', hf_conf)

    # Forming occupation number numpy array
    occ_array = np.zeros(norb)
    for n in range(norb):
        occ_array[n] = int(hf_conf[n])

    amp_dict = dict()
    det_vs_amp_ind = dict() # dictionary of t2 amplitudes corresponding to spatial orbital occupation number

    # double amplitudes
    strings_t2 = []
    for exc_from_i in range(np.shape(t2)[0]):
        for exc_from_j in range(np.shape(t2)[1]): 
            for exc_to_a in range(np.shape(t2)[2]):  
                for exc_to_b in range(np.shape(t2)[3]):
                    # print("ijab", 2*exc_from_i, 2*exc_from_j, 2*exc_to_a+nelec, 2*exc_to_b+nelec)
                    ini_occ = occ_array.copy()
                    ini_occ[exc_from_i] = ini_occ[exc_from_i] - 1
                    ini_occ[exc_from_j] = ini_occ[exc_from_j] - 1
                    ini_occ[exc_to_a + int(nelec/2)] = ini_occ[exc_to_a + int(nelec/2)] + 1
                    ini_occ[exc_to_b + int(nelec/2)] = ini_occ[exc_to_b + int(nelec/2)] + 1
                    string = ''
                    for l in range(np.shape(ini_occ)[0]):
                        string += str(int(ini_occ[l]))
                    if string in amp_dict.keys():
                        amp_dict[string].append(t2[exc_from_i, exc_from_j, exc_to_a, exc_to_b])
                        det_vs_amp_ind[string].append([exc_from_i, exc_from_j, exc_to_a, exc_to_b])
                    else:
                        amp_dict[string] = [t2[exc_from_i, exc_from_j, exc_to_a, exc_to_b]]
                        det_vs_amp_ind[string] = [[exc_from_i, exc_from_j, exc_to_a, exc_to_b]]
                    strings_t2.append(string)
                    
    print("Number of determinants formed from CCSD t2 amplitudes", len(strings_t2))


    det_vs_amp_t1_ind = dict() # dictionary of t1 amplitudes corresponding to spatial orbital occupation number
    # single amplitudes
    strings_t1 = []
    for exc_from_i in range(np.shape(t1)[0]):
        for exc_to_a in range(np.shape(t1)[1]):
            ini_occ = occ_array.copy()
            ini_occ[exc_from_i] = ini_occ[exc_from_i] - 1
            ini_occ[exc_to_a + int(nelec/2)] = ini_occ[exc_to_a + int(nelec/2)] + 1
            string = ''
            for l in range(np.shape(ini_occ)[0]):
                string += str(int(ini_occ[l]))

            if string in amp_dict.keys():
                amp_dict[string].append(t1[exc_from_i, exc_to_a])
                det_vs_amp_t1_ind[string].append([exc_from_i, exc_to_a])
            else:
                amp_dict[string] = [t1[exc_from_i, exc_to_a]]
                det_vs_amp_t1_ind[string] = [[exc_from_i, exc_to_a]]
            strings_t1.append(string)
    print(det_vs_amp_t1_ind)
    print("num of CCSD t1 dets", len(strings_t1))
    # print(Counter(strings_t1).keys())
    # print(Counter(strings_t1).values())

    # MP2
    print("\nMP2")
    pt = mp.MP2(mf).run()
    mp2 = pt.run()
    mp2_dict = dict()

    # mp2 amplitudes
    strings_mp2 = []
    amp_inds_smaller_than_thresh = []
    for exc_from_i in range(np.shape(mp2.t2)[0]):
        for exc_from_j in range(np.shape(mp2.t2)[1]): 
            for exc_to_a in range(np.shape(mp2.t2)[2]):  
                for exc_to_b in range(np.shape(mp2.t2)[3]):
                    if np.abs(mp2.t2[exc_from_i, exc_from_j, exc_to_a, exc_to_b]) < mp2_thresh:
                        amp_inds_smaller_than_thresh.append([exc_from_i, exc_from_j, exc_to_a, exc_to_b])
                    # print("ijab", 2*exc_from_i, 2*exc_from_j, 2*exc_to_a+nelec, 2*exc_to_b+nelec)
                    ini_occ = occ_array.copy()
                    ini_occ[exc_from_i] = ini_occ[exc_from_i] - 1
                    ini_occ[exc_from_j] = ini_occ[exc_from_j] - 1
                    ini_occ[exc_to_a + int(nelec/2)] = ini_occ[exc_to_a + int(nelec/2)] + 1
                    ini_occ[exc_to_b + int(nelec/2)] = ini_occ[exc_to_b + int(nelec/2)] + 1
                    string = ''
                    for l in range(np.shape(ini_occ)[0]):
                        string += str(int(ini_occ[l]))
                    # print("str", string)
                    if string in mp2_dict.keys():
                        mp2_dict[string].append(mp2.t2[exc_from_i, exc_from_j, exc_to_a, exc_to_b])
                    else:
                        mp2_dict[string] = [mp2.t2[exc_from_i, exc_from_j, exc_to_a, exc_to_b]]
                    strings_mp2.append(string)
                    
    print("number of MP2 t2 dets", len(strings_mp2))
    # print("MP2 amps smaller than {} is \n{}".format(mp2_thresh, amp_inds_smaller_than_thresh))
    # print(Counter(strings_mp2).keys())
    # print(Counter(strings_mp2).values())

    # Selected CI
    print("\nSelected CI")
    sci_solver = hci.SCI(mol)
    sci_solver.select_cutoff = sci_select_cutoff
    sci_solver.ci_coeff_cutoff = sci_ci_coeff_cutoff

    nmo = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    h2 = ao2mo.full(mol, mf.mo_coeff)
    e_sci, civec = sci_solver.kernel(h1, h2, mf.mo_coeff.shape[1], mol.nelectron, ecore=mol.energy_nuc())
    print("e_sci", e_sci)
    print("Total number of selected CI determinants: {}".format(len(civec[0])))

    # For conversion of SCI to FCI vector for comparison
    nelec = int(nelec/2), int(nelec/2)
    neleca, nelecb = nelec

    # Transforming SCI vector to FCI format
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    stradic = dict(zip(strsa,range(strsa.__len__())))
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    strbdic = dict(zip(strsb,range(strsb.__len__())))
    na = len(stradic)
    nb = len(strbdic)
    ndet = len(civec[0])
    fcivec = np.zeros((na,nb))
    for idet, (stra, strb) in enumerate(civec[0]._strs.reshape(ndet,2,-1)):
        ka = stradic[stra[0]]
        kb = strbdic[strb[0]]
        fcivec[ka,kb] = civec[0][idet] # SCI coefficients array in FCI format (with alpha and beta strings) 


    # Obtaining spatial orbital occupation numbers for SCI
    coeff_sci = list()
    spatial_orb_strings_sci = list()
    det_dict_sci = dict()
    for alpha_addr in range(fcivec.shape[0]):
        for beta_addr in range(fcivec.shape[0]):
            # In pyscf the lower energy orbitals are to the RIGHT!!! so we have to reverse the strings

            a_str = bin(cistring.addrs2str(norb, int(nelec[0]),[alpha_addr])[0])[2:].rjust(norb, '0')[::-1] # converting to strings with occ numbers
            b_str = bin(cistring.addrs2str(norb, int(nelec[1]),[beta_addr])[0])[2:].rjust(norb, '0')[::-1]
            spatial_orb_str = ''
            for n in range(len(a_str)):
                spatial_orb_str += str(int(a_str[n]) + int(b_str[n]))

            if spatial_orb_str in det_dict_sci.keys():
                det_dict_sci[spatial_orb_str].append(fcivec[alpha_addr, beta_addr])
            else:
                det_dict_sci[spatial_orb_str] = [fcivec[alpha_addr, beta_addr]]
            spatial_orb_strings_sci.append(spatial_orb_str)
            coeff_sci.append(fcivec[alpha_addr, beta_addr])


    filtered_sci_dets = filter_dets_from_dict(det_dict_sci, det_thresh)
    print("Number of SCI spatial orbitals configurations with CI coefficients > {} is {}".format(det_thresh, len(filtered_sci_dets.keys())))


    # Filter amps using SCI
    thrown_out_dets = []
    thrown_out_amps = []
    for key in det_dict_sci.keys():
        if key not in filtered_sci_dets.keys():
            thrown_out_dets.append(key) # this gives us list of dets that were filtered out
            # print(key, det_dict_sci[key])
            if key in amp_dict.keys():
                thrown_out_amps.append(np.abs(np.max(amp_dict[key]))) 

    # Filter amps using MP2

    # Printing dictionaries with determinants for FCI, CCSD, MP2, SCI

    # print('\nFCI Spatial orbital configurations and Full CI coefficients in descending order : ')
    # print_sorted_dict(det_dict)

    # print('\nCCSD Spatial orbital configurations and CCSD amplitudes in descending order: ')
    # print_sorted_dict(amp_dict)

    # print('\nMP2 Spatial orbital configurations and MP2 amplitudes in descending order:: ')
    # print_sorted_dict(mp2_dict)

    # print('\nSCI Spatial orbital configurations and CI coefficients in descending order: ')
    # print_sorted_dict(det_dict_sci)

    # print('\nSCI spatial orbital configurations and CI coefficients in descending order > threshold: ')
    # print_sorted_dict(filtered_sci_dets)


    # Creating list of t2 amplitudes to keep after SCI screening
    # matches amps with dets 
    amps_to_keep = []
    n_small_amps_kept = 0
    for key in filtered_sci_dets.keys():
    # for key in filtered_sci_dets.keys():
        if key in det_vs_amp_ind.keys():
            for amp in det_vs_amp_ind[key]:
                # EXPERIMENT!!!!!!!!!!!
                if np.abs(t2[amp[0], amp[1], amp[2], amp[3]]) > 1e-10:
                    # print(t2[amp[0], amp[1], amp[2], amp[3]]) 
                    amps_to_keep.append(amp)
                else:
                    # print(t2[amp[0], amp[1], amp[2], amp[3]]) 
                    n_small_amps_kept +=1 
        else:
            # print("key {}  is not in dict".format(key))
            pass

    # Creating list of t1 amplitudes to keep after SCI screening
    # matches amps with dets 
    amps_t1_to_keep = []
    n_small_t1_amps_kept = 0
    for key in filtered_sci_dets.keys():
    # for key in filtered_sci_dets.keys():
        if key in det_vs_amp_t1_ind.keys():
            for amp in det_vs_amp_t1_ind[key]:
                # EXPERIMENT!!!!!!!!!!!
                # print("t1", t1[amp[0], amp[1]])
                if np.abs(t1[amp[0], amp[1]]) > 1e-10:
                    # print(t2[amp[0], amp[1], amp[2], amp[3]]) 
                    amps_t1_to_keep.append(amp)
                else:
                    # print(t2[amp[0], amp[1], amp[2], amp[3]]) 
                    n_small_t1_amps_kept +=1 
        else:
            # print("key {}  is not in dict".format(key))
            pass
    # print("t1 amplitudes to keep:", amps_t1_to_keep)
    # print("t1 det vs amp", det_vs_amp_t1_ind)

    # Creating array for t2 amplitudes where we do prescreening using selected CI
    t2_filtered = np.zeros(np.shape(t2))
    for inds in amps_to_keep:
        t2_filtered[inds[0], inds[1], inds[2], inds[3]] = t2[inds[0], inds[1], inds[2], inds[3]]
        # print(t2[inds[0], inds[1], inds[2], inds[3]])

    # Creating array for t1 amplitudes where we do prescreening using selected CI
    t1_filtered = np.zeros(np.shape(t1))
    for inds in amps_t1_to_keep:
        t1_filtered[inds[0], inds[1]] = t1[inds[0], inds[1]]

    # filtering amps by MP2
    t2_filtered_by_mp2 = t2.copy()
    for inds in amp_inds_smaller_than_thresh:
        t2_filtered_by_mp2[inds[0], inds[1], inds[2], inds[3]] = 0

    # Results to save for analysis

    # Energies
    e_ccsd_mp2_filter = mycc.energy(t1, t2_filtered_by_mp2) + e_hf
    e_ccsd_sci_filter = mycc.energy(t1, t2_filtered) + e_hf
    # e_ccsd_sci_filter = mycc.energy(t1_filtered, t2_filtered) + e_hf # filtering t1 too
    e_ccsd = mycc.energy(t1, t2) + e_hf

    # Errors
    sci_err_wrt_fci = e_sci[0] - e_fci
    ccsd_err_wrt_fci = e_ccsd - e_fci
    err_wrt_ccsd = e_ccsd_sci_filter - e_ccsd
    err_wrt_fci = e_ccsd_sci_filter - e_fci
    err_ccsd_m2_filter = e_ccsd_mp2_filter - e_fci

    # Amplitudes and determinants
    n_dets_fci = len(det_dict.keys())
    n_dets_sci = len(civec[0])
    n_ccsd_amp = np.shape(t2)[0]*np.shape(t2)[1]*np.shape(t2)[2]*np.shape(t2)[3]
    n_t2_amps_mp2_filter = n_ccsd_amp - len(amp_inds_smaller_than_thresh)
    n_amps_to_keep = len(amps_to_keep)
    thrown_out_array = np.asarray(thrown_out_amps)
    max_amp_thrown_out = np.max(thrown_out_array)
    n_amps_thrown_out = len(thrown_out_array[thrown_out_array>discarded_amp_thresh])

    print("\nResults:")

    print("{:<40s}{:>30.8e}".format("SCI Error WRT FCI:", sci_err_wrt_fci))
    print("{:<40s}{:>30.8e}".format("CCSD Error WRT FCI:", ccsd_err_wrt_fci))
    print("\n{:<40s}{:>30.8e}".format("Error WRT CCSD due to cutoff:", err_wrt_ccsd))
    print("{:<40s}{:>30.8e}".format("Error WRT FCI due to cutoff:", err_wrt_fci))

    print("\n{:<40s}{:>30.8e}".format("MP2 cutoff threshold", mp2_thresh))
    print("{:<40s}{:>30.8e}".format("Error of CCSD WRT FCI, MP2 cutoff ", err_ccsd_m2_filter))
    print("{:<40s}{:>29d}".format("Number of CCSD t2 amplitudes, MP2 filter:", n_t2_amps_mp2_filter))


    print("\n{:<40s}{:>30d}".format("Total number of FCI determinants:", n_dets_fci))
    print("{:<40s}{:>30d}".format("Total number of SCI determinants:", n_dets_sci))
    print("\n{:<40s}{:>30d}".format("Number of CCSD t2 amplitudes", n_ccsd_amp))
    print("{:<40s}{:>30d}".format("Number of t2 amplitudes screened by SCI:", n_amps_to_keep))
    print("{:<30s}{:<2.2e}:{:>21d}".format("Number of kept amplitudes smaller than ", discarded_amp_thresh, n_small_amps_kept))
    print("{:<40s}{:<2.2e} thrown out:{:>10d}".format("Number of amplitudes larger than", discarded_amp_thresh,
        n_amps_thrown_out))
    print("\n{:<40s}{:>30.8e}".format("Largest discarded amplitude:", max_amp_thrown_out))
    curr_r_dict = dict()
    
    curr_r_dict['r'] = r
    curr_r_dict['sci_select_cutoff'] = sci_select_cutoff
    curr_r_dict['sci_ci_coeff_cutoff'] = sci_ci_coeff_cutoff
    curr_r_dict['basis'] = basis
    curr_r_dict['n_t2_amp_mp2_sel'] = n_t2_amps_mp2_filter
    curr_r_dict['mp2_thresh'] = mp2_thresh
    curr_r_dict['n_t2_amp'] = n_ccsd_amp
    curr_r_dict['n_t2_amp_sci_sel'] = n_amps_to_keep
    curr_r_dict['n_det_fci'] = n_dets_fci
    curr_r_dict['n_det_sci'] = n_dets_sci
    curr_r_dict['ccsd_err_wrt_fci'] = ccsd_err_wrt_fci
    curr_r_dict['sci_err_wrt_fci'] = sci_err_wrt_fci
    curr_r_dict['err_wrt_ccsd_sci_sel'] = err_wrt_ccsd
    curr_r_dict['err_wrt_fci_sci_sel'] = err_wrt_fci
    curr_r_dict['n_amps_lt_mp2_thresh_discarded'] = n_amps_thrown_out
    curr_r_dict['e_fci'] = e_fci
    curr_r_dict['e_hf'] = e_hf
    curr_r_dict['e_sci'] = e_sci
    curr_r_dict['e_ccsd'] = e_ccsd
    curr_r_dict['e_ccsd_mp2_sel'] = e_ccsd_mp2_filter
    curr_r_dict['e_ccsd_sci_sel'] = e_ccsd_mp2_filter
    curr_r_dict['ccsd_conv'] = ccsd_conv
    
    # Testing the conversion from interleaved spins [ij -> ab] to block spins [ia -> jb] (qiskit format)


    # print("t1 amplitudes kept")
    # for amp in amps_t1_to_keep:
    #     print(amp, t1[amp[0], amp[1]])

    print("t2 amplitudes kept")
    for amp in amps_to_keep:
        print(amp, t2[amp[0], amp[1], amp[2], amp[3]])
    single_dets = filter_out_excitations_from_det_dict(det_dict, 1, mol.nelectron)
    double_dets = filter_out_excitations_from_det_dict(det_dict, 2, mol.nelectron)
    triple_dets = filter_out_excitations_from_det_dict(det_dict, 3, mol.nelectron)
    
    
    print("\nPrinting triply excited determinants")
    filtered_triples = filter_dets_from_dict(triple_dets, thresh=1e-04)
    print_sorted_dict(filtered_triples)
    # print(occ_array)
    triples_dict = generate_triple_exc_dets(mol.nelectron, norb, occ_array)
    # print("triples_dict", triples_dict)
    print("Triple indices to include")
    t3_ind_to_include = []
    for key in filtered_triples.keys():
        print("key", key)
        print(triples_dict[key])
        for item in triples_dict[key]:
            t3_ind_to_include.append(item)

    print(nelec)
    print(norb)
    print("t1 before conversion", amps_t1_to_keep)    
    print("CONVERTED", convert_t1_to_blocks(amps_t1_to_keep, mol.nelectron, norb))

    print("\nt2 conversion")
    converted_t2 = convert_t2_to_blocks(amps_to_keep, mol.nelectron, norb)  
    for n in range(len(amps_to_keep)):
        print(amps_to_keep[n], "--->", converted_t2[n], t2[amps_to_keep[n][0], amps_to_keep[n][1], amps_to_keep[n][2], amps_to_keep[n][3]])

    # print("TEST")
    # t2_filtered[1, 2, 2, 3] = 0
    # e_test = mycc.energy(t1, t2_filtered) + e_hf
    # print(e_ccsd_sci_filter - e_test)

    
    print("t3 before conversion\n", t3_ind_to_include)
    print("\nCONVERTED\n", convert_t3_to_blocks(t3_ind_to_include, mol.nelectron, norb))

    return curr_r_dict


# Input parameters
filename = 'test.hdf5'
basis = 'sto-3g'
sci_select_cutoff = .001
sci_ci_coeff_cutoff = .001
det_thresh = 1e-15 # filters SCI selected dets (in FCI format there's a lot of zeros)
discarded_amp_thresh = 1e-06 # for counting amplitudes filtered out by SCI
mp2_thresh = 1e-6 # throw out mp amps smaller than this 

dict_list = []

# # Loop over distances
# r_ini = 0.8
# r_end = 3
# n_points = 2
# distances = np.linspace(r_ini, r_end, num=n_points)
# for r in distances:

# Loop over ci_select threshold
# coeff_array = [0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
coeff_array = [0.0001]
sci_ci_coeff_cutoff = 0.001
r = 1.5
for sci_select_cutoff in coeff_array:

    # For H2O molecule
    # theta = 104.5
    # c,s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    # R = np.array(((c, -s), (s, c))) #Rotation matrix
    # x,y = R @ np.array((0, r))
    # mol = gto.M(atom='O .0 .0 .0; H .0 .0 {}; H .0 {} {}'.format(r, x, y), basis=basis)
    # mol = gto.M(atom='Be 0 0 0; H 0 0 {}; H 0 0 {}'.format(r, -r), basis=basis)
    # mol = gto.M(atom='H 0 0 0; H 0 0 {}; H 0 0 {}; H 0 0 {}'.format(r, r*2, r*3), basis=basis)
    # mol = gto.M(atom='H 0 0 0; H 0 0 {}; H 0 0 {}; H 0 0 {}; H 0 0 {}; H 0 0 {}'.format(r, r*2, r*3, r*4, r*5), basis=basis)
    # mol = gto.M(atom='N 0 0 0; N 0 0 {}'.format(r), basis=basis)
    mol = gto.M(atom='Li 0 0 0; H 0 0 {}'.format(r), basis=basis)
    dict_list.append(run_calcs(r, mol, basis, sci_select_cutoff, sci_ci_coeff_cutoff, det_thresh, discarded_amp_thresh, mp2_thresh))
    # try:
    #     dict_list.append(run_calcs(r, mol, basis, sci_select_cutoff, sci_ci_coeff_cutoff, det_thresh, discarded_amp_thresh, mp2_thresh))
    # except:
    #     print("Calculation failed")
    #     continue


e_hf_array = []
e_fci_array = []
e_sci_array = []
e_ccsd_array = []
e_ccsd_mp2_sel_array = []
e_ccsd_sci_sel_array = []
n_amps_lt_mp2_thresh_discarded_array = []
err_wrt_fci_sci_sel_array = []
err_wrt_ccsd_sci_sel_array = []
sci_err_wrt_fci_array = []
ccsd_err_wrt_fci_array = []
n_det_sci_array = []
n_det_fci_array = []
n_t2_amp_sci_sel_array = []
n_t2_amp_mp2_sel_array = []
n_t2_amp_array = []
r_array = []
ccsd_conv_array = []
sci_select_cutoff_array = []

for data in dict_list:
    r_array.append(data['r'])
    e_hf_array.append(data['e_hf'])
    e_fci_array.append(data['e_fci'])
    e_sci_array.append(data['e_sci'])
    e_ccsd_array.append(data['e_ccsd'])
    e_ccsd_mp2_sel_array.append(data['e_ccsd_mp2_sel'])
    e_ccsd_sci_sel_array.append(data['e_ccsd_sci_sel'])
    n_amps_lt_mp2_thresh_discarded_array.append(data['n_amps_lt_mp2_thresh_discarded'])
    err_wrt_fci_sci_sel_array.append(data['err_wrt_fci_sci_sel'])
    err_wrt_ccsd_sci_sel_array.append(data['err_wrt_ccsd_sci_sel'])
    sci_err_wrt_fci_array.append(data['sci_err_wrt_fci'])
    ccsd_err_wrt_fci_array.append(data['ccsd_err_wrt_fci'])
    n_det_sci_array.append(data['n_det_sci'])
    print("DEBUG", data['n_det_sci'])
    n_det_fci_array.append(data['n_det_fci'])
    n_t2_amp_sci_sel_array.append(data['n_t2_amp_sci_sel'])
    n_t2_amp_mp2_sel_array.append(data['n_t2_amp_mp2_sel'])
    n_t2_amp_array.append(data['n_t2_amp'])
    ccsd_conv_array.append(data['ccsd_conv'])
    sci_select_cutoff_array.append(data['sci_select_cutoff'])

# Writing datasets into hdf5 
hf = h5py.File(filename, 'w')
hf.create_dataset('r', data=r_array)
hf.create_dataset('e_fci', data=e_fci_array)
hf.create_dataset('e_hf', data=e_hf_array)
hf.create_dataset('e_sci', data=e_sci_array)
hf.create_dataset('e_ccsd', data=e_ccsd_array)
hf.create_dataset('e_ccsd_sci_sel', data=e_ccsd_sci_sel_array)
hf.create_dataset('e_ccsd_mp2_sel', data=e_ccsd_mp2_sel_array)
hf.create_dataset('n_amps_lt_mp2_thresh_discarded', data=n_amps_lt_mp2_thresh_discarded_array)
hf.create_dataset('err_wrt_fci_sci_sel', data=err_wrt_fci_sci_sel_array)
hf.create_dataset('err_wrt_ccsd_sci_sel', data=err_wrt_ccsd_sci_sel_array)
hf.create_dataset('sci_err_wrt_fci', data=sci_err_wrt_fci_array)
hf.create_dataset('ccsd_err_wrt_fci', data=ccsd_err_wrt_fci_array)
hf.create_dataset('n_det_sci', data=n_det_sci_array)
hf.create_dataset('n_det_fci', data=n_det_fci_array)
hf.create_dataset('n_t2_amp_sci_sel', data=n_t2_amp_sci_sel_array)
hf.create_dataset('n_t2_amp_mp2_sel', data=n_t2_amp_mp2_sel_array)
hf.create_dataset('n_t2_amp', data=n_t2_amp_array)
hf.create_dataset('sci_select_cutoff', data=sci_select_cutoff_array)

# Writing attributes into hdf5
hf['basis'] = basis
# hf['sci_select_cutoff'] = sci_select_cutoff
hf['sci_ci_coeff_cutoff'] = sci_ci_coeff_cutoff
hf['mp2_thresh'] = mp2_thresh
