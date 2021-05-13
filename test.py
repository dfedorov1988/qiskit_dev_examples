from pyscf import gto, ao2mo
import h5py
import numpy
def view(h5file, dataname='eri_mo'):
    with h5py.File(h5file, 'r') as f5:
        print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
mo1 = numpy.random.random((mol.nao_nr(), 10))

eri1 = ao2mo.full(mol, mo1)
print(eri1.shape)

eri = mol.intor('int2e_sph', aosym='s8')
eri1 = ao2mo.full(eri, mo1, compact=False)
print(eri1.shape)

ao2mo.full(mol, mo1, 'full.h5')
view('full.h5')

ao2mo.full(mol, mo1, 'full.h5', dataname='new', compact=False)
view('full.h5', 'new')

ao2mo.full(mol, mo1, 'full.h5', intor='int2e_ip1_sph', aosym='s1', comp=3)
view('full.h5')

ao2mo.full(mol, mo1, 'full.h5', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
view('full.h5')
