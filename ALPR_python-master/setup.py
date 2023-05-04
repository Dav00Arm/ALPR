from setuptools import setup
from Cython.Build import cythonize
from getmac import get_mac_address as gma
import os

# data = gma()
# with open(fname, 'a') as f:
#     f.write(f'\nmc_address = {[data]}')

setup(
    ext_modules = cythonize(['aug.py', 'ach.py', 'bun.py', 'bu.py', 'cmn.py', 'crftour.py', 'crftut.py',
                            'crft.py', 'dproc.py', 'dldrs.py', 'dtst.py', 'dtc.py', 'dtcin.py', 'dwn.py', 'esycor.py', 
                            'expr.py', 'pexrt.py', 'futrextr.py', 'flut.py', 'frmod.py', 'genr.py', 'detgenr.py', 'gtmdls.py', 
                            'iprmg.py', 'namlakfltr.py', 'infr.py', 'ichmatch.py', 'licut.py', 'linassig.py', 'mtrc.py', 
                            'pltdtmsc.py', 'csm.py', 'vslite.py', 'v2.py', 'v1lite.py', 'mdl.py', 'nnmatch.py', 'pslt.py', 
                            'dictpred.py', 'prdctr.py', 'prcsng.py', 'rfnt.py', 'dtsnd.py', 'mdlseq.py', 'dse.py', 'trchutl.py', 
                            'trck.py', 'trckpi.py', 'utllscr.py', 'tuls.py', 'bnvgff.py', 'pklp.py', 'trnsfrmkl.py', 'wqsd.py','draw_spots.py'],
    language_level = '3',gdb_debug=True)
)
