local1=4200
local2=1400
ncutoff=20
crossbandcov=0.16
export exptname="lgetkfcvms_${local1}_${local2}_${ncutoff}_crosscov${crossbandcov}_16mem16grps_dek25_768obs"  
python -u sqg_lgetkf_cvms.py "[${local1}.e3,${local2}.e3]" "[${ncutoff}]" "[${crossbandcov}]" >& ${exptname}.out
local1=4200
local2=1400
ncutoff=24
crossbandcov=0.16
export exptname="lgetkfcvms_${local1}_${local2}_${ncutoff}_crosscov${crossbandcov}_16mem16grps_dek25_768obs"  
python -u sqg_lgetkf_cvms.py "[${local1}.e3,${local2}.e3]" "[${ncutoff}]" "[${crossbandcov}]" >& ${exptname}.out
