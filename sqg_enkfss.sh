local1=2100
export exptname="lgetkfcv_${local1}_16mem16grps_4096obs"  
python -u sqg_lgetkf_cv.py ${local1}.e3 >& ${exptname}.out
local1=1900
export exptname="lgetkfcv_${local1}_16mem16grps_4096obs"  
python -u sqg_lgetkf_cv.py ${local1}.e3 >& ${exptname}.out
local1=2200
export exptname="lgetkfcv_${local1}_16mem16grps_4096obs"  
python -u sqg_lgetkf_cv.py ${local1}.e3 >& ${exptname}.out
