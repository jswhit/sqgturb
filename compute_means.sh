corrl=1500
while [ $corrl -le 3000 ]; do
   echo  $corrl
   python compute_means.py sqg_zloc64_6hrly_rtps0p4_${corrl}.out 100 >> sqg_zloc64_6hrly_rtps0p4.out $corrl
   corrl=$((corrl+100))
done
