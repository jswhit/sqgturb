corrl=1500
while [ $corrl -le 3000 ]; do
   echo $corrl
   /bin/cp -f sqg_enkf.sh.template sqg_enkf.sh
   sed -i -e "s/<corrl>/${corrl}/g" sqg_enkf.sh
   sbatch sqg_enkf.sh
   corrl=$((corrl+100))
done
