for i in {1..20}
do
    cp -v Ensembles_118.py  Ensembles_118_$i.py
    sed -i "s/par_number = 1/par_number = $i/" Ensembles_118_$i.py
    cp -v Ensembles_118.sh Ensembles_118_$i.sh
    sed -i "s/BSUB -J Ensembles_118/BSUB -J Ensembles_118_$i/" Ensembles_118_$i.sh
    sed -i "s/Ensembles_118_1.py/Ensembles_118_$i.py/" Ensembles_118_$i.sh
    sed -i "s/BSUB -o Output_1.out/BSUB -o Output_$i.out/" Ensembles_118_$i.sh
    sed -i "s/BSUB -e Error_1.out/BSUB -e Error_$i.out/" Ensembles_118_$i.sh
done
