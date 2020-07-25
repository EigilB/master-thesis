for i in {1..100}
do
    cp -v Hyperparameter_tuning3.py  Hyperparameter_tuning3_$i.py
    sed -i "s/par_number = 1/par_number = $i/" Hyperparameter_tuning3_$i.py
    cp -v Hyperparameter_tuning3.sh Hyperparameter_tuning3_$i.sh
    sed -i "s/BSUB -J Hyperparameter_tuning3/BSUB -J Hyperparameter_tuning3_$i/" Hyperparameter_tuning3_$i.sh
    sed -i "s/Hyperparameter_tuning3_1.py/Hyperparameter_tuning3_$i.py/" Hyperparameter_tuning3_$i.sh
done
