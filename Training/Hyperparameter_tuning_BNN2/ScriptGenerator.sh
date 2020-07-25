for i in {1..100}
do
    cp -v Hyperparameter_tuning_BNN2.py  Hyperparameter_tuning_BNN2_$i.py
    sed -i "s/par_number = 1/par_number = $i/" Hyperparameter_tuning_BNN2_$i.py
    cp -v Hyperparameter_tuning_BNN2.sh Hyperparameter_tuning_BNN2_$i.sh
    sed -i "s/BSUB -J Hyperparameter_tuning_BNN2/BSUB -J Hyperparameter_tuning_BNN2_$i/" Hyperparameter_tuning_BNN2_$i.sh
    sed -i "s/Hyperparameter_tuning_BNN2_1.py/Hyperparameter_tuning_BNN2_$i.py/" Hyperparameter_tuning_BNN2_$i.sh
done
