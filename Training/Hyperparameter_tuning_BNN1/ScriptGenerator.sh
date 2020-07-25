for i in {1..10}
do
    cp -v Hyperparameter_tuning_BNN1.py  Hyperparameter_tuning_BNN1_$i.py
    sed -i "s/par_number = 1/par_number = $i/" Hyperparameter_tuning_BNN1_$i.py
    cp -v Hyperparameter_tuning_BNN1.sh Hyperparameter_tuning_BNN1_$i.sh
    sed -i "s/BSUB -J Hyperparameter_tuning_BNN1/BSUB -J Hyperparameter_tuning_BNN1_$i/" Hyperparameter_tuning_BNN1_$i.sh
    sed -i "s/Hyperparameter_tuning_BNN1_1.py/Hyperparameter_tuning_BNN1_$i.py/" Hyperparameter_tuning_BNN1_$i.sh
    sed -i "s/OUT_Hyperparameter_BNN1_1.out/OUT_Hyperparameter_BNN1_$i.out/" Hyperparameter_tuning_BNN1_$i.sh
done


#cp -v pythoncode pythoncode2

#echo hej
#sed -i '' 's/par_number = 1/par_number = 2/' pythoncode

#Please note: On a linux, it is
# sed -i 's/par_number = 1/par_number = 2/' pythoncode
