for i in {1..10}
do
    cp -v Hyperparameter_tuning2.py  Hyperparameter_tuning2_$i.py
    sed -i "s/par_number = 1/par_number = $i/" Hyperparameter_tuning2_$i.py
    cp -v Hyperparameter_tuning2.sh Hyperparameter_tuning2_$i.sh
    sed -i "s/BSUB -J Hyperparameter_tuning2/BSUB -J Hyperparameter_tuning2_$i/" Hyperparameter_tuning2_$i.sh
    sed -i "s/Hyperparameter_tuning2_1.py/Hyperparameter_tuning2_$i.py/" Hyperparameter_tuning2_$i.sh
    sed -i "s/OUT_Hyperparameter_tuning2_1.out/OUT_Hyperparameter_tuning2_$i.out/" Hyperparameter_tuning2_$i.sh
done


#cp -v pythoncode pythoncode2

#echo hej
#sed -i '' 's/par_number = 1/par_number = 2/' pythoncode

#Please note: On a linux, it is
# sed -i 's/par_number = 1/par_number = 2/' pythoncode
