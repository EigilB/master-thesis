for i in {1..100}
do
    cp -v HypTun_Big_BNN2.py  HypTun_Big_BNN2_$i.py
    sed -i "s/par_number = 1/par_number = $i/" HypTun_Big_BNN2_$i.py
    cp -v HypTun_Big_BNN2.sh HypTun_Big_BNN2_$i.sh
    sed -i "s/BSUB -J HypTun_Big_BNN2/BSUB -J HypTun_Big_BNN2_$i/" HypTun_Big_BNN2_$i.sh
    sed -i "s/HypTun_Big_BNN2_1.py/HypTun_Big_BNN2_$i.py/" HypTun_Big_BNN2_$i.sh
done
