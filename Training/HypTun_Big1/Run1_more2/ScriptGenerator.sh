for i in {1..100}
do
    cp -v HypTun_Big1.py  HypTun_Big1_$i.py
    sed -i "s/par_number = 1/par_number = $i/" HypTun_Big1_$i.py
    cp -v HypTun_Big1.sh HypTun_Big1_$i.sh
    sed -i "s/BSUB -J HypTun_Big1/BSUB -J HypTun_Big1_$i/" HypTun_Big1_$i.sh
    sed -i "s/HypTun_Big1_1.py/HypTun_Big1_$i.py/" HypTun_Big1_$i.sh
#    sed -i "s/OUT_HypTun_Big1_1.out/OUT_HypTun_Big1_$i.out/" HypTun_Big1_$i.sh
done
