for i in {1..100}
do
    cp -v HypTun_Big2.py  HypTun_Big2_$i.py
    sed -i "s/par_number = 1/par_number = $i/" HypTun_Big2_$i.py
    cp -v HypTun_Big2.sh HypTun_Big2_$i.sh
    sed -i "s/BSUB -J HypTun_Big2/BSUB -J HypTun_Big2_$i/" HypTun_Big2_$i.sh
    sed -i "s/HypTun_Big2_1.py/HypTun_Big2_$i.py/" HypTun_Big2_$i.sh
#    sed -i "s/OUT_HypTun_Big2_1.out/OUT_HypTun_Big2_$i.out/" HypTun_Big2_$i.sh
done
