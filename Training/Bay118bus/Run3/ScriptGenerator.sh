for i in {1..1000}
do
    cp -v Bay118bus.py  Bay118bus_$i.py
    sed -i "s/par_number = 1/par_number = $i/" Bay118bus_$i.py
    cp -v Bay118bus.sh Bay118bus_$i.sh
    sed -i "s/BSUB -J Bay118bus/BSUB -J Bay118bus_$i/" Bay118bus_$i.sh
    sed -i "s/Bay118bus_1.py/Bay118bus_$i.py/" Bay118bus_$i.sh
    sed -i "s/BSUB -o Output_1.out/BSUB -o Output_$i.out/" Bay118bus_$i.sh
    sed -i "s/BSUB -e Error_1.out/BSUB -e Error_$i.out/" Bay118bus_$i.sh
done
