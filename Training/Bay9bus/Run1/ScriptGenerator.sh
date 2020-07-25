for i in {1..100}
do
    cp -v Bay9bus.py  Bay9bus_$i.py
    sed -i "s/par_number = 1/par_number = $i/" Bay9bus_$i.py
    cp -v Bay9bus.sh Bay9bus_$i.sh
    sed -i "s/BSUB -J Bay9bus/BSUB -J Bay9bus_$i/" Bay9bus_$i.sh
    sed -i "s/Bay9bus_1.py/Bay9bus_$i.py/" Bay9bus_$i.sh
done
