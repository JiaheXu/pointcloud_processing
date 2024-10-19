for i in $(seq 1 35);
do
    python3 icp.py -t fixed_rack -d $i
done
