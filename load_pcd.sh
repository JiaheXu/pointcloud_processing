for i in $(seq 1 35);
do
    python3 load_object_pcd.py -t fixed_rack -d $i
done
