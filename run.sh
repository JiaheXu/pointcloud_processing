for i in $(seq 1 35);
do
    python3 3docp_preprocessing.py -t fixed_rack -d $i
done
