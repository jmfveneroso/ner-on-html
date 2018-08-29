for i in $(seq -f "%03g" 1 149)
do
  echo $i
  python tokenizer.py $i > dataset/$i.txt
done
