for i in $(seq -f "%03g" 1 149)
do
  echo $i
  python tokenizer.py htmls/$i.html target_names/target_names_$i.txt names.txt > dataset/$i.txt
done
