for f in bla/*; do
  echo "$f"
  python ../tokenize_html.py $f >> joelma.txt
done < unis.txt
