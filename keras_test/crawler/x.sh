count=1393
while read p; do
  echo "$p"
  count=$((count+1))
  curl $p > htmls/$count.html
done < unis.txt
