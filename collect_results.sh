arr=(
  nb
  hmm-1
  hmm-2
  hmm-3
  maxent
  crf
  lstm-crf
  lstm-crf-cnn
  lstm-crf-lstm
)

arr2=(
  nb
  hmm-1
  hmm-2
  hmm-3
)

# # Gazetteer matching.
# python train_model.py partial_match
# python train_model.py exact_match

for i in "${arr[@]}"
do
  python train_model.py $i -d $1
done

for i in "${arr[@]}"
do
  python train_model.py $i -f -d $1
done

for i in "${arr2[@]}"
do
  python train_model.py $i -f -s -d $1
done
