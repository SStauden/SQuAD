
cd InferSent/dataset
./get_data.bash

cd ..
curl -Lo encoder/infersent.allnli.pickle https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle

echo "import nltk; nltk.download('punkt')" | python 