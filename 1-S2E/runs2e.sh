!# /bin/bash

mkdir datasets_binary

python3 removeNonUtf.py

cp ./m2/ABCN.dev.gold.bea19.m2 ./datasets_binary/valid.m2

python3 randomM2.py --m2 ./m2/ABC.train.gold.bea19.m2 --percent 20

cp -r ./datasets_binary ./datasets_multi

python3 preprocessTrainValid.py --dir ./datasets_binary --outDir ./datasets_binary/preProcessed --mode bin
python3 preprocessTrainValid.py --dir ./datasets_multi --outDir ./datasets_multi/preProcessed --mode cat1

echo 'Preprocessed the files'

python3 toTextTrainValidTest.py

accelerate launch --num_processes 8 ./token_ged/train.py --train_input ./datasets_binary/preProcessed/train.json --valid_input ./datasets_binary/preProcessed/valid.json --output ./models/binary/ --epochs 20
accelerate launch --num_processes 8 ./token_ged/train.py --train_input ./datasets_multi/preProcessed/train.json --valid_input ./datasets_multi/preProcessed/valid.json --output ./models/multi/ --epochs 20

echo 'Training Complete'
echo 'Plotting losses against epoch'
python3 plotResults.py

echo 'Running evaluations for S2E for both models'

mkdir evaluations

accelerate launch --num_processes 4 ./token_ged/evaluate.py --restore_dir ./models/binary/best/ --test_json ./datasets_binary/preProcessed/test.json > ./evaluations/binEval.txt
accelerate launch --num_processes 4 ./token_ged/evaluate.py --restore_dir ./models/multi/best/ --test_json ./datasets_multi/preProcessed/test.json > ./evaluations/multiEval.txt

echo 'Running predictions on all training, validation and test files'

python3 predictTrainTestValid.py

echo 'Running predictions for actual test file'

mkdir actualTestPredictions

accelerate launch --num_processes 8 ./token_ged/predict.py --input ./test/ABCN.test.bea19.orig --restore_dir ./models/binary/best/ --output ./actualTestPredictions/binTestPred.txt
accelerate launch --num_processes 8 ./token_ged/predict.py --input ./test/ABCN.test.bea19.orig --restore_dir ./models/multi/best/ --output ./actualTestPredictions/multiTestPred.txt

echo 'Done!'