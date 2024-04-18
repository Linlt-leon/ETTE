hyp_file=/hy-tmp/CS4248_Project/scripts/model_t5/template-2class/checkpoint-1000/results_test/predictions.txt

gold="/hy-tmp/CS4248_Project/dataWithPredictions/datasets_binary/test.m2"

python ./m2scorer_master/scripts/m2scorer.py $hyp_file $gold \
    | tee ${hyp_file}.score