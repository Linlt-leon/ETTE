# Token-level grammatical error detection

This implementation supports binary or multi-class grammatical error detection.

# Preprocessing
The input can be either parallel or M2 format.

```sh
python preprocess.py \
    --src <path to a source file> \
    --trg <path to a target file> \
    --out <output path> \
    --mode bin

python preprocess.py \
    --m2 <path to M2 file> \
    --ref_id <integer of the number of reference> \
    --out <output path> \
    --mode bin
```

You can use `--mode` option to specify the type of detection labels.
- `--mode bin` provides 2-class labels, correct and incorrect. (default)
- `--mode cat1` provides 4-class labels, correct, replacement, missing, and unnecessary.
- `--mode cat2` provides 25-class labels, correct and 24 labels without UNK of ERRANT's definition.
- `--mode cat3` provides 55-class labels, correct and 54 labels like `M:NOUN`. Refer to Appendix A in the [Bryant+ 17](https://aclanthology.org/P17-1074.pdf)

The output is a JSON file. The format is the following if `--mode bin`:
```json
{
    "id2label": {
        "0": "CORRECT",
        "1": "INCORRECT"
    },
    "label2id": {
        "CORRECT": 0,
        "INCORRECT": 1
    },
    "num_labels": 2,
    "dataset": {
        "Line 0": {
            "src": "It 's difficult answer at ...",
            "label": [
                0,
                0,
                0,
                1,
                1,
                ...
            ]
        },
    }
}
```

# Train
Use Accelerate to support distributed training.
```sh
accelerate launch train.py \
    --model_id bert-large-cased \
    --train <training JSON file> \
    --valid <validation JSON file> \
    --epochs 5 \
    --outdir models/sample
```

The construction of the `--outdir` is like this:
```
models/sample
├── best
│   ├── config.json
│   ├── model.safetensors
│   ├── score.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── training_state.json
│   └── vocab.txt
└── log.json
```

The `--outdir` directory will be created automatically. The output includes best and last checkpoints.  
The best checkpoints is determined
- by the F0.5 for an incorrect label in the binary setting.
- by the macro F0.5 otherwise (e.g. 4-class setting).

# Inference

### API
```py
from predict import predict
from transformers import AutoModelForTokenClassification, AutoTokenizer
restore_dir = 'gotutiyan/token-ged-electra-large-bin'
model = AutoModelForTokenClassification.from_pretrained(restore_dir)
tokenizer = AutoTokenizer.from_pretrained(restore_dir)
srcs = ['This are wrong sentece .', 'This is correct .']

# predict() returns word-level error detection labels
# If return_id=True
results = predict(
    srcs=srcs,
    model=model,
    tokenizer=tokenizer,
    return_id=True,
    batch_size=32
)
print(results)
# An example of outputs: [[0, 1, 0, 1, 0], [0, 0, 0, 0]]

# If return_id=False
results = predict(
    srcs=srcs,
    model=model,
    tokenizer=tokenizer,
    return_id=False,
    batch_size=32
)
print(results)
# An exmaple of outputs:
# [['CORRECT', 'INCORRECT', 'CORRECT', 'INCORRECT', 'CORRECT'],
#  ['CORRECT', 'CORRECT', 'CORRECT', 'CORRECT']]
```

# Evaluate
To calculate precision, recall, and F0.5 score, you can use `evaluate.py`.

First, use `preprocess.py` to prepare labels of evaluation data. You can input data as both M2 or parallel format.  
If the data has multiple reference, you can use `--ref_id` option to specify which annotation will be used.

Second, execute `evaluate.py`.
```sh
python evaluate.py \
    --test <test JSON file> \
    --restore_dir <trained model to be evaluated>
```

You can see both binarised score and multi-class score.

# Trained models
I trained token-level GED models using this implementation.  
Like [Yuan+ 21](https://aclanthology.org/2021.emnlp-main.687/), large models of BERT, XLNet, and ELECTRA were used.  
I also used FCE-train for the training data and BEA19-dev for the validation data.  

# Binary error detection performance
This corresponds to Table 2 in [Yuan+ 21](https://aclanthology.org/2021.emnlp-main.687/).  
All of scores are shown in `(Precision/Recall/F0.5)` format.
|Model|BEA19-dev|FCE-test|CoNLL14 test 1|CoNLL14 test 2|
|:--|:-:|:-:|:-:|:-:|
|[Yuan+ 21](https://aclanthology.org/2021.emnlp-main.687/), BERT|(65.48 / 42.85 / 59.23)|(75.73 / 47.98 / 67.88)|(49.73 / 34.23 / 45.60)|(64.52 / 32.33 / 53.80)|
|[Yuan+ 21](https://aclanthology.org/2021.emnlp-main.687/), XLNet|(70.03 / 45.55 / 63.23)|(77.50 / 49.81 / 69.75)|(53.23 / 36.17 / 48.64)|(70.68 / 34.95 / 58.68)|
|[Yuan+ 21](https://aclanthology.org/2021.emnlp-main.687/), ELECTRA|(72.81 / 46.85 / 65.54)|(82.05 / 50.49 / 72.93)|(55.15 / 39.78 / 51.19)|(76.44 / 40.13 / 64.73)|
|[gotutiyan/token-ged-bert-large-cased-bin](https://huggingface.co/gotutiyan/token-ged-bert-large-cased-bin)| (67.88 / 39.3 / 59.26) | (75.97 / 44.82 / 66.7) | (52.13 / 31.35 / 46.03) | (64.36 / 29.37 / 51.97) |
|[gotutiyan/token-ged-xlnet-large-cased-bin](https://huggingface.co/gotutiyan/token-ged-xlnet-large-cased-bin)| (72.38 / 40.99 / 62.77) | (77.72 / 46.74 / 68.63) | (54.41 / 34.22 / 48.67) | (67.25 / 32.11 / 55.17) |
|[gotutiyan/token-ged-electra-large-bin](https://huggingface.co/gotutiyan/token-ged-electra-large-bin)| (74.18 / 48.5 / 67.08) | (82.37 / 55.69 / 75.17) | (55.04 / 42.79 / 52.06) | (72.84 / 42.97 / 63.95) |

# Binary and multi-class error detection performance
This corresponds to Table 3 in [Yuan+ 21](https://aclanthology.org/2021.emnlp-main.687/).  
The format in each cell is `(binarised F0.5 / macro F0.5)`.
|Mode|BEA19-dev|FCE-test|
|:--|:-:|:-:|
|[Yuan+ 21](https://aclanthology.org/2021.emnlp-main.687/), binary|(65.54 / 80.39)|(72.93 / 83.54)|
|[Yuan+ 21](https://aclanthology.org/2021.emnlp-main.687/), 4-class|(66.10 / 67.07)|(72.57 / 70.95)|
|[Yuan+ 21](https://aclanthology.org/2021.emnlp-main.687/), 25-class|(63.08 / 47.28)|(72.08 / 54.59)|
|[Yuan+ 21](https://aclanthology.org/2021.emnlp-main.687/), 55-class|(65.81 / 32.99)|(73.85 / 34.88)|
|[gotutiyan/token-ged-electra-large-bin](https://huggingface.co/gotutiyan/token-ged-electra-large-bin)| (67.07 / 81.97) | (75.16 / 85.72) |
|[gotutiyan/token-ged-electra-large-4cls](https://huggingface.co/gotutiyan/token-ged-electra-large-4cls)|(65.91 / 67.14)|(73.58 / 71.37)|
|[gotutiyan/token-ged-electra-large-25cls](https://huggingface.co/gotutiyan/token-ged-electra-large-25cls)|(65.04 / 44.33)|(72.59 / 46.73)|
|[gotutiyan/token-ged-electra-large-55cls](https://huggingface.co/gotutiyan/token-ged-electra-large-55cls)|(63.85 / 39.26)|(72.49 / 44.83)|