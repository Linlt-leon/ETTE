from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import argparse
from predict import predict
import json
from sklearn.metrics import classification_report
from transformers import pipeline
from collections import Counter

def load_json(file_path):
    data = json.load(open(file_path))
    srcs = [d['src'] for d in data['dataset'].values()]
    labels = [d['label'] for d in data['dataset'].values()]
    id2label = data['id2label']
    id2label = {int(k):v for k,v in id2label.items()}
    return srcs, labels, id2label

def binarised_labels(labels):
    return ['C' if l == 'C' else 'I' for l in labels]

def main(args):
    model = AutoModelForTokenClassification.from_pretrained(args.restore_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    srcs, labels, id2label = load_json(args.test_json)
    results = predict(
        srcs=srcs,
        model=model,
        tokenizer=tokenizer,
        return_id=False,
        batch_size=args.batch_size
    )
    # Sometimes the label for truncated tokens is missing due to max_length.
    # To solve this, we add a 'C' label for the truncated token.
    # The reason why we choose 'C' is that it is the majority label.
    for i in range(len(srcs)):
        diff = len(labels[i]) - len(results[i])
        results[i] += ['C'] * diff

    # flatten
    gold_labels = [id2label[l] for label in labels for l in label]
    pred_labels = [l for label in results for l in label]
    # get binarized labels
    bin_gold_labels = binarised_labels(gold_labels)
    bin_pred_labels = binarised_labels(pred_labels)

    # print(Counter(gold_labels))
    # print(Counter(pred_labels))
    # print(Counter(bin_gold_labels))
    # print(Counter(bin_pred_labels))

    def f05(p, r, beta=0.5):
        try:
            return ((1+beta**2) * p * r) / ((beta**2)*p + r)
        except ZeroDivisionError:
            return 0

    print('=== Binarized score ===')
    results = classification_report(y_true=bin_gold_labels, y_pred=bin_pred_labels, output_dict=True)
    print(results)
    # p = results['I']['precision']
    # r = results['I']['recall']
    # print(f'Precision: {p}\nRecall: {r}\nF0.5: {f05(p, r)}')

    print('\n=== Multi-class score ===')
    results = classification_report(y_true=gold_labels, y_pred=pred_labels, output_dict=True)
    print(results)
    # p = results['macro avg']['precision']
    # r = results['macro avg']['recall']
    # print(f'Precision: {p}\nRecall: {r}\nF0.5: {f05(p, r)}')


    # print(classification_report(y_true=bin_gold_labels, y_pred=bin_pred_labels, output_dict=False))
    # print(classification_report(y_true=gold_labels, y_pred=pred_labels, output_dict=False))
    
    
    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_json')
    parser.add_argument('--restore_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)