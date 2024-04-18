from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import argparse
from dataset import generate_dataset_for_inference
from torch.utils.data import DataLoader
import torch

def predict(
    srcs,
    model,
    tokenizer,
    return_id=False,
    batch_size=32
):
    dataset = generate_dataset_for_inference(
        srcs=srcs,
        tokenizer=tokenizer,
        max_len=128
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    predictions = []
    if torch.cuda.is_available():
        model.cuda()
    for batch in loader:
        if torch.cuda.is_available():
            batch = {k:v.cuda() for k, v in batch.items()}
        logits = model(**batch).logits
        # (batch, seq_len, num_labels) -> (batch, seq_len, 1)
        pred_labels = torch.argmax(logits, dim=-1)
        # Convert the subword-level label into word-level label
        # batch['labels'] is dummy labels to know the subword masking
        # batch['labels'] != 100 means indices of the first position of each subword
        for temp_label, pred_label in zip(batch['labels'], pred_labels):
            word_level_pred_label = pred_label[temp_label != -100].tolist()
            if not return_id:
                id2label = model.config.id2label
                word_level_pred_label = [id2label[l] for l in word_level_pred_label]
            predictions.append(word_level_pred_label)
    return predictions

def main(args):
    model = AutoModelForTokenClassification.from_pretrained(args.restore_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    if args.demo_input:
        srcs = ['This are wrong sentece .', 'This is correct .']
    else:
        srcs = open(args.input).read().rstrip().split('\n')
    results = predict(
        srcs=srcs,
        model=model,
        tokenizer=tokenizer,
        return_id=args.return_id,
        batch_size=args.batch_size
    )
    with open(args.output, 'w') as file:
        file.write('\n'.join([' '.join(labels) for labels in results]))
    print(results[:100])
    
    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--demo_input', action='store_true')
    parser.add_argument('--restore_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--return_id', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)