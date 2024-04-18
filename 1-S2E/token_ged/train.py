
import argparse
from transformers import AutoTokenizer, get_scheduler, AutoModelForTokenClassification
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from collections import OrderedDict
import json
from accelerate import Accelerator
import numpy as np
import random
from dataset import generate_dataset
from sklearn.metrics import classification_report
import json

def f05(p, r):
    try: 
        return 1.25 * (p * r) / (0.25*p + r)
    except ZeroDivisionError:
        return 0

def train(
    model,
    loader: DataLoader,
    optimizer,
    epoch: int,
    accelerator: Accelerator,
    lr_scheduler
) -> float:
    model.train()
    log = {
        'loss': 0
    }
    with tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_main_process) as pbar:
        for _, batch in pbar:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                log['loss'] += loss.item()
                if accelerator.is_main_process:
                    pbar.set_description(f'[Epoch {epoch}] [TRAIN]')
                    pbar.set_postfix(OrderedDict(
                        loss=loss.item(),
                        lr=optimizer.optimizer.param_groups[0]['lr']
                    ))
    return {k: v/len(loader) for k, v in log.items()}

def valid(model,
    loader: DataLoader,
    epoch: int,
    accelerator: Accelerator
) -> float:
    model.eval()
    loss = 0
    pred_labels = []
    gold_labels = []
    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_main_process) as pbar:
            for _, batch in pbar:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss += outputs.loss.item()
                    plabels = torch.argmax(outputs.logits, dim=-1).view(-1)
                    glabels = batch['labels'].view(-1)
                    pred_labels += plabels[glabels != -100].tolist()
                    gold_labels += glabels[glabels != -100].tolist()
                    if accelerator.is_main_process:
                        pbar.set_description(f'[Epoch {epoch}] [VALID]')
                        pbar.set_postfix(OrderedDict(
                            loss=outputs.loss.item()
                        ))
    cr_result = classification_report(
        y_true=gold_labels,
        y_pred=pred_labels,
        output_dict=True
    )
    log = dict()
    log['loss'] = loss / len(loader)
    log['cls_report'] = cr_result
    # Add f0.5 scores
    for k in ['0', '1', 'macro avg', 'macro avg', 'weighted avg']:
        log['cls_report'][k]['f05-score'] = f05(
            log['cls_report'][k]['precision'],
            log['cls_report'][k]['recall'],
        )
    return log

def main(args):
    config = json.load(open(os.path.join(args.restore_dir, 'training_state.json'))) if args.restore_dir else {'argparse': dict()}
    current_epoch = config.get('current_epoch', -1) + 1
    max_f05 = config.get('max_f05', 0)
    seed = config['argparse'].get('seed', args.seed)
    max_len = config['argparse'].get('max_len', args.max_len)
    log_dict = json.load(open(os.path.join(args.restore_dir, '../log.json'))) if args.restore_dir else dict()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    if args.restore_dir is not None:
        model = AutoModelForTokenClassification.from_pretrained(args.restore_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    else:
        d = json.load(open(args.train_input))
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_id,
            id2label=d['id2label'],
            label2id=d['label2id']
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, add_prefix_space=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    train_dataset = generate_dataset(
        input_file=args.train_input,
        tokenizer=tokenizer,
        max_len=max_len
    )
    valid_dataset = generate_dataset(
        input_file=args.valid_input,
        tokenizer=tokenizer,
        max_len=max_len
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = None
    if args.test_input is not None:
        test_dataset = generate_dataset(
            input_file=args.test_input,
            tokenizer=tokenizer,
            max_len=max_len
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=2, pin_memory=True
        )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.accumulation,
        num_training_steps=len(train_loader) * args.epochs,
    )
    best_path = os.path.join(args.outdir, 'best')
    last_path = os.path.join(args.outdir, 'last')
    os.makedirs(best_path, exist_ok=True)
    os.makedirs(last_path, exist_ok=True)
    tokenizer.save_pretrained(best_path)
    tokenizer.save_pretrained(last_path)
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation)
    model, optimizer, train_loader, valid_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, test_loader, lr_scheduler
    )
    accelerator.wait_for_everyone()
    for epoch in range(current_epoch, args.epochs):
        train_log = train(model, train_loader, optimizer, epoch, accelerator, lr_scheduler)
        valid_log = valid(model, valid_loader, epoch, accelerator)
        if len(model.config.id2label) == 2:
            # If binary setting
            valid_log['cls_report'] = valid_log['cls_report']['1']
        else:
            # otherwise (e.g. 25-class)
            valid_log['cls_report'] = valid_log['cls_report']['macro avg']
        log_dict[f'Epoch {epoch}'] = {
            'train_log': train_log,
            'valid_log': valid_log
        }
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            score = valid_log['cls_report']['f05-score'] 
            if max_f05 < score:
                # Save the best chckpoint
                accelerator.unwrap_model(model).save_pretrained(best_path)
                max_f05 = score
                training_state = {
                    'current_epoch': epoch,
                    'max_f05': max_f05,
                    'argparse': args.__dict__
                }
                with open(os.path.join(best_path, 'training_state.json'), 'w') as fp:
                    json.dump(training_state, fp, indent=4)
                if test_loader is not None:
                    test_log = valid(
                        model, test_loader, epoch, accelerator
                    )
                    if len(model.config.id2label) == 2:
                        log_dict[f'Epoch {epoch}'][f'test_log'] = test_log['cls_report']['1']
                    else:
                        log_dict[f'Epoch {epoch}'][f'test_log'] = test_log['cls_report']['macro avg']
            # Save checkpoint as the last checkpoint in each epoch
            # accelerator.unwrap_model(model).save_pretrained(last_path)
            # training_state = {
            #     'current_epoch': epoch,
            #     'max_f05': max_f05,
            #     'argparse': args.__dict__
            # }
            # with open(os.path.join(last_path, 'training_state.json'), 'w') as fp:
            #         json.dump(training_state, fp, indent=4)
            with open(os.path.join(args.outdir, 'log.json'), 'w') as fp:
                json.dump(log_dict, fp, indent=4)
    print('Finish')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input', required=True)
    parser.add_argument('--valid_input', required=True)
    parser.add_argument('--test_input')
    parser.add_argument('--model_id', default='bert-base-cased')
    parser.add_argument('--outdir', default='models/sample/')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--restore_dir', default=None)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)