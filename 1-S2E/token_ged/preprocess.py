import argparse
from gecommon import Parallel
import json
from tqdm import tqdm

def main(args):
    gec = None
    if args.m2:
        gec = Parallel.from_m2(
            args.m2,
            ref_id=args.ref_id
        )
    else:
        assert args.src is not None and args.trg is not None
        gec = Parallel.from_parallel(
            args.src, args.trg
        )
    assert gec is not None
    labels = gec.ged_labels_token(mode=args.mode, return_id=True)
    assert len(gec.srcs) == len(labels)
    dataset = dict()
    for i, (src, label) in tqdm(enumerate(zip(gec.srcs, labels)), total=len(labels)):
        dataset[f'Line{i}'] = {'src': src, 'label': label}
    data = dict()
    data['dataset'] = dataset
    data['id2label'] = gec.get_ged_id2label(mode=args.mode)
    data['label2id'] = gec.get_ged_label2id(mode=args.mode)
    data['num_labels'] = len(data['label2id'])
    with open(args.out, 'w') as f:
        json.dump(data, f, indent=4)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src')
    parser.add_argument('--trg')
    parser.add_argument('--m2')
    parser.add_argument('--ref_id', type=int, default=0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', default='bin', choices=['bin', 'cat1', 'cat2', 'cat3'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)