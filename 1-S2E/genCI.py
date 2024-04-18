import json
from tqdm import tqdm

with open('/home/ano/CS4248/Project/GED_baseline/ged_baselines/token_ged/out/0.20ABCtrain.preprocess') as f:
    data = json.load(f)
    
srclst = []
lablst = []
for s in range(len(data['dataset'])):

    strl = 'Line' + str(s)
    srclst.append(data['dataset'][strl]['src'])

    labels = data['dataset'][strl]['label']
    for l in range(len(labels)):
        labels[l] = 'CORRECT' if labels[l] == 0 else 'INCORRECT'

    lablst.append(labels)
with open('./0.20ABCtrain.src.lst', 'w' ) as f:
    f.write(f"{srclst}")
    f.write('\n')
with open('./0.20ABCtrain.gold.lst', 'w' ) as f:    
    f.write(f"{lablst}")
    f.write('\n')


data = dict()
dataset = dict()
for i, (src, label) in tqdm(enumerate(zip(srclst, lablst)), total=len(lablst)):
    dataset[f'Line{i}'] = {'src': src, 'label': label}
data['/0.20ABCtrain'] = dataset

with open('./0.20ABCtrain.json', 'w') as f:
    json.dump(data, f, indent=4)
