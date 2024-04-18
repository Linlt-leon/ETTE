# import pandas as pd
# import argparse
import json
from tqdm import tqdm

# preprocPath = '/home/ano/CS4248/Project/GED_baseline/ged_baselines/token_ged/out/0.20ABCtrain.preprocess'
def c2pairTxt(jsonPath,jsonName):
    # jsonPath = '/home/ano/CS4248/Project/Corpus/wi+locness/sentence_pairV2/'
    # jsonName = 'ABC.train.gold.bea19.json'
    with open(jsonPath + jsonName) as f:
        data = json.load(f)
        jsonName
    srclst = []
    trglst = []
    # for s in range(len(data['dataset'])):
    with open('.'.join((jsonPath + jsonName).split('.')[:-1]) + ".txt", 'w' ) as f:
        for s in tqdm(range(len(data)), total=len(data)):

            # srclst.append(data['src'])
            # trglst.append(data['tgt'])
            f.write(f"{data[s]['src']}\t{data[s]['tgt']}")
            f.write('\n')

# Example usage
# c2pairTxt('/home/ano/CS4248/Project/Corpus/wi+locness/sentence_pairV2/','ABC.train.gold.bea19.json')


def dup2pairTxt(testPath, testName):
    with open('.'.join((testPath + testName).split('.')[:-1]) +  ".pairtxt", 'w' ) as fw:
        with open(testPath + testName, 'r') as f:
            for line in f:
                line = line.strip()
                fw.write(f"{line}\t{line}\n")
                
dup2pairTxt('/home/ano/CS4248/Project/Corpus/wi+locness/test/', 'ABCN.test.bea19.orig')            