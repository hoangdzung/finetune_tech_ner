import os 
import sys
import argparse
from tqdm import tqdm 
import json 

def process_patent(datadir, outfile):
    with open(outfile, 'w') as f:
        for i in os.listdir(datadir):
            for line in tqdm(open(os.path.join(datadir,i)), desc='Process {}'.format(i)):
                data_dict = json.loads(line)
                f.write(data_dict['abstract']+'\n')

parser = argparse.ArgumentParser()
parser.add_argument('--source')
parser.add_argument('--datadir')
parser.add_argument('--out')
args = parser.parse_args()

if args.source == 'patent':
    process_patent(args.datadir, args.out)   
else:
    raise NotImplementedError()