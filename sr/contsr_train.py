import rrdbnet_arch
import torch
from dataset import *
from utils import *
import json
import argparse
import contsr_model
from collections import namedtuple





def main():
    
    from types import SimpleNamespace
    
    with open('config_cont.json') as json_file:
        json_data = json.load(json_file)

    
    args = namedtuple("object", json_data.keys())(*json_data.values())

    with open('config_cont2.json') as json_file:
        json_data = json.load(json_file)

    
    args2 = namedtuple("object", json_data.keys())(*json_data.values())



    gpu = args.gpu
    print("GPU:",gpu)
  
    #os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    
    model = contsr_model.ContSR(args, args2)
    model.build()

    if args.phase == "train":    
        model.load()
        model.train()
    elif args.phase == 'test':
        model.load()
        model.inference()


if __name__ == '__main__':
    main()