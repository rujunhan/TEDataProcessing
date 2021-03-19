from pathlib import Path
import pickle
import argparse
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
from typing import Iterator, List, Mapping, Union, Optional, Set
import random
from load_data import FlatRelation, read_relations, matres_label_map, tbd_label_map
from processTBD import TBDDoc, TBDRelation, TBDEntity
from dataclasses import dataclass
from featureFuncs import *
import multiprocessing as mp
from functools import partial
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
import logging as log
# from pytorch_pretrained_bert.modeling import BertModel, BertConfig, PreTrainedBertModel
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from sklearn.utils import resample
import os
import torch
from torch.utils import data
import time

@dataclass
class Event():
    id: str
    type: str
    text: str
    tense: str
    polarity: str
    span: (int, int)

def main(args):

    data_dir = args.data_dir
    opt_args = {}

    if args.nr > 0:
        opt_args['neg_rate'] = args.nr
        opt_args['eval_list'] = args.eval_list

    opt_args['data_type'] = args.data_type
    opt_args['pred_window'] = args.pred_win
    opt_args['shuffle_all'] = args.shuffle_all
    opt_args['backward_sample'] = args.backward_sample
    opt_args['joint'] = args.joint
    log.info(f"Reading datasets to memory from {data_dir}")

    train_data = list(read_relations(Path(data_dir), 'train', **opt_args))

    train_out = {}
    for ex in train_data:
        left_event = Event(ex.left['id'], ex.left['type'], ex.left['text'], ex.left['tense'],
                           ex.left['polarity'], (ex.left['span'][0], ex.left['span'][1]))
        right_event = Event(ex.right['id'], ex.right['type'], ex.right['text'], ex.right['tense'],
                           ex.right['polarity'], (ex.right['span'][0], ex.right['span'][1]))
        train_out[ex.id] = {'rel_type': ex.rel_type,
                            'rev': ex.rev,
                            'doc_dictionary':ex.doc['pos_dict'],
                            'event_labels': ex.doc['entity_labels'],
                            'left_event': left_event,
                            'right_event': right_event,
                            'doc_id': ex.doc_id}

    with open(args.save_dir + '/train.pickle', 'wb') as handle:
        pickle.dump(train_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    
    dev_data = []
    if args.data_type in ["tbd"]:
        opt_args['neg_rate'] = 0.0 # a random large number to include all                                                
        dev_data = list(read_relations(Path(data_dir), 'dev', **opt_args))
        dev_out = {}
        for ex in dev_data:
            left_event = Event(ex.left['id'], ex.left['type'], ex.left['text'], ex.left['tense'],
                               ex.left['polarity'], (ex.left['span'][0], ex.left['span'][1]))
            right_event = Event(ex.right['id'], ex.right['type'], ex.right['text'], ex.right['tense'],
                                ex.right['polarity'], (ex.right['span'][0], ex.right['span'][1]))
            dev_out[ex.id] = {'rel_type': ex.rel_type,
                              'doc_dictionary':ex.doc['pos_dict'],
                              'rev': ex.rev,
                              'event_labels': ex.doc['entity_labels'],
                              'left_event': left_event,
                              'right_event': right_event,
                              'doc_id': ex.doc_id}

        with open(args.save_dir + '/dev.pickle', 'wb') as handle:
            pickle.dump(dev_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
    
    test_data = list(read_relations(Path(data_dir), 'test', **opt_args))
    test_out = {}
    for ex in test_data:
        left_event = Event(ex.left['id'], ex.left['type'], ex.left['text'], ex.left['tense'],
                           ex.left['polarity'], (ex.left['span'][0], ex.left['span'][1]))
        right_event = Event(ex.right['id'], ex.right['type'], ex.right['text'], ex.right['tense'],
                            ex.right['polarity'], (ex.right['span'][0], ex.right['span'][1]))
        test_out[ex.id] = {'rel_type': ex.rel_type,
                           'rev': ex.rev,
                           'doc_dictionary':ex.doc['pos_dict'],
                           'event_labels': ex.doc['entity_labels'],
                           'left_event': left_event,
                           'right_event': right_event,
                           'doc_id': ex.doc_id}

    with open(args.save_dir + '/test.pickle', 'wb') as handle:
        pickle.dump(test_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # arguments for data processing
    p.add_argument('-data_dir', type=str)
    p.add_argument('-save_dir', type=str, default='')
    p.add_argument('-nr', type=float, default=0.0, help='Negative sample rate.')
    p.add_argument('-include_types', type=set, default={'TLINK'})
    p.add_argument('-eval_list', type=list, default=[])
    p.add_argument('-shuffle_all', type=bool, default=False)
    p.add_argument('-backward_sample', type=bool, default=False)
    p.add_argument('-data_dir_u', type=str,
                   help='Path to directory of unlabeld data (TE3-Silver). This should be output of '
                        '"ldcred.py flexnlp"')
    p.add_argument('-pred_win', type=int, default=200)
    p.add_argument('-data_type', type=str, default="tbd")
    p.add_argument('-joint', type=bool, default=False)
    p.add_argument('-loss_u', type=str, default='')
    p.add_argument('-emb', type=int, default=300)
    p.add_argument('-split', type=str, default='all') 
    p.add_argument('-seed', type=int, default=7)
    p.add_argument('-n_fts', type=int, default=1)
    args = p.parse_args()

    args.data_type = "matres"

    if args.data_type == "tbd":
        args.data_dir = "./tbd_output/"
        args.save_dir += args.data_type
    elif args.data_type == "matres":
        args.data_dir = "./matres_output/"
        args.save_dir += args.data_type
        
    main(args)
