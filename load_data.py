from pathlib import Path
import pickle
import sys
import argparse
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
import logging as log
import abc
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import random
import os
from typing import (List, Iterator, Tuple, Union, Dict, Mapping, Callable, Set, Type, TextIO, Optional)
import glob
from lxml import etree
import json
import pickle
import random
from collections import deque
from collections import OrderedDict, Counter, defaultdict
import re
import nltk

random_seed = 7
sys.path.append(str(Path(__file__).parent.absolute()))
from processTBD import TBDDoc, TBDRelation, TBDEntity
from featureFuncs import *
log.basicConfig(level=log.INFO)


matres_label_map = OrderedDict([('VAGUE', 'VAGUE'),
                             ('BEFORE', 'BEFORE'),
                             ('AFTER', 'AFTER'),
                             ('SIMULTANEOUS', 'SIMULTANEOUS')
                         ])

tbd_label_map = OrderedDict([('VAGUE', 'VAGUE'),
                             ('BEFORE', 'BEFORE'),
                             ('AFTER', 'AFTER'),
                             ('SIMULTANEOUS', 'SIMULTANEOUS'),
                             ('INCLUDES', 'INCLUDES'),
                             ('IS_INCLUDED', 'IS_INCLUDED'),
                         ])

@dataclass
class FlatRelation:
    doc_id: str
    id: str
    rev: bool
    left: TBDEntity
    right: TBDEntity
    doc: TBDDoc
    rel_type: str


def print_annotation_stats(data_dir: Path):
    """                                                                                                                   
    Prints stats such as relation types per each dataset split (train/test/dev)                                           
    Args:                                                                                                                 
        data_dir:                                                                                                         
                                                                                                                          
    Returns:                                                                                                              
    """

    for split in ('dev', 'test', 'train'):
        l1_types = defaultdict(int)
        l2_types = defaultdict(int)

        split_dir = data_dir / split
        for file in split_dir.glob('*/*.pkl'):
            with file.open('rb') as fh:
                doc: REDDoc = pickle.load(fh)
                for rel_id, rel in doc.relations.items():
                    l1_types[rel.type] += 1
                    for v in [_ for n, _ in rel.properties if n == 'type']:
                        l2_types[rel.type + '::' + v] += 1
        print(f"#{split} L1 ")
        for n, v in sorted(l1_types.items(), reverse=True, key=lambda x: x[0]):
            print(f'{n}\t{v}')

        print(f"##{split} L2 ")
        for n, v in sorted(l2_types.items(), reverse=True, key=lambda x: x[0]):
            print(f'{n}\t{v}')



def print_flat_relation_stats(data_dir: Path):
    """                                                                                                                   
    Same as `print_annotation_stats` but flattens the event-event relation annotations                                    
    Args:                                                                                                                 
        data_dir:                                                                                                         
                                                                                                                          
    Returns:                                                                                                              
                                                                                                                          
    """
    for split in ('dev', 'test', 'train'):
        flat_rels = read_relations(data_dir, split)
        stats = Counter(r.rel_type for r in flat_rels)
        print(f"#{split} Flattened")
        for n, v in sorted(stats.items(), reverse=True, key=lambda x: x[0]):
            print(f'{n}\t{v}')

def spans(txt):
    tokens = nltk.word_tokenize(txt)
    offset = 0
    for token, tag in nltk.pos_tag(tokens):
        offset = txt.find(token, offset)
        yield token, offset, offset + len(token), tag
        offset += len(token)

def find_token_index(passage):
    tokens = spans(passage)
    token2idx = {}
    count = 0
    for token, start, end, tag in tokens:
        token2idx["[%s:%s)" % (start, end)] = (token, count, tag)
        count += 1
    return token2idx


def read_relations(data_dir: Path,
                   split: Optional[str] = None,
                   data_type: str = "red",
                   exclude_types: Optional[Set[str]] = None,
                   include_types: Optional[Set[str]] = None,
                   other_label: str='OTHER',
                   neg_rate: float=0,
                   neg_label: str='NONE',
                   eval_list: list=[],
                   pred_window: int=200,
                   shuffle_all: bool=False,
                   joint: bool=False,
                   backward_sample: bool=False) -> Iterator[FlatRelation]:

    sample_type = 'L'
    if data_type in ['matres', 'tbd']:
        doc_type = TBDDoc
        ent_type = TBDEntity
        relation_type = TBDRelation
        if data_type == 'tbd':
            label_map = tbd_label_map
        else:
            label_map = matres_label_map

    split_dir = data_dir / split if split else data_dir
    print("%s processing %s %s" % ("="*10, split, "="*10))

    neg_counter = 0
    doc_dict = {}
    all_samples = []

    pos_counter = 0
    neg_counter = 0

    causal_counter = 0

    file_count = 0

    all_files = []
    with open(data_dir) as infile:
        for line in infile:
            data = json.loads(line)
            if data['split'] == split:
                all_files.append(data)

    sent_dists = []
    for doc in all_files:
        file_count += 1
        print(file_count)

        all_events = [k for k,v in doc['entities'].items() if v['type'] in ['EVENT']]#, 'TIMEX3']]
        #all_timex = [v for _,v in doc['entities'].items() if v['type'] in ['TIMEX3']]

        pos_dict = find_token_index(doc['raw_text'])
        doc['pos_dict'] = pos_dict

        #doc.pos_dict = pos_dict
        # create a entity label map: Span --> 0/1 indicator                                                               
        entity_labels = OrderedDict([(k, 0) for k,v in pos_dict.items()])

        pos_count = 0
        neg_count = 0
        pos_relations = set()   # remember to exclude these from randomly generated negatives                             

        for rel_id, rel in doc['relations'].items():
            rel_type = rel['type']
            other_type = ((include_types and rel_type not in include_types)
                          or (exclude_types and rel_type in exclude_types))

            if other_type:
                rel_type = other_label

            elif rel['type']== 'ALINK' or rel['type'] == 'TLINK':
                assert 'type' in rel['properties'] and len(rel['properties']['type']) == 1

                rel_type = rel['properties']['type'][0]
                rel_type = label_map[rel_type]


            # find the events associated with this relation                                                               
            events = [(n, v)
                      for n, vs in rel['properties'].items()
                      for v in vs
                      if isinstance(v, dict)]

            first_event_name, first_event = events[0]
            if first_event['type'] == 'TIMEX3':
                continue
            # all the second relations should have same name                                                              
            assert len(set(x for x, y in events[1:])) == 1
            for second_event_name, second_event in events[1:]:
                if second_event['type'] == 'TIMEX3':
                    continue
                all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(first_event['span'], second_event['span'], pos_dict)

                in_seq = [pos_dict[x][0] for x in all_keys[lidx_start:ridx_end+1]]

                # update entity_label if event                                                                            
                for i in all_keys[lidx_start:lidx_end+1] + all_keys[ridx_start:ridx_end+1]:
                    entity_labels[i] = 1

                first_mid = (lidx_start + lidx_end) / 2.0
                second_mid = (ridx_start + ridx_end) / 2.0
                tok_dist = np.abs(first_mid - second_mid)

                if rel_type is not None:
                    #rev_ind = False                                                                                      

                    if first_mid > second_mid:
                        left = second_event
                        right = first_event
                        rev_ind = True
                        sent_dists.append(len([x for x in in_seq if x == '.']))
                    # weird case where both events are the same text, but can be timex and event                          
                    elif first_mid == second_mid:
                        print(label)
                        print(first_event)
                        print(second_event)
                        print('*'*50)

                    else:
                        left = first_event
                        right = second_event
                        rev_ind = False
                        sent_dists.append(len([x for x in in_seq if x == '.']))


                    label = rel_type
                    pos_count += 1

                    if backward_sample or rev_ind:
                        # simple take the symmetric label rather then _rev;                                               
                        # only use rev_ind as reverse indicator                                                           
                        all_samples.append(("%s%s" % (sample_type, pos_counter), backward_sample, left, right, doc['id'], rev_map[label]))# + "_rev"))
                        pos_relations.add((right.id, left.id))
                    else:
                        all_samples.append(("%s%s" % (sample_type, pos_counter), backward_sample, left, right, doc['id'], label))
                        pos_relations.add((left['id'], right['id']))
                    pos_counter += 1

        doc['entity_labels'] = entity_labels
        doc_dict[doc['id']] = doc

        neg_sample_size = int(neg_rate * pos_count)

        if neg_sample_size > 0:
            all_neg = [(l_id, r_id) for l_id, r_id in combinations(all_events, 2)
                       if (l_id, r_id) not in pos_relations and (r_id, l_id) not in pos_relations]

            if split in eval_list:
                neg_sample_size = len(all_neg)

            random.Random(random_seed).shuffle(all_neg)
            for left_id, right_id in all_neg:#[:neg_sample_size]:                                                         

                # filter out entities that are the same: could be both as time and event                                  
                if ((int(doc.entities[right_id].span[0]) == int(doc.entities[left_id].span[0])) and  
                    (int(doc.entities[right_id].span[1]) == int(doc.entities[left_id].span[1]))):
                    continue

                # exclude rels that are more than 2*ngbrs token-dist away                                                 
                all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(doc.entities[left_id].span, 
                                                                                 doc.entities[right_id].span, pos_dict)
                in_seq = [pos_dict[x][0] for x in all_keys[lidx_start:ridx_end+1]]

                left_mid = (lidx_start + lidx_end) / 2.0
                right_mid = (ridx_start + ridx_end) / 2.0
                rev_ind = False

                ## be really careful this is only for one seq RNN model                                                   
                if left_mid > right_mid or len([x for x in in_seq if x == '.']) > 1:
                    continue
                elif left_mid == right_mid:
                    continue
                else:
                    left = doc.entities[left_id]
                    right = doc.entities[right_id]
                    rev_ind = False
                    neg_count += 1
                    all_samples.append(("N%s"%neg_counter, rev_ind, left, right, doc.id, neg_label))
                    neg_counter += 1

    print(Counter(sent_dists))
    print("Total positive sample size is: %s" % pos_counter)
    print("Total negative sample size is: %s" % neg_counter)
    print("Total causal sample size is: %s" % causal_counter)

    # with open("%s/%s_docs.txt" % (str(data_dir), split), 'w') as file:
    #     for k in doc_dict.keys():
    #         file.write(k)
    #         file.write('\n')
    # file.close()

    if shuffle_all:
        random.Random(random_seed).shuffle(all_samples)

    for s in all_samples:
        yield FlatRelation(s[4], s[0], s[1], s[2], s[3], doc_dict[s[4]], s[5])

def main(args):

    data_dir = args.data_dir
    opt_args = {}

    if args.tempo_filter:
        opt_args['include_types'] = args.include_types
    if args.skip_other:
        opt_args['other_label'] = None

    opt_args['data_type'] = args.data_type
    opt_args['pred_window'] = args.pred_win
    opt_args['shuffle_all'] = args.shuffle_all
    opt_args['backward_sample'] = args.backward_sample
    opt_args['joint'] = args.joint
    log.info(f"Reading datasets to memory from {data_dir}")

    # buffering data in memory --> it could cause OOM
    train_data = list(read_relations(Path(data_dir), 'train', **opt_args))
    with open('%s/%s/train.pickle' % (args.save_data_dir, args.data_type), 'wb') as fh:
        pickle.dump(train_data , fh)

    if args.data_type in ["tbd"]:
        opt_args['neg_rate'] = 0.0 # a random large number to include all                               
        dev_data = list(read_relations(Path(data_dir), 'dev', **opt_args))
        with open('%s/%s/dev.pickle' % (args.save_data_dir, args.data_type), 'wb') as fh:
            pickle.dump(dev_data, fh)

    test_data = list(read_relations(Path(data_dir), 'test', **opt_args))
    with open('%s/%s/test.pickle' % (args.save_data_dir, args.data_type), 'wb') as fh:
        pickle.dump(test_data, fh)

    if args.data_type == "matres":
        label_map = matres_label_map
    elif args.data_type == "tbd":
        label_map = tbd_label_map

    all_labels = list(OrderedDict.fromkeys(label_map.values()))

    args._label_to_id = OrderedDict([(all_labels[l],l) for l in range(len(all_labels))])
    args._id_to_label = OrderedDict([(l,all_labels[l]) for l in range(len(all_labels))])
    print(args._label_to_id)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # arguments for data processing                                                                    \
                                                                                                        
    p.add_argument('-data_dir', type=str,
                   help='Path to directory having RED data. This should be output of '
                        '"ldcred.py flexnlp"')
    p.add_argument('-save_data_dir', type=str, default="/Users/RJ/GitHub/EMNLP-2019/data_json/")
    p.add_argument('--tempo_filter', action='store_true',
                   help='Include Causal and Temporal relations only. By default all relations are'
                        ' included. When --filter is specified, non temporal and causal relations '
                        'will be labelled as OTHER')
    p.add_argument('--skip_other', action='store_true',
                   help='when --tempo-filter is applied, the excluded types are marked as OTHER.'
                        'By enabling --skip-other, OTHER types are skipped.')
    p.add_argument('-nr', '--neg-rate', type=float, default=0.0,help='Negative sample rate.')
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
    p.add_argument('-seed', type=int, default=7)
    p.add_argument('-n_fts', type=int, default=1)
    args = p.parse_args()

    args.eval_list = []
    args.data_type = "tbd"
    if args.data_type == "tbd":
        args.data_dir = "./tbd_output/"
        #args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        #args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]
    elif args.data_type == "matres":
        args.data_dir = "./matres_output/"
        #args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]

    main(args)
