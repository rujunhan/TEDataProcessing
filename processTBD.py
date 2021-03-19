#!/usr/bin/env python                                                                                                   
# Author : RJ ; Created: Oct 26, 2018                                                                                  
import os
from dataclasses import dataclass
from typing import (List, Iterator, Tuple, Union, Dict, Mapping, Callable, Set, Type, TextIO, Optional)
from collections import defaultdict as ddict
import glob
import logging as log
from lxml import etree
from pathlib import Path
from attr import attrs
import json
import pickle
import random
from collections import deque
from collections import OrderedDict, Counter
log.basicConfig(level=log.DEBUG)


label_map = OrderedDict([('v','VAGUE'), 
                         ('b', 'BEFORE'), 
                         ('a', 'AFTER'), 
                         ('ii', 'IS_INCLUDED'), 
                         ('i', 'INCLUDES'), 
                         ('s', 'SIMULTANEOUS')])

@dataclass
class TBDEntity():
    id: str
    type: str
    text: str
    tense: str
    aspect: str
    polarity: str
    span: Tuple[int, int]

    @classmethod
    def new(self, cls, fts, counter):
        res = {}
        res['text'] = cls.text
        res['span'] = [counter+1, counter+len(res['text'])]
        
        if cls.tag == "EVENT":
            res['id'] = cls.attrib['eid']
            res['type'] = cls.tag
            res['tense'] = fts[cls.attrib['eid']][0]
            res['aspect'] = None
            res['polarity'] = fts[cls.attrib['eid']][1]
        elif cls.tag == "TIMEX3":
            res['id'] = cls.attrib['tid']
            res['type'] = cls.tag
            res['aspect'] = None
            res['tense'] = None
            res['polarity'] = None
        return TBDEntity(**res)

@dataclass
class TBDRelation():
    id: str
    type: str
    properties: Dict[str, List[Union[str, TBDEntity]]]
    
    def new(pair, label, entities, id, id_counter):
        
        res = {}
        # relation type
        
        # TBDense data doesn't supply relation id
        res['properties'] = {}
        res['id'] = id + '_' + str(id_counter)
        res['type'] = 'TLINK'
        source, target = pair
        try:
            source = source[0] + source[1:] if source[0] == 'e' else source
        except:
            print(pair)
            print(label)
            print(source)

        target = target[0] + target[1:] if target[0] =='e' else target

        #assert entities[source].text == pair_text[0]
        #assert entities[target].text == pair_text[1]

        res['properties']['source'] = [entities[source]]
        res['properties']['target'] = [entities[target]]
        res['properties']['type'] = [label_map[label]]
        return TBDRelation(**res)

@dataclass
class TBDDoc:
    id: str
    raw_text: str = ""
    entities: Mapping[str, TBDEntity] = None
    relations: Mapping[str, TBDRelation] = None

    def parse_entities(self, entities, all_text, event_fts={}):

        res = []
        
        # this new dataset doesn't have span indicators, so we need to create it manually
        q, raw_text = self.build_queue(all_text)
        
        for e in entities:
            m = q.popleft()
            while m[0] != e.text:
                m = q.popleft()
            entity = TBDEntity.new(e, event_fts, m[1])
            # make sure span created is correct
            assert raw_text[entity.span[0] : entity.span[1] + 1] == e.text
            res.append(entity)
            
        return res, raw_text
    
    def build_queue(self, all_text):
        raw_text = ""
        counter = -1
        q = deque()
        for tx in all_text:
            q.append((tx, counter))
            counter += len(tx)
            raw_text += tx

        return q, raw_text

    def parse(self, id: str, text_path: Path, dense_pairs: {}):
        
        print(id)
        with text_path.open(encoding='utf-8') as f:
            xml_tree = etree.parse(f)
            
            events = xml_tree.xpath('.//EVENT')
            #timex = [t for t in xml_tree.xpath('.//TIMEX3') if t.attrib['functionInDocument'] != 'CREATION_TIME']
            
            event_fts = {e.attrib['eventID']: (e.attrib['tense'], e.attrib['polarity'], e.attrib['eiid']) for e in xml_tree.xpath('.//MAKEINSTANCE')}
            all_text = list(xml_tree.xpath('.//TEXT')[0].itertext())
            
            events, raw_text = self.parse_entities(events, all_text, event_fts)
            #timexs, _ = self.parse_entities(timex, all_text)
            entities = events #+ timexs
            entities = OrderedDict([(e.id, e) for e in entities])

            relations = []
            pos_pairs = []

            total_count = 0
            missing_count = 0
            
            timex = 0
            id_counter = 1
            for pair, label in dense_pairs[id].items():
                # It seems t0 in the TBDense dataset denote doc_create_time
                if (pair[0][0] == 't' or pair[1][0] == 't') or (pair[0] == 't0' or pair[1] == 't0'):
                    #print('Doc Time, skip')
                    #print(pair)
                    timex += 1
                else:
                    relations.append(TBDRelation.new(pair, label, entities, id, id_counter))
                    id_counter += 1

            assert (len(relations) + timex) == len(dense_pairs[id]) 
            relations = {r.id: r for r in relations}
            return TBDDoc(id, raw_text, entities, relations)


class PackageReader:

    def __init__(self, dir_path: str, dense_pairs: {}):
        self.root = os.path.abspath(dir_path)

        assert os.path.exists(self.root) and os.path.isdir(self.root)

        raw_dir = f'{self.root}/*'

        src_docs = glob.glob(raw_dir)
        print(len(src_docs))

        assert len(src_docs) == 183

        self.all_samples = dense_pairs

        suffix = ".tml"
        src_to_id = {src:src.replace(suffix, "")[len(raw_dir)-1:] for src in src_docs}
 
        dev_files = ["APW19980227.0487",
                     "CNN19980223.1130.0960", 
                     "NYT19980212.0019",
                     "PRI19980216.2000.0170", 
                     "ed980111.1130.0089"]

        test_files = ["APW19980227.0489",
                      "APW19980227.0494", 
                      "APW19980308.0201", 
                      "APW19980418.0210",
                      "CNN19980126.1600.1104", 
                      "CNN19980213.2130.0155",
                      "NYT19980402.0453", 
                      "PRI19980115.2000.0186",
                      "PRI19980306.2000.1675"]

        src_to_id = {k: v for k,v in src_to_id.items() if v in self.all_samples.keys()}

        assert len(src_to_id) == 36

        self.train_files = {k:v for k,v in list(src_to_id.items()) if v not in dev_files + test_files}
        self.test_files = {k:v for k,v in list(src_to_id.items()) if v in test_files}
        self.dev_files = {k:v for k,v in list(src_to_id.items()) if v in dev_files}

        print(len(self.train_files))
        print(len(self.dev_files))
        print(len(self.test_files))

        assert len(self.train_files) + len(self.dev_files) + len(self.test_files) == len(src_to_id)


    def read_split(self, split_name) -> Iterator[TBDDoc]:
        assert split_name in ('train', 'dev', 'test')

        src_id_map = {'train': self.train_files, 
                      'dev': self.dev_files,
                      'test': self.test_files}[split_name]
        
        for src_file, doc_id in src_id_map.items():
            doc = TBDDoc(doc_id)
            yield doc.parse(doc_id, Path(src_file), self.all_samples)

def tb_dense(dir_path):
    # tb_dense dataset only uses a subset of time bank dataset: 36 / 183                                
    # input: txt file specifying: doc, ent1, ent2, rel_type
    #output: dictionary - {doc: {(ent1, ent2):rel_type}}

    tbd = open("./TBDPairs.txt", 'r')
    count = 0
    tbd_samples = {}
    labels = []

    for line in tbd:
        line = line.strip().split('\t')
        line = [x for x in line if len(x) > 0]
        doc_id = line[0]
        ent1_id = line[1]
        ent2_id = line[2]
        label = line[3]
        
        if doc_id in tbd_samples.keys():
            tbd_samples[doc_id][(ent1_id, ent2_id)] = label
        else:
            tbd_samples[doc_id] = {(ent1_id, ent2_id): label}
        count += 1
        labels.append(label)
                    
    lc = Counter(labels)
    print(lc)
    docs = list(tbd_samples.keys())
    print("Total %s TBD samples processed" % count)
    print("Total %s TBD docs processed" % len(docs))

    return tbd_samples

@attrs(auto_attribs=True, slots=True)
class JsonSerializer:
    types: Set[Type]

    def __call__(self, obj):
        if type(obj) is list:
            return [self(x) for x in obj]
        elif type(obj) is dict:
            return {k: self(v) for k, v in obj.items()}
        elif type(obj) in self.types:
            return self(vars(obj))
        return obj

def export_json_lines(reader: PackageReader, out: TextIO,
                      splits: Iterator[str] = ('dev', 'test', 'train')):

    serializer = JsonSerializer({TBDEntity, TBDRelation, TBDDoc})
    count = 0
    for split in splits:
        for doc in reader.read_split(split):
            doc = vars(doc)
            doc['split'] = split
            line = json.dumps(doc, ensure_ascii=False, default=serializer)
            out.write(line)
            out.write('\n')
            count += 1
    log.info(f"wrote {count} docs")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-task', choices={'json', 'flexnlp'}, type=str,
                   help='Tasks. "json" will simply exports documents to jsonline file;'
                        ' "flexnlp" will annotate docs and dump them as pickle files')
    p.add_argument('-dir', help='Path to RED directory (extracted .tgz)', type=Path)
    p.add_argument('-out', help='output File (if task=json) or Folder (if task=flexnlp)', type=Path)
    p.add_argument('-dense_pairs', help='all samples in TBDense', type=dict)
    
    args = p.parse_args()
    
    args.task = 'json'
    args.dir = Path('./TBDense/timeml')
    args.out = Path('./tbd_output/')
    print(args)

    args.dense_pairs = tb_dense(args.dir)
    pr = PackageReader(args.dir, args.dense_pairs)
    
    if args.task == 'json':
        assert not args.out.exists() or args.out.is_file()
        with args.out.open('w', encoding='utf-8', errors='replace') as out:
            export_json_lines(pr, out)
    
