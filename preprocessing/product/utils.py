from __future__ import annotations
import pandas as pd
import pandarallel
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from scipy.sparse import vstack, csr_matrix
from janome.analyzer import Analyzer
from janome.charfilter import (RegexReplaceCharFilter,
                               UnicodeNormalizeCharFilter)
from janome.tokenfilter import (CompoundNounFilter, ExtractAttributeFilter,
                                LowerCaseFilter, POSKeepFilter,
                                TokenCountFilter)
from typing import List, Tuple, Dict
import logging
from sentence_transformers import SentenceTransformer

def join_nouns(text, hinshi=['名詞', '動詞', '形容詞', ])-> str:
    '''
    形態素解析, 品詞の抽出, 
    '''
    if pd.isna(text):
        return None
    char_filters = [
        UnicodeNormalizeCharFilter(),
        RegexReplaceCharFilter('^^[ぁ-ゖ]{1,2}$$', ''),
        RegexReplaceCharFilter('<br>', ''),
        RegexReplaceCharFilter('[#!:;<.*?>{}・`,()-=$/_\'"\[\]\|~-]+', ''),
    #RegexReplaceCharFilter('[#!:;<.*?>{}・`,()-=$/_\'"\[\]\|]+', ' '),
    ]
    token_filters = [
        POSKeepFilter(hinshi),
        LowerCaseFilter(),
        ExtractAttributeFilter('base_form'),
        #CompoundNounFilter()
        #TokenCountFilter(),
    ]
    analyzer = Analyzer(char_filters = char_filters, token_filters = token_filters)
    token_nouns = analyzer.analyze(text)
    ng_words = set(['あう', 'する', 'れる', 'さ', 'ある', 'よう', '等', 'など',
                'いる', 'ため', 'こと', 'ござる', 'くださる','おる','ある', 'あり',
                'なる', 'の', 'ん', 'そう', 'くる', 'いう', 'もの', 'ない',
                'ろ', 'そう', 'それ', 'うえ', 'さん', 'せる', 'おり', 'こ', 'す',
                'め', 'ば', 'こ', 'ゅ', 'ら', 'てる'])
    token_nouns = [l for l in token_nouns if l not in ng_words]
    #token_nouns = [l.surface for l in token_nouns]
    
    joint_nouns = (' ').join(token_nouns)
    return joint_nouns