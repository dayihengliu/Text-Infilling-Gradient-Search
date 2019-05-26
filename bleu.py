import sys
import codecs
import os
import math
import operator
import json
from functools import reduce


def fetch_data(cand, ref):
    """ Store each reference and candidate sentences as a list """
    references = []
    if '.txt' in ref:
        reference_file = codecs.open(ref, 'r', 'utf-8')
        references.append(reference_file.readlines())
    else:
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference_file = codecs.open(os.path.join(root, f), 'r', 'utf-8')
                references.append(reference_file.readlines())
    candidate_file = codecs.open(cand, 'r', 'utf-8')
    candidate = candidate_file.readlines()
    return candidate, references

def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        #print('cand_dict',cand_dict)
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        #print('clipped_count',clipped_count)
        #print('count',count)
        #print('len(words)', len(words))
        r += best_length_match(ref_lengths, len(words))
        #print('best_match',r)

        c += len(words)
        #print('c', c)
    if clipped_count == 0:
        pr = 0
        #print('pr',pr)
    else:
        pr = float(clipped_count) / count
        #print('pr',pr)
    bp = brevity_penalty(c, r)
    #print('bp, c, r', bp, c, r)
    return pr, bp

def _count_ngram(candidate, ref_counts, ref_lengths, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    si = 0
    cand_sentence = candidate[si]
    cand_dict = {}
    words = cand_sentence.strip().split()
    limits = len(words) - n + 1
    for i in range(0, limits):
        ngram = ' '.join(words[i:i + n]).lower()
        if ngram in cand_dict:
            cand_dict[ngram] += 1
        else:
            cand_dict[ngram] = 1
    #print('cand_dict',cand_dict)
    clipped_count += clip_count(cand_dict, ref_counts)
    count += limits
    #print('clipped_count',clipped_count)
    #print('count',count)
    #print('len(words)', len(words))
    r += best_length_match(ref_lengths, len(words))
    #print('best_match',r)

    c += len(words)
    #print('c', c)
    if clipped_count == 0:
        pr = 0
        #print('pr',pr)
    else:
        pr = float(clipped_count) / count
        #print('pr',pr)
    bp = brevity_penalty(c, r)
    #print('bp, c, r', bp, c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c == 0:
        return 0.0
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def get_reference_count(references, n):
    ref_counts = []
    ref_lengths = []
    # Build dictionary of ngram counts
    for reference in references:
        ref_sentence = reference[0]
        ngram_d = {}
        words = ref_sentence.strip().split()
        ref_lengths.append(len(words))
        limits = len(words) - n + 1
        # loop through the sentance consider the ngram length
        for i in range(limits):
            ngram = ' '.join(words[i:i+n]).lower()
            if ngram in ngram_d.keys():
                ngram_d[ngram] += 1
            else:
                ngram_d[ngram] = 1
        ref_counts.append(ngram_d)
        
    return ref_counts, ref_lengths


def BLEU(candidate, references, gram=4):
    precisions = []
    for i in range(gram):
        pr, bp = count_ngram(candidate, references, i+1)
        #print pr, bp
        precisions.append(pr)
    #print geometric_mean(precisions), bp    
    bleu = geometric_mean(precisions) * bp
    return bleu

def _BLEU(candidate, ref_counts_n, ref_lengths_n, gram):
    assert len(ref_counts_n) == gram
    assert len(ref_lengths_n) == gram
    precisions = []
    bleu_gram = []
    for i in range(gram):
        pr, bp = _count_ngram(candidate, ref_counts_n[i], ref_lengths_n[i], i+1)
        precisions.append(pr)
        bleu = geometric_mean(precisions) * bp
        bleu_gram.append(bleu)
    
    return bleu_gram