
#script for generating peptide sequences based on BLOSUM62


import random
from Bio.SubsMat import MatrixInfo 
import numpy as np 

blosum62 = MatrixInfo.blosum62
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def estimate_max_variants(template):
    template_parts = parse_template(template)
    total = 1

    for part_type, content in template_parts:
        if part_type == 'variable':
            for aa in content:

                options = set()
                for (a1, a2), score in blosum62.items():
                    if a1 == aa and score > 0:
                        options.add(a2)
                    elif a2 == aa and score > 0:
                        options.add(a1)

                if not options:
                    total *= 20
                else:
                    total *= len(options)

    return total


def parse_template(template): 

    parts = []
    buffer = ""
    inside_quotes = False

    for char in template: 

        if char == '"':

            if inside_quotes:
                
                parts.append(('fixed', buffer))
                buffer = ""

            else: 

                if buffer:
                    parts.append(('variable', buffer))
                    buffer = ""

            inside_quotes = not inside_quotes
        
        else: 

            buffer += char 
    if buffer: 

        parts.append(('variable', buffer))
    
    return parts


def get_blosum_substitutions(aa):

    scores = {}

    for(a1, a2), score in blosum62.items():

        if a1 == aa and a2 in AMINO_ACIDS:

            scores[a2] = score 
        
        elif a2 == aa and a1 in AMINO_ACIDS: 

            scores[a1] = score

    if not scores: 
        return random.choices(AMINO_ACIDS, k=1)
    
    aa_list = list(scores.keys())
    weights = np.array([max(scores[a], 0.1) for a in aa_list])
    weights = weights / weights.sum()

    return random.choices(aa_list, weights=weights, k=1)


def generate_blosum_sequences(template_parts): 

    seq = ""

    for part_type, content in template_parts: 

        if part_type == 'fixed': 
            seq += content 

        else: 

            for aa in content: 

                mutated = get_blosum_substitutions(aa)[0]
                seq += mutated 

    return seq 


def generate_sequences(template, n):
    template_parts = parse_template(template)
    sequences = set()

    while len(sequences) < n:
        seq = generate_blosum_sequences(template_parts)
        sequences.add(seq)

    return list(sequences)

