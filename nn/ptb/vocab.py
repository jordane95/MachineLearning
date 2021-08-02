from collections import Counter
from operator import itemgetter


RAW_DATA = 'ptb.train.txt'
VOCAB_OUTPUT = 'ptb.vocab'

word_counts = Counter()
with open(RAW_DATA, 'r', encoding='utf-8') as f:
    for line in f:
        for w in line.strip().split():
            word_counts[w] += 1

sorted_word_counts = sorted(word_counts.items(), key=itemgetter(1), reverse=True)

words = ['<eos>'] + [pair[0] for pair in sorted_word_counts]

with open(VOCAB_OUTPUT, 'w', encoding='utf-8') as f:
    for w in words:
        f.write(w+'\n')

