VOCAB = 'ptb.vocab'

# build word 2 id dictionary
with open(VOCAB, 'r', encoding='utf-8') as f:
    words = [line.strip() for line in f]
word_to_id = {w: i for w, i in zip(words, range(len(words)))}


def get_word_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<eos>']


def word2id(RAW_DATA, OUTPUT_DATA):
    f_in = open(RAW_DATA, 'r', encoding='utf-8')
    f_out = open(OUTPUT_DATA, 'w', encoding='utf-8')
    for line in f_in:
        words = line.strip().split() + ['<eos>']
        ids = ' '.join([str(get_word_id(w)) for w in words])
        f_out.write(ids)
    f_in.close()
    f_out.close()


raw_dirs = ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']
output_dirs = ['ptb.train', 'ptb.valid', 'ptb.test']

for raw_data, output_data in zip(raw_dirs, output_dirs):
    word2id(raw_data, output_data)
