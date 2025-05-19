# import spacy
import torch, random, os, math, sys, bisect
import unicodedata
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
from copy import deepcopy
import argparse
import csv
import pandas as pd
from tqdm import tqdm


torch.manual_seed(12345)
random.seed(1234)
csv.field_size_limit(1024 * 1024)

EOS_TOK = 1
LEIPZIG_DIR = 'leipzig-corpora'
EURALEX_DIR = 'euralex'
WIKIPRON_DIR = 'wikipron'
LEMMA_DIR = 'leipzig-lemmas'

MAX_LEN = 1024


def pad_and_resize_sequence(x, max_len):
    # Step 1: Pad the sequence with zeros using pad_sequence
    padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)

    # Step 2: Resize and fill with zeros if needed
    current_len = padded.size(1)
    if current_len < max_len:
        padded = torch.cat([padded, torch.zeros(len(x), max_len - current_len, device=padded.device)], dim=1)

    return padded.long()


def pad_batches(tensor_list):
    '''
    Pads groups of tensors to the same length
    :param tensor_list: List of tensors
    :return: new tensor padded
    '''
    max_len = max(map(lambda x: x.size(1), tensor_list))

    new_list = []
    for t in tensor_list:
        new_t = torch.cat([t, torch.zeros(t.size(0), max_len - t.size(1), device=t.device)], dim=1)
        new_list.append(new_t)

    return torch.cat(new_list, dim=0).long()


def PAD_COLLATE_FN(samples):
    '''
    Function for padding samples (to be used with the Torch dataloader class)
    :param samples: list of tuples of N tensors
    :return: N padded tensors where N is the number of tensors in the tuples in samples
    '''
    # for dataloaders
    transposed_samples = list(zip(*samples))
    max_len = max(max(tensor.size(0) for tensor in tensors) for tensors in transposed_samples)
    output = [pad_and_resize_sequence(x, max_len) for x in transposed_samples]
    return *output,


def FILTER_SPACES(word: str):
    '''
    Does what is says on the tin
    :param word: string
    :return: string without spaces
    '''
    return word.replace(' ', '')


def DECOMPOSE_CHARS(word: str):
    return unicodedata.normalize('NFD', word)


class AttnSeq2SeqCNN(torch.nn.Module):
    # similar to paper https://arxiv.org/pdf/1705.03122 but using Bahdanau attention because
    # this paper is written inscrutably and at least Bahdanau attention was already created.
    def __init__(self, alphabet_sz, hidden_sz, kernel_size, num_layers):
        super(AttnSeq2SeqCNN, self).__init__()
        def grad_hook(grad):
            return grad * torch.tril(torch.ones(hidden_sz, hidden_sz, device=grad.device))

        self.kernel_size = kernel_size
        self.embedding = torch.nn.Embedding(alphabet_sz, hidden_sz)
        self.pos_embedding = torch.nn.Embedding(MAX_LEN, hidden_sz)
        # channel is doubled for GLUs
        self.input_convolutions = torch.nn.ModuleList([torch.nn.Conv1d(hidden_sz, 2 * hidden_sz,
                                                                          kernel_size, padding='same')
                                                          for _ in range(num_layers)])
        self.target_convolutions = torch.nn.ModuleList([torch.nn.Conv1d(hidden_sz, 2 * hidden_sz,
                                                                          kernel_size)
                                                          for _ in range(num_layers)])
        self.residual_connections = torch.nn.ModuleList([torch.nn.Linear(hidden_sz, hidden_sz) for _ in range(num_layers)])

        self.target_residual_connections = torch.nn.ModuleList([torch.nn.Linear(hidden_sz, hidden_sz) for _ in range(num_layers)])

        self.attention_layers = torch.nn.ModuleList([BahdanauAttention(hidden_sz) for _ in range(num_layers)])

        # self.input_convolution = torch.nn.Conv1d(hidden_sz, 2*hidden_sz, kernel_size, padding='same')
        # self.target_convolution = torch.nn.Conv1d(hidden_sz, 2*hidden_sz, kernel_size)

        # self.residual_connection = torch.nn.Linear(hidden_sz, hidden_sz)
        # self.query = torch.nn.Linear(hidden_sz, hidden_sz)

        # self.attention = BahdanauAttention(hidden_sz)

        # gating function, split dimension is the channel dimension (i.e., the second dimension)
        self.gating = torch.nn.GLU(dim=1)
        self.decoder_dnn = torch.nn.Linear(hidden_sz, alphabet_sz)


        for resid in self.target_residual_connections:
            resid.weight.register_hook(grad_hook)

    def target_padding(self, target):
        '''
        Pads beginning of sequence per paper https://arxiv.org/pdf/1705.03122
        Padding is k-1 in the front
        Assumes sequence dimension is last dimension already
        :param target:
        :return:
        '''
        return torch.nn.functional.pad(target, (self.kernel_size-1, 0), 'constant', 0.0)

    # TO DO: Support multiple layers
    def encode(self, x):
        for c, r in zip(self.input_convolutions, self.residual_connections):
            resid = r(torch.permute(x, (0, 2, 1)))
            new = c(x)
            new = self.gating(new)
            x = new + torch.permute(resid, (0, 2, 1)) #residual conections
        return x

    def decode(self, target, x):
        att_x = torch.permute(x, (0, 2, 1))
        # First permute the target to (batch, channels, seq_len) for convolution
        target_conv = torch.permute(target, (0, 2, 1))

        for c, a, r in zip(self.target_convolutions, self.attention_layers,  self.target_residual_connections):
            # Pad front of target per paper specs
            # Prevents future information from being accessible during teacher forcing
            padded_target = self.target_padding(target_conv)
            # Apply convolution and gating
            gated_target = self.gating(c(padded_target))
            # Permute back to (batch, seq_len, channels) for attention
            att_target = torch.permute(gated_target, (0, 2, 1))
            # Attention layer
            target, _ = a.batch_forward(att_target, att_x)
            # For next layer, permute back to (batch, channels, seq_len)
            target_conv = torch.permute(target + r(torch.permute(target_conv, (0, 2, 1))), (0, 2, 1))

        # Return in the format (batch, seq_len, channels) as expected by forward
        return target

    def forward_step(self, x, encoded):
        # Comment out old single-layer implementation
        # context, _ = self.attention(x, encoded)

        # For multi-layer, we'll use the first attention layer
        context, _ = self.attention_layers[0](x, encoded)
        return context

    def forward(self, x, target_in):
        # input are longs of shape batch x seq_len
        device = x.device
        target_embedding = self.embedding(target_in)
        input_embedding = self.embedding(x)
        _, input_seq_len, _ = input_embedding.size()
        _, target_seq_len, _ = target_embedding.size()
        input_embedding += self.pos_embedding(torch.arange(0, input_seq_len).to(device))
        target_embedding += self.pos_embedding(torch.arange(0, target_seq_len).to(device))

        # Comment out old single-layer implementation
        # input_residual = self.residual_connection(input_embedding)

        # permutations put channel values in the middle
        input_embedding = torch.permute(input_embedding, (0, 2, 1))
        # same but pad front of target per paper specs
        # prevents future information from being accessible during teacher forcing
        # target_embedding = self.target_padding(torch.permute(target_embedding, (0, 2, 1)))

        # Comment out old single-layer implementation
        # conv_input = self.input_convolution(input_embedding)
        # conv_target = self.target_convolution(target_embedding)

        # gated_input = self.gating(conv_input)
        # gated_target = self.gating(conv_target)

        # add residual connnections, must permute as with the embeddings
        # deep_input = gated_input + torch.permute(input_residual, (0, 2, 1))

        # deep_input = torch.permute(deep_input, (0, 2, 1))
        # gated_target = torch.permute(gated_target, (0, 2, 1))

        # decoder_outputs, _ = self.attention.batch_forward(gated_target, deep_input)

        # New multi-layer implementation
        encoded = self.encode(input_embedding)
        decoded = self.decode(target_embedding, encoded)

        # The decode method now returns in the format (batch, seq_len, channels)
        # so we don't need to permute it again

        logits = self.decoder_dnn(decoded)

        return logits

    def init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.01)
        with torch.no_grad():
            for resid in self.target_residual_connections:
                resid.weight.copy_(torch.tril(resid.weight))


class SimpleSeq2SeqCNN(torch.nn.Module):
    def __init__(self, alphabet_sz, hidden_sz, kernel_size, num_stacks):
        super(SimpleSeq2SeqCNN, self).__init__()
        self.embedding = torch.nn.Embedding(alphabet_sz, hidden_sz)

        self.convolutions = torch.nn.ModuleList([torch.nn.Conv1d(hidden_sz, hidden_sz, kernel_size, padding='same')
                                                 for _ in range(num_stacks)]
        )

        self.decoder_dnn = torch.nn.Linear(hidden_sz, alphabet_sz)

    def forward(self, x, target_in):
        # if torch.max(x).item() > 20677:
        #     breakpoint()
        target_length = x.size(1)
        embed = self.embedding(x)
        inputs = embed.permute(0, 2, 1)  # reorganize to swap sequence length and embedding dimension

        # Encoder
        for conv in self.convolutions:
            inputs = conv(inputs)

        logits = self.decoder_dnn(inputs)

        return logits

    def init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.01)


class BahdanauAttention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = torch.nn.Linear(hidden_size, hidden_size)
        self.Ua = torch.nn.Linear(hidden_size, hidden_size)
        self.Va = torch.nn.Linear(hidden_size, 1)

    def batch_forward(self, queries, keys):
        # for when queries and keys can be processed in parallel
        queries_ = queries.unsqueeze(dim=2)
        keys_ = keys.unsqueeze(dim=1)
        scores = self.Va(torch.tanh(self.Wa(queries_) + self.Ua(keys_)))
        scores = scores.squeeze()

        weights = torch.nn.functional.softmax(scores, dim=-1)
        contexts = torch.bmm(weights, keys)

        return contexts, weights

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class Seq2SeqLSTM(torch.nn.Module):
    def __init__(self, alphabet_sz, hidden_sz, encoder_lstm_stacks, decoder_lstm_stacks):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder_stacks = encoder_lstm_stacks
        self.decoder_stacks = decoder_lstm_stacks
        self.embedding_layer = torch.nn.Embedding(alphabet_sz, hidden_sz)
        self.encoder = torch.nn.LSTM(hidden_sz, hidden_sz, encoder_lstm_stacks, batch_first=True)
        self.attention = BahdanauAttention(hidden_sz)
        self.decoder = torch.nn.LSTM(hidden_sz*2, hidden_sz, decoder_lstm_stacks, batch_first=True)
        self.decoder_dnn = torch.nn.Linear(hidden_sz, alphabet_sz)

    def forward(self, x, target_in):
        x_embedding = self.embedding_layer(x)
        device = x.device
        target_embedding = self.embedding_layer(target_in)
        batch_sz, seq_len, embed_dim = target_embedding.size()
        encoding_seq = self.encoder(x_embedding)[0]
        decoder_outputs = []
        hx = (torch.zeros(self.decoder_stacks, batch_sz, embed_dim).to(device),
              torch.zeros(self.decoder_stacks, batch_sz, embed_dim).to(device))

        for t in range(seq_len):
            out, hx, attn = self.forward_step(target_embedding[:, t, :].unsqueeze(dim=1), hx, encoding_seq)
            decoder_outputs.append(out)

        decoding = torch.cat(decoder_outputs, dim=1)
        logits = self.decoder_dnn(decoding)

        return logits

    def forward_step(self, x, hx, encoded):
        h, c = hx
        query = h[-1, :, :].unsqueeze(dim=1)
        context, attention = self.attention(query, encoded)
        lstm_in = torch.cat([x, context], dim=2)
        out, out_hx = self.decoder(lstm_in, hx)
        return out, out_hx, attention

    def init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.01)


class Phon2OrthDataset(torch.utils.data.Dataset):
    def __init__(self, lang: str, lang_dict: dict, transcription_type: str,
                 script: str, word_counter: Counter = None):

        self.ipa_dict = lang_dict[lang]
        self.tokenizer = gen_tokenizer(lang_dict.values())
        self.num_items = len(set([word for word in self.ipa_dict.keys()]))
        self.transcription_type = transcription_type
        self.script = script

        if word_counter is not None:
            sorted_words = reversed(sorted(word_counter, key=lambda x: word_counter[x]))

            self.words = list(sorted_words)
            self.forms = [self.ipa_dict[word] for word in self.words]
            self.counts = [word_counter[word] for word in self.words]
            self.cumulative_counts = [0] + list(self._cumulative_sum(self.counts))
        else:
            # If no word_counter is provided, default to the full ipa_dict
            self.words = list(self.ipa_dict.keys())
            self.forms = [self.ipa_dict[word] for word in self.words]
            self.counts = [1] * len(self.words)  # Assume each word appears once
            self.cumulative_counts = list(range(len(self.words) + 1))

    def _cumulative_sum(self, counts):
        total = 0
        for count in counts:
            total += count
            yield total

    def __len__(self):
        return self.cumulative_counts[-1]

    def __getitem__(self, indx):
        word_idx = bisect.bisect_right(self.cumulative_counts, indx) - 1
        word = self.words[word_idx]
        ipa = self.forms[word_idx]
        BOS = len(self.tokenizer) + 2

        phonology = torch.LongTensor([BOS] + [self.tokenizer[c] for c in ipa] + [EOS_TOK])
        orthography_target = torch.LongTensor([EOS_TOK] + [self.tokenizer[c] for c in word])
        orthography_out = torch.LongTensor([self.tokenizer[c] for c in word] + [EOS_TOK])

        return phonology, orthography_target, orthography_out

    def control(self):
        controlled_dataset = deepcopy(self)
        controlled_dataset.forms = ['' for _ in controlled_dataset.forms]
        return controlled_dataset

    def prune(self, size: int):
        # prunes dataset such that the number of words in the dataset is of size int
        self.words = self.words[0:size]
        self.counts = self.counts[0:size]
        self.cumulative_counts = self.cumulative_counts[0:size]
        self.forms = self.forms[0:size]

    def frequency_sorted(self):
        all_data = zip(self.counts, self.words, self.forms)
        sorted_data = sorted(all_data, key=lambda x: x[0])
        self.counts, self.words, self.forms = zip(*sorted_data)
        self.cumulative_counts = [0] + list(self._cumulative_sum(self.counts))

    def delete_counts(self):
        self.counts = [1] * len(self.words)
        self.cumulative_counts = list(range(len(self.words) + 1))



class Orth2PhonDataset(torch.utils.data.Dataset):
    def __init__(self, lang: str, lang_dict: dict, transcription_type: str,
                 script: str, word_counter: Counter = None):
        self.ipa_dict = lang_dict[lang]
        self.tokenizer = gen_tokenizer(lang_dict.values())
        self.num_items = len(set([word for word in self.ipa_dict.keys()]))
        self.transcription_type = transcription_type
        self.lang = lang
        self.script = script

        if word_counter is not None:
            sorted_words = reversed(sorted(word_counter, key=lambda x: word_counter[x]))

            self.words = list(sorted_words)
            self.forms = [self.ipa_dict[word] for word in self.words]
            self.counts = [word_counter[word] for word in self.words]
            self.cumulative_counts = [0] + list(self._cumulative_sum(self.counts))
        else:
            # If no word_counter is provided, default to the full ipa_dict
            self.words = list(self.ipa_dict.keys())
            self.forms = [self.ipa_dict[word] for word in self.words]
            self.counts = [1] * len(self.words)  # Assume each word appears once
            self.cumulative_counts = list(range(len(self.words) + 1))

    def _cumulative_sum(self, counts):
        total = 0
        for count in counts:
            total += count
            yield total

    def __len__(self):
        return self.cumulative_counts[-1]

    def __getitem__(self, indx):
        word_idx = bisect.bisect_right(self.cumulative_counts, indx) - 1
        word = self.words[word_idx]
        ipa = self.forms[word_idx]
        BOS = len(self.tokenizer)+2

        phonology_out = torch.LongTensor([self.tokenizer[c] for c in ipa] + [EOS_TOK])
        phonology_target = torch.LongTensor([EOS_TOK] + [self.tokenizer[c] for c in ipa])
        orthography = torch.LongTensor([BOS] + [self.tokenizer[c] for c in word] + [EOS_TOK])

        return orthography, phonology_target, phonology_out

    def control(self):
        controlled_dataset = deepcopy(self)
        controlled_dataset.words = ['' for _ in controlled_dataset.words]
        return controlled_dataset

    def prune(self, size: int):
        # prunes dataset such that the number of words in the dataset is of size int
        self.words = self.words[0:size]
        self.counts = self.counts[0:size]
        self.cumulative_counts = self.cumulative_counts[0:size]
        self.forms = self.forms[0:size]

    def frequency_sorted(self):
        all_data = zip(self.counts, self.words, self.forms)
        sorted_data = sorted(all_data, key=lambda x: x[0])
        self.counts, self.words, self.forms = zip(*sorted_data)
        self.cumulative_counts = [0] + list(self._cumulative_sum(self.counts))

    def delete_counts(self):
        self.counts = [1] * len(self.words)
        self.cumulative_counts = list(range(len(self.words) + 1))


def prune_datasets(datasets: dict):
    max_type_sz = min(map(lambda x: len(x.words), datasets.values()))
    for _, dataset in datasets.items():
        dataset.prune(max_type_sz)


def read_dict_file(filename, drop_chars='\u200d', exclude_chars='1234567890-?.!;()'):
    ipa_dict = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            forms = line.split()
            orth = forms[0]
            if any([ex in orth for ex in exclude_chars]):
                continue
            # .strip() does not work here
            phon = ''.join([c for c in forms[1] if c not in drop_chars])
            ipa_dict[orth] = phon
        return ipa_dict


def read_lex_data(directory):
    concept = pd.read_csv(os.path.join(directory, 'northeuralex-0.9-concept-data.tsv'), sep='\t')
    lex = pd.read_csv(os.path.join(directory, 'northeuralex-0.9-forms.tsv'), sep='\t')
    unified = pd.merge(lex, concept, left_on='Concept_ID', right_on='id_nelex')
    return unified


def load_lemma_data(directory: str, lang_directory: str, broad_default=True,
                    drop_chars: str = '%#@^<>$!*(){}[]",0123456789'):
    # WORD LIST LOADING SECTION
    files = [os.path.join(directory, lang_directory, f)
             for f in os.listdir(os.path.join(directory, lang_directory))]

    target_file = os.path.join(directory, lang_directory, f'{lang_directory}_lemmas.txt')

    # WORD PRONUNCIATION LOADING SECTION
    pronunciation_files = [f for f in files if ('narrow' in f) or ('broad' in f)]
    pronunciation_file = None
    file_type = None
    for file in pronunciation_files:
        # pronunciation file will be default first, then other, then nothing
        if broad_default:
            if 'broad' in file:
                pronunciation_file = file
                file_type = 'broad'
            elif 'narrow' in file and pronunciation_file is None:
                pronunciation_file = file
                file_type = 'narrow'

        else:
            if 'narrow' in file:
                pronunciation_file = file
                file_type = 'narrow'
            elif 'broad' in file and pronunciation_file is None:
                pronunciation_file = file
                file_type = 'broad'

    if pronunciation_file is None:
        # if no suitable file exists for this language, skip the rest of the function
        return None

    ipa_dict = {}
    with open(pronunciation_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t', quotechar=None)
        for row in reader:
            if len(row) == 2:  # Ensure there are exactly 2 columns
                key, value = row
                key = DECOMPOSE_CHARS(key)
                if key not in ipa_dict:
                    if len(key) < MAX_LEN:
                        ipa_dict[key] = FILTER_SPACES(value) # remove spaces from IPA text

    word_counter = Counter()
    with open(target_file, mode='r', encoding='utf-8') as words:
        reader = csv.reader(words, delimiter='\t', quotechar=None)
        for row in reader:
            word = DECOMPOSE_CHARS(str(row[0]))
            count = int(row[1])
            if word in ipa_dict.keys() and not any(c in word for c in drop_chars):
                word_counter[word] = count

    return word_counter, ipa_dict, file_type


def load_leipzig_data(directory: str, lang_directory: str, sizes: dict,
                      broad_default=True, max_size=False, target_size='100K',
                      drop_chars: str = '%#@^<>$!*(){}[]",0123456789'):
    # WORD LIST LOADING SECTION
    files = [os.path.join(directory, lang_directory, f)
             for f in os.listdir(os.path.join(directory, lang_directory))]
    if max_size:
        # find file with max size under max size condition
        target_file = None
        max_file_size = 0
        for file in files:
            for size in sizes.keys():
                if size in file and sizes[size] > max_file_size:
                    target_file = file
                    max_file_size = sizes[size]

    else:
        # otherwise we return the target file
        target_file = None
        for file in files:
            if target_size in file:
                target_file = file

    # WORD PRONUNCIATION LOADING SECTION
    pronunciation_files = [f for f in files if ('narrow' in f) or ('broad' in f)]
    pronunciation_file = None
    file_type = None
    for file in pronunciation_files:
        # pronunciation file will be default first, then other, then nothing
        if broad_default:
            if 'broad' in file:
                pronunciation_file = file
                file_type = 'broad'
            elif 'narrow' in file and pronunciation_file is None:
                pronunciation_file = file
                file_type = 'narrow'

        else:
            if 'narrow' in file:
                pronunciation_file = file
                file_type = 'narrow'
            elif 'broad' in file and pronunciation_file is None:
                pronunciation_file = file
                file_type = 'broad'

        script = file.split('_')[1]  #  script is always second part of file name

    if target_file is None or pronunciation_file is None:
        # if no suitable file exists for this language, skip the rest of the function
        return None

    ipa_dict = {}
    with open(pronunciation_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t', quotechar=None)
        for row in reader:
            if len(row) == 2:  # Ensure there are exactly 2 columns
                key, value = row
                key = DECOMPOSE_CHARS(key)
                if key not in ipa_dict:
                    if len(key) < MAX_LEN:
                        ipa_dict[key] = FILTER_SPACES(value) # remove spaces from IPA text

    word_counter = Counter()
    with open(target_file, mode='r', encoding='utf-8') as words:
        reader = csv.reader(words, delimiter='\t', quotechar=None)
        for row in reader:
            word = DECOMPOSE_CHARS(str(row[1]))
            count = int(row[2])
            if word in ipa_dict.keys() and not any(c in word for c in drop_chars):
                word_counter[word] = count

    return word_counter, ipa_dict, file_type, script


def load_wikipron_set(language_file: str, type_cutoff=10000):
    ipa_dict = {}

    with open(language_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t', quotechar=None)
        for row in reader:
            if len(row) == 2:
                key, value = row
                key = DECOMPOSE_CHARS(key)
                if key not in ipa_dict:
                    if len(key) < MAX_LEN:
                        ipa_dict[key] = FILTER_SPACES(value)
    if len(ipa_dict) < type_cutoff:
        return None

    return ipa_dict


def make_wikipron_sets(directory: str, language_list: list, dataset_type: str, type_cutoff=10000, exclude_language_list=None):
    allowable_dialect = {'us', 'la', 'sw', 'clas', 'e', 'bz', 'saigon', 'gen'} # for now: hardcoded dialects of interest
    lang_ipa_dict = {}
    file_types = {}
    scripts = {}
    languages = []

    if language_list is None:
        # Run this if no list is provided (pulls all languages)
        for file in tqdm(os.listdir(directory), desc='Reading languages.'):
            if 'filtered' in file:
                # skip filtered wikipron datasets
                continue

            # after skipping filtered files, grab relevant features for dataset logging
            features = file.split('_')
            if len(features) == 3:
                language, script, file_type = features
                dialect = 'gen'
            elif len(features) == 4:
                language, script, dialect, file_type = features
            else:
                continue

            file_type = 'broad' if 'broad' in file_type else 'narrow'  # remove extraneous text

            # skip languages not in list if language list is provided
            if language_list is not None:
                if language not in language_list or dialect not in allowable_dialect:
                    continue

            # skip languages in exclude list
            if exclude_language_list is not None and language in exclude_language_list:
                continue

            full_name = os.path.join(directory, file)
            ipa_dict = load_wikipron_set(full_name, type_cutoff)
            if ipa_dict is None:
                continue

            name = f'{language}_{file_type}_{script}'
            languages.append(name)
            lang_ipa_dict[name] = ipa_dict
            file_types[name] = file_type
            scripts[name] = script

    if dataset_type == "orth" or dataset_type == "orthography":
        datasets = {l: Orth2PhonDataset(l, lang_ipa_dict, file_types[l], scripts[l])
                    for l in tqdm(languages, desc='Converting.')}
    else:
        datasets = {l: Phon2OrthDataset(l, lang_ipa_dict, file_types[l], scripts[l])
                    for l in tqdm(languages, desc='Converting.')}

    return datasets


def make_leipzig_datasets(directory: str, language_list: list, dataset_type: str, broad_default=True, max_size=False,
                          target_size='100K', token_cutoff=100_000, type_cutoff=5_000, exclude_language_list=None):
    sizes = {'30K': 1, '100K': 2, '300K': 3, '1M': 4}  # possible sizes for the Leipzig file
    assert target_size in sizes.keys()

    lang_ipa_dict = {}
    lang_data = {}
    file_types = {}
    script = {}
    languages = []

    if language_list is None:
        # Run this if no list is provided (pulls all languages)
        for lang_directory in tqdm(os.listdir(directory), desc='Reading languages.'):
            language = lang_directory
            # Skip if language is in exclude list
            if exclude_language_list is not None and language in exclude_language_list:
                continue
            load = load_leipzig_data(directory, lang_directory, sizes, broad_default, max_size, target_size)
            if load is None:
                continue
            languages.append(language)
            lang_data[language], lang_ipa_dict[language], file_types[language], script[language] = load

    else:
        # Run this if language list is provided
        for language in tqdm(language_list, desc='Reading languages.'):
            # Skip if language is in exclude list
            if exclude_language_list is not None and language in exclude_language_list:
                continue
            lang_directory = language
            load = load_leipzig_data(directory, lang_directory, sizes, broad_default, max_size, target_size)
            if load is None:
                continue
            languages.append(language)
            lang_data[language], lang_ipa_dict[language], file_types[language], script[language] = load

    # breakpoint()
    if dataset_type == "orth" or dataset_type == "orthography":
        datasets = {l: Orth2PhonDataset(l, lang_ipa_dict, file_types[l], script[l], lang_data[l])
                    for l in tqdm(languages, desc='Converting.')}
    else:
        datasets = {l: Phon2OrthDataset(l, lang_ipa_dict, file_types[l], script[l], lang_data[l])
                    for l in tqdm(languages, desc='Converting.')}

    datasets = {l: dataset for l, dataset in datasets.items() if len(dataset.words) > type_cutoff}
    prune_datasets(datasets)
    datasets = {l: dataset for l, dataset in datasets.items() if len(dataset) > token_cutoff}

    lengths = list(map(lambda x: len(x.words), datasets.values()))
    assert all([lengths[0] == length for length in lengths])

    # breakpoint()
    # return filtered datasets
    return datasets


def make_lemma_datasets(directory: str, language_list: list, dataset_type: str, broad_default=True,
                        token_cutoff=100_000, type_cutoff=5_000, exclude_language_list=None):
    script = 'na'  # not used for this batch

    lang_ipa_dict = {}
    lang_data = {}
    file_types = {}
    languages = []

    if language_list is None:
        # Run this if no list is provided (pulls all languages)
        for lang_directory in tqdm(os.listdir(directory), desc='Reading languages.'):
            language = lang_directory
            # Skip if language is in exclude list
            if exclude_language_list is not None and language in exclude_language_list:
                continue
            load = load_lemma_data(directory, lang_directory, broad_default)
            if load is None:
                continue
            languages.append(language)
            lang_data[language], lang_ipa_dict[language], file_types[language] = load

    else:
        # Run this if language list is provided
        for language in tqdm(language_list, desc='Reading languages.'):
            # Skip if language is in exclude list
            if exclude_language_list is not None and language in exclude_language_list:
                continue
            lang_directory = language
            load = load_lemma_data(directory, lang_directory, broad_default)
            if load is None:
                continue
            languages.append(language)
            lang_data[language], lang_ipa_dict[language], file_types[language] = load

    # breakpoint()
    if dataset_type == "orth" or dataset_type == "orthography":
        datasets = {l: Orth2PhonDataset(l, lang_ipa_dict, file_types[l], script, lang_data[l])
                    for l in tqdm(languages, desc='Converting.')}
    else:
        datasets = {l: Phon2OrthDataset(l, lang_ipa_dict, file_types[l], script, lang_data[l])
                    for l in tqdm(languages, desc='Converting.')}

    datasets = {l: dataset for l, dataset in datasets.items() if len(dataset.words) > type_cutoff}
    prune_datasets(datasets)
    datasets = {l: dataset for l, dataset in datasets.items() if len(dataset) > token_cutoff}

    lengths = list(map(lambda x: len(x.words), datasets.values()))
    assert all([lengths[0] == length for length in lengths])

    # return filtered datasets
    return datasets


def gen_tokenizer(dict_list: list):
    token_indx = 2
    tokenizer = {}
    for ipa_dict in dict_list:
        for k, v in ipa_dict.items():
            for c in k:
                if c not in tokenizer:
                    tokenizer[str(c)] = token_indx
                    token_indx +=1
            for c in v:
                if c not in tokenizer:
                    tokenizer[str(c)] = token_indx
                    token_indx +=1

    return tokenizer


def training(model, dataloader, optimizer, epochs, patience):
    model.train()
    no_improvement = 0
    best_loss = float('inf')
    for epoch in epochs:
        for batch in dataloader:
            optimizer.zero_grad()
            # Forward pass
            inputs, target_in, target_out = batch
            inputs = inputs.to('cuda')
            target_in = target_in.to('cuda')
            target_out = target_out.to('cuda')
            target_mask = target_out != 0
            outputs = model(inputs, target_in)

            loss = torch.nn.functional.cross_entropy(outputs[target_mask], target_out[target_mask], reduction='sum')

            loss.backward()
            optimizer.step()

            if loss > best_loss:
                no_improvement += 1
            else:
                best_loss = loss
                no_improvement = 0

            if no_improvement > patience:
                break


def prequential_coding_fast(model, dataset, set_name, epochs, learning_rate, batch_size, seed):
    # Initialize the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate) #, momentum=.95)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    model.to('cuda')

    # Set up LR scheduler

    model.train()  # Set the model to training mode
    code_length = 0

    torch.manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=NUM_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=PAD_COLLATE_FN, sampler=sampler)
    num_preds = 0
    num_in = 0

    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f'Compressing data for {set_name} (epoch {epoch}): ', leave=False):
            optim.zero_grad()
            # Forward pass
            inputs, target_in, target_out = batch
            inputs = inputs.to('cuda')
            target_in = target_in.to('cuda')
            target_out = target_out.to('cuda')
            target_mask = target_out != 0
            num_in += (inputs != 0).sum().item()
            num_preds += target_mask.sum().item()
            outputs = model(inputs, target_in)
            try:
                loss = torch.nn.functional.cross_entropy(outputs[target_mask], target_out[target_mask], reduction='sum')
            except:
                breakpoint()
            code_length += loss.item()

            # Backward and optimize
            loss.backward()
            optim.step()

    print(f"Performance for {set_name}: Prequential code length: {code_length}")

    return model, code_length, num_preds, num_in


def prequential_coding_block(model, dataset, set_name, epochs, learning_rate, batch_size,
                             seed, stop_points, patience=20):
    """
    :param model:
    :param dataset:
    :param set_name:
    :param epochs:
    :param learning_rate:
    :param batch_size:
    :param seed:
    :param stop_points:
    :param patience:
    :return:
    """
    # Initialize the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    model.to('cuda')

    # Set up LR scheduler
    code_length = 0
    num_in = 0
    num_preds = 0

    torch.manual_seed(seed)

    if stop_points[-1] != 1:
        stop_points.append(1)
    if stop_points[0] != 0:
        stop_points.insert(0, 0)

    chunk_sizes = []

    for stop_point_i, stop_point_j in zip(stop_points[1:], stop_points[:-1]):
        chunk_sizes.append(stop_point_i - stop_point_j)

    subset = torch.utils.data.random_split(dataset, [NUM_SAMPLES, len(dataset)-NUM_SAMPLES])[0]
    chunks = torch.utils.data.random_split(subset, chunk_sizes)
    train_chunks = []
    for i in range(len(chunks)):
        train_chunks.append(torch.utils.data.ConcatDataset(chunks[:i+1]))

    init_params = deepcopy(model.state_dict())

    for train, eval in tqdm(zip(train_chunks, chunks), desc=f'Compressing data for {set_name}: ', leave=False):
        train_dataloader = DataLoader(train, batch_size=batch_size, collate_fn=PAD_COLLATE_FN, shuffle=True)
        val_dataloader = DataLoader(eval, batch_size=batch_size, collate_fn=PAD_COLLATE_FN)
        for batch in val_dataloader:
            inputs, target_in, target_out = batch
            inputs = inputs.to('cuda')
            target_in = target_in.to('cuda')
            target_out = target_out.to('cuda')
            target_mask = target_out != 0
            num_in += (inputs != 0).sum().item()
            num_preds += target_mask.sum().item()
            outputs = model(inputs, target_in)

            loss = torch.nn.functional.cross_entropy(outputs[target_mask], target_out[target_mask], reduction='sum')

            code_length += loss.item()

        if train == train_chunks[-1]:
            break

        model.load_state_dict(deepcopy(init_params))
        model.train()
        no_improvement = 0
        best_loss = float('inf')
        for epoch in tqdm(range(epochs), desc=f'Compressing data current iteration: ', leave=False):
            for batch in train_dataloader:
                optim.zero_grad()
                inputs, target_in, target_out = batch
                inputs = inputs.to('cuda')
                target_in = target_in.to('cuda')
                target_out = target_out.to('cuda')
                target_mask = target_out != 0
                outputs = model(inputs, target_in)
                loss = torch.nn.functional.cross_entropy(outputs[target_mask], target_out[target_mask], reduction='sum')

                # Backward and optimize
                loss.backward()
                optim.step()
                if loss < best_loss:
                    no_improvement = 0
                    best_loss = loss
                else:
                    no_improvement += 0
                    if no_improvement > patience:
                        break
            if no_improvement > patience:
                break

            model.eval()

    print(f"Performance for {set_name}: Prequential code length: {code_length}")

    return model, code_length, num_preds, num_in


def prequential_coding_mirs(model, dataset, set_name, n_replay_streams, learning_rate, batch_size,
                            seed, alpha):
    """
    :param model:
    :param dataset:
    :param set_name:
    :param n_replay_streams:
    :param learning_rate:
    :param batch_size:
    :param seed:
    :param patience:
    :param alpha:
    :return:
    """
    # Initialize the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    beta = torch.nn.Parameter(torch.tensor(1.0))
    beta_optim = torch.optim.Adam([beta], lr=learning_rate)
    f_beta = torch.nn.Softplus()
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    model.to('cuda')

    # Set up LR scheduler
    code_length = 0
    num_in = 0
    num_preds = 0

    torch.manual_seed(seed)
    random.seed(seed)

    gen = torch.Generator()
    gen.manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=NUM_SAMPLES, generator=gen)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=PAD_COLLATE_FN, sampler=sampler)

    replay_streams = [0 for _ in range(n_replay_streams)]
    buffer = []

    model.train()
    ema_params = {}
    trained_params = {}
    for name, param in model.named_parameters():
        ema_params[name] = param.clone().detach()
        trained_params[name] = param.clone().detach()

    t = 0
    for batch in tqdm(dataloader, desc=f'Compressing dataset {set_name}: ', leave=False):
        buffer.append(batch)
        for name, param in model.named_parameters():
            trained_params[name] = param.clone().detach()
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(ema_params[name])
        beta_optim.zero_grad()
        inputs, target_in, target_out = batch
        inputs = inputs.to('cuda')
        target_in = target_in.to('cuda')
        target_out = target_out.to('cuda')
        target_mask = target_out != 0
        num_in += (inputs != 0).sum().item()
        num_preds += target_mask.sum().item()
        outputs = model(inputs, target_in) * f_beta(beta)

        loss = torch.nn.functional.cross_entropy(outputs[target_mask], target_out[target_mask], reduction='sum')
        code_length += loss.item()
        loss.backward()
        beta_optim.step()
        optim.zero_grad()
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(trained_params[name])

        outputs = model(inputs, target_in)
        loss = torch.nn.functional.cross_entropy(outputs[target_mask], target_out[target_mask], reduction='sum')
        loss.backward()
        optim.step()
        for name, param in model.named_parameters():
            ema_params[name] = ema_params[name]*(1-alpha) + param*alpha

        for k, replay_stream in enumerate(replay_streams):
            optim.zero_grad()

            inputs, target_in, target_out = buffer[replay_stream]
            inputs = inputs.to('cuda')
            target_in = target_in.to('cuda')
            target_out = target_out.to('cuda')
            target_mask = target_out != 0
            outputs = model(inputs, target_in)

            loss = torch.nn.functional.cross_entropy(outputs[target_mask], target_out[target_mask], reduction='sum')

            loss.backward()
            beta_optim.step()
            optim.step()
            for name, param in model.named_parameters():
                ema_params[name] = ema_params[name] * (1 - alpha) + param * alpha

            if random.random() < 1/(t+2):  # +2 accounts for zero-indexing AND having to use the next value of t
                # On first iteration half should reset, then on the second 1/3, then 1/4 etc...
                replay_streams[k] = 0
            else:
                replay_streams[k] += 1
        t += 1

    print(f"Performance for {set_name}: Prequential code length: {code_length}")

    return model, code_length, num_preds, num_in


def ordered_prequential_coding_mirs(model, dataset, set_name, n_replay_streams, learning_rate, batch_size,
                                    seed, alpha):
    """
    :param model:
    :param dataset:
    :param set_name:
    :param n_replay_streams:
    :param learning_rate:
    :param batch_size:
    :param seed:
    :param patience:
    :param alpha:
    :return:
    """
    # Initialize the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    beta = torch.nn.Parameter(torch.tensor(1.0))
    beta_optim = torch.optim.Adam([beta], lr=learning_rate)
    f_beta = torch.nn.Softplus()
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    model.to('cuda')

    # Set up LR scheduler
    code_length = 0
    num_in = 0
    num_preds = 0

    torch.manual_seed(seed)
    random.seed(seed)

    gen = torch.Generator()
    gen.manual_seed(seed)
    #sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=NUM_SAMPLES, generator=gen)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=PAD_COLLATE_FN, shuffle=False)

    replay_streams = [0 for _ in range(n_replay_streams)]
    buffer = []

    model.train()
    ema_params = {}
    trained_params = {}
    for name, param in model.named_parameters():
        ema_params[name] = param.clone().detach()
        trained_params[name] = param.clone().detach()

    t = 0
    for e, batch in tqdm(enumerate(dataloader), desc=f'Compressing dataset {set_name}: ', leave=False, total=len(dataloader)):
        buffer.append(batch)
        for name, param in model.named_parameters():
            trained_params[name] = param.clone().detach()
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(ema_params[name])
        beta_optim.zero_grad()
        inputs, target_in, target_out = batch
        inputs = inputs.to('cuda')
        target_in = target_in.to('cuda')
        target_out = target_out.to('cuda')
        target_mask = target_out != 0
        num_in += (inputs != 0).sum().item()
        num_preds += target_mask.sum().item()
        outputs = model(inputs, target_in) * f_beta(beta)

        loss = torch.nn.functional.cross_entropy(outputs[target_mask], target_out[target_mask], reduction='sum')
        code_length += loss.item()
        yield (e+1)*batch_size, code_length, num_preds, num_in
        loss.backward()
        beta_optim.step()
        optim.zero_grad()
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(trained_params[name])

        outputs = model(inputs, target_in)
        loss = torch.nn.functional.cross_entropy(outputs[target_mask], target_out[target_mask], reduction='sum')
        loss.backward()
        optim.step()
        for name, param in model.named_parameters():
            ema_params[name] = ema_params[name]*(1-alpha) + param*alpha

        for k, replay_stream in enumerate(replay_streams):
            optim.zero_grad()

            inputs, target_in, target_out = buffer[replay_stream]
            inputs = inputs.to('cuda')
            target_in = target_in.to('cuda')
            target_out = target_out.to('cuda')
            target_mask = target_out != 0
            outputs = model(inputs, target_in)

            loss = torch.nn.functional.cross_entropy(outputs[target_mask], target_out[target_mask], reduction='sum')

            loss.backward()
            beta_optim.step()
            optim.step()
            for name, param in model.named_parameters():
                ema_params[name] = ema_params[name] * (1 - alpha) + param * alpha

            if random.random() < 1/(t+2):  # +2 accounts for zero-indexing AND having to use the next value of t
                # On first iteration half should reset, then on the second 1/3, then 1/4 etc...
                replay_streams[k] = 0
            else:
                replay_streams[k] += 1
        t += 1



def find_ordered_prequential_code_lengths(dataset, shuffled_dataset, set_name, lr, model_type,
                                          model_args, epochs, batch_size, seed):
    torch.manual_seed(seed)

    torch.manual_seed(seed)

    if model_type == 'lstm':
        model = Seq2SeqLSTM(*model_args)
    elif model_type == 'cnn':
        model = AttnSeq2SeqCNN(*model_args)
    else:
        raise ValueError()

    model.init_weights()

    conditional_model = deepcopy(model)
    data_sizes = []
    code_lengths = []

    conditional_code_lengths = []
    num_predictions = []
    num_inputs = []

    for results in ordered_prequential_coding_mirs(model, shuffled_dataset, set_name + ' shuffled',
                                                   epochs, lr, batch_size, seed, ALPHA):
        data_size, code_length, num_preds, _ = results
        data_sizes.append(data_size)
        code_lengths.append(code_length)
        num_predictions.append(num_preds)

    for results in ordered_prequential_coding_mirs(conditional_model, dataset, set_name,
                                                   epochs, lr, batch_size, seed, ALPHA):
        _, conditional_code_length, _, num_in = results
        conditional_code_lengths.append(conditional_code_length)
        num_inputs.append(num_in)

    return code_lengths, conditional_code_lengths, data_sizes, num_predictions, num_inputs




def find_prequential_code_length(dataset, shuffled_dataset, set_name, lr, model_type,
                                 model_args, epochs, batch_size, seed, complete):
    '''
    :param dataset:
    :param shuffled_dataset:
    :param set_name:
    :param lr:
    :param model_args:
    :return:
    '''

    torch.manual_seed(seed)

    if model_type == 'lstm':
        model = Seq2SeqLSTM(*model_args)
    elif model_type == 'cnn':
        model = AttnSeq2SeqCNN(*model_args)
    else:
        raise ValueError()

    model.init_weights()

    conditional_model = deepcopy(model)

    if complete:
        _, code_length, _, _ = prequential_coding_block(model, shuffled_dataset, set_name + ' shuffled',
                                                           epochs, lr, batch_size, seed, STOP_POINTS)  # I know
        _, conditional_code_length, preds, ins = prequential_coding_block(conditional_model, dataset, set_name,
                                                                             epochs, lr, batch_size, seed, STOP_POINTS)

    else:
        _, code_length, _, _ = prequential_coding_mirs(model, shuffled_dataset, set_name + ' shuffled',
                                                       epochs, lr, batch_size, seed, ALPHA)
        _, conditional_code_length, preds, ins = prequential_coding_mirs(conditional_model, dataset, set_name,
                                                                         epochs, lr, batch_size, seed, ALPHA)

    return code_length, conditional_code_length, preds, ins


if __name__ == '__main__':
    # parse arguments (self descriptive)
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('-lang', '--languages', nargs='+', help='List of languages to include', default=None)

    parser.add_argument('-exclude_lang', '--exclude_languages', nargs='+', help='List of languages to exclude', default=None)

    parser.add_argument('-f', '--filename', type=str, help='Input filename', required=True)

    parser.add_argument('-type', '--dataset_type', type=str, choices=['orth', 'phon'],
                        help="Whether input data is orthography or phonology", required = True)

    parser.add_argument('-n', '--num_trials', type=int, help='Number of trials to run', required=True)

    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=1)

    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=1e-2)

    parser.add_argument('-dim', '--dimension', type=int, help='Hidden layer dimension', default=100)

    parser.add_argument('-dir', '--directory', type=str, help='Directory for Euralex data', default=None)

    parser.add_argument('-set', '--dataset_source', type=str, help='Source of data.',
                        choices=['euralex', 'leipzig', 'wikipron', 'lemma'])

    parser.add_argument('-N', '--num_samples', type=int, help='Number of samples from dataset.',
                        default=None)

    parser.add_argument('-b', '--broad', type=bool,  default=True,
                        help='Whether to prioritize broad or narrow IPA if both are availabe. Default is broad. ')

    parser.add_argument('-m', '--is_max', type=bool, default=False,
                        help='Whether or not to use the max file size available.')

    parser.add_argument('-sz', '--batch_size', type=int, default=64,
                        help='Size of batches used at each iteration.')

    parser.add_argument('-tar', '--target_size', type=str, default='100K',
                        choices=['30K', '100K', '300K', '1M'], help='Which size of file to use (if not is_max).')

    parser.add_argument('-tok', '--token_cutoff', type=int, default=250_000,
                        help='Minimum size of dataset by tokens.')

    parser.add_argument('-typ', '--type_cutoff', type=int, default=10_000,
                        help='Minimum size of dataset by types.')

    parser.add_argument('-mod', '--model_type', type=str, choices=['lstm', 'cnn'],
                        help='Type of model to use.')

    parser.add_argument('-fil', '--filters', type=int, default=20,
                        help='Number of CNN filters.')

    parser.add_argument('-ker', '--kernel', type=int, default=3,
                        help='Size of CNN kernel.')

    parser.add_argument('-ly', '--layers', type=int, default=2,
                        help='Number of LSTM or CNN layers used in model.')

    parser.add_argument('-bl', '--block', type=bool, default=False)

    parser.add_argument('-sp', '--stop_points', type=float, nargs='+',
                        default=list(reversed([1/(2**n) for n in range(1, 10)])))

    parser.add_argument('-a', '--alpha', help='Polyak average coefficient (for MIRS).',
                        type=float, default=.01)

    parser.add_argument('-o', '--ordered', help='Whether dataset is ordered and taken granularly.',
                        type=bool, default=True)

    args = parser.parse_args()


    #breakpoint()

    # assign arguments to constants where useful
    NUM_EPOCHS = args.epochs
    LR = args.learning_rate
    DIM = args.dimension
    FILTERS = args.filters
    MODEL_TYPE = args.model_type
    BATCH_SIZE = args.batch_size
    KERNEL_SIZE = args.kernel
    LAYERS = args.layers
    STOP_POINTS = args.stop_points
    BLOCK = args.block
    ALPHA = args.alpha
    ORDERED = args.ordered

    INF = float('inf')

    if args.directory is None:
        if args.dataset_source == 'leipzig':
            DIRECTORY = LEIPZIG_DIR
            large = False
        elif args.dataset_source == 'wikipron':
            DIRECTORY = WIKIPRON_DIR
            large = False
        elif args.dataset_source == 'lemma':
            DIRECTORY = LEMMA_DIR
            large = False
        else:
            DIRECTORY = EURALEX_DIR
            large = False
    else:
        DIRECTORY = args.directory

    if args.dataset_source == 'leipzig':
        datasets = make_leipzig_datasets(DIRECTORY, args.languages, args.dataset_type, args.broad, args.is_max,
                                         args.target_size, args.token_cutoff, args.type_cutoff, args.exclude_languages)
    elif args.dataset_source == 'wikipron':
        datasets = make_wikipron_sets(DIRECTORY, args.languages, args.dataset_type, args.type_cutoff, args.exclude_languages)

    elif args.dataset_source == 'lemma':
        datasets = make_lemma_datasets(DIRECTORY, args.languages, args.dataset_type, args.broad, 
                                      args.token_cutoff, args.type_cutoff, args.exclude_languages)

    if args.num_samples is None:
        NUM_SAMPLES = min(map(len, datasets.values()))
    else:
        NUM_SAMPLES = args.num_samples

    print(f"Preparing to compress {NUM_SAMPLES} words for each dataset.")
    print(f'Languages to compress are {", ".join([k for k in datasets.keys()])}.')


    with open(args.filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not ORDERED:
            csv_writer.writerow(['language', 'script', 'trial number', 'best conditional complexity', 'best complexity',
                                 'mutual algorithmic information', 'tokens', 'types', 'transcription type',
                                 'total prediction characters', 'total input characters'])
        else:
            csv_writer.writerow(['data size', 'language', 'script', 'trial number', 'best conditional complexity',
                                 'best complexity', 'mutual algorithmic information', 'tokens', 'types',
                                 'transcription type', 'total prediction characters', 'total input characters'])
        csvfile.flush()

    for lang, dataset in tqdm(reversed(sorted(datasets.items(), key=lambda x: x[0])),
                              desc=f'Compressing {len(datasets)} language datasets.'):
        control_dataset = dataset.control()
        word_total = len(dataset)

        for i in range(args.num_trials):
            if not ORDERED:
                alphabet_sz = len(dataset.tokenizer) + 3  # length of tokenizer plus BOS, EOS, and padding tokens

                if MODEL_TYPE == 'lstm':
                    model_args = (alphabet_sz, DIM, LAYERS, LAYERS)
                else:
                    model_args = (alphabet_sz, DIM, KERNEL_SIZE, LAYERS)

                returned_values = find_prequential_code_length(dataset, control_dataset, lang, LR, MODEL_TYPE, model_args,
                                                               NUM_EPOCHS, BATCH_SIZE, i, BLOCK)
                complexity, conditional_complexity, preds, num_in = returned_values

                with open(args.filename, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([lang, dataset.script, i, conditional_complexity,
                                         complexity, complexity-conditional_complexity,
                                         word_total, len(dataset.words), dataset.transcription_type, preds, num_in])

                    csvfile.flush()

            # TODO separate frequency counting and ordering processes
            else:
                dataset.frequency_sorted()
                dataset.delete_counts()
                dataset.prune(NUM_SAMPLES)
                control_dataset.frequency_sorted()
                control_dataset.delete_counts()
                control_dataset.prune(NUM_SAMPLES)
                alphabet_sz = len(dataset.tokenizer) + 3  # length of tokenizer plus BOS, EOS, and padding tokens

                if MODEL_TYPE == 'lstm':
                    model_args = (alphabet_sz, DIM, LAYERS, LAYERS)
                else:
                    model_args = (alphabet_sz, DIM, KERNEL_SIZE, LAYERS)


                returned_values = find_ordered_prequential_code_lengths(dataset, control_dataset, lang, LR, MODEL_TYPE,
                                                                        model_args, NUM_EPOCHS, BATCH_SIZE, i)

                code_lengths, conditional_code_lengths, data_sizes, num_predictions, num_inputs = returned_values

                with open(args.filename, 'a', newline='') as csvfile:
                    for complexity, conditional_complexity, data_sz, preds, num_in in zip(*returned_values):
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([data_sz, lang, dataset.script, i, conditional_complexity,
                                             complexity, complexity - conditional_complexity,
                                             word_total, len(dataset.words), dataset.transcription_type,
                                             preds, num_in])

                        csvfile.flush()


# Commands to run

# python opacity-measurement.py -f wikipron_massively_multilingual_orth_lstm.csv -n 40 -type orth -typ 5000 -N 5000 -set wikipron -sz 4 -mod lstm
# python opacity-measurement.py -f wikipron_massively_multilingual_phon_lstm.csv -n 40 -type phon -typ 5000 -N 5000 -set wikipron -sz 4 -mod lstm
# python opacity-measurement.py -f wikipron_massively_multilingual_orth_cnn.csv -n 20 -type orth -typ 5000 -N 5000 -set wikipron -sz 4 -mod cnn
# python opacity-measurement.py -f wikipron_massively_multilingual_phon_cnn.csv -n 20 -type phon -typ 5000 -N 5000 -set wikipron -sz 4 -mod cnn


# ordered results

# python opacity-measurement.py -f granular_orth_cnn.csv -n 10 -type orth -N 100000 -set leipzig -sz 64 -mod cnn -o True -e 25
# python opacity-measurement.py -f granular_phon_cnn.csv -n 10 -type phon  -N 100000 -set leipzig -sz 64 -mod cnn -o True -e 25

# ordered results (multi-layer)

# python opacity-measurement.py -f granular_orth_cnn.csv -n 10 -type orth -N 100000 -set leipzig -sz 64 -mod cnn -o True -e 25 -ly 5 -ker 5
# python opacity-measurement.py -f granular_phon_cnn.csv -n 10 -type phon  -N 100000 -set leipzig -sz 64 -mod cnn -o True -e 25 -ly 5 -ker 5



# python opacity-measurement.py -f lemma_orth_cnn_kern3.csv -n 20 -type orth -set lemma -mod cnn
# python opacity-measurement.py -f lemma_orth_cnn_kern5.csv -n 20 -type orth -set lemma -mod cnn -ker 5
# python opacity-measurement.py -f lemma_orth_cnn_kern10.csv -n 20 -type orth -set lemma -mod cnn -ker 10

# python opacity-measurement.py -f lemma_phon_cnn_kern3.csv -n 20 -type phon -set lemma -mod cnn
# python opacity-measurement.py -f lemma_phon_cnn_kern5.csv -n 20 -type phon -set lemma -mod cnn -ker 5
# python opacity-measurement.py -f lemma_phon_cnn_kern10.csv -n 20 -type phon -set lemma -mod cnn -ker 10
