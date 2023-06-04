from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch

def get_vocab(dataset):
    train_iter = dataset(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab, tokenizer

class APCDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, vocab, tokenizer, chunk_size):
        super().__init__()
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.dataset_iter = iter(dataset)
        self.chunk_size = chunk_size
        self.current_chunk_idx = 0
        self.current_chunk = []
    
    def _next_chunk_iter(self):
        for item in self.dataset_iter:
            chunk = torch.tensor(self.vocab(self.tokenizer(item)))
            if len(chunk) > self.chunk_size:
                yield chunk       
    
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        while True:
            if len(self.current_chunk)==0:
                self.current_chunk = next(self._next_chunk_iter())
            if self.current_chunk_idx+self.chunk_size+1 > len(self.current_chunk):
                data = self.current_chunk[self.current_chunk_idx:-1]
                target = self.current_chunk[self.current_chunk_idx+1:]
                self.current_chunk = next(self._next_chunk_iter())
                self.current_chunk_idx = 0
            else:
                data = self.current_chunk[self.current_chunk_idx:self.current_chunk_idx+self.chunk_size]
                target = self.current_chunk[self.current_chunk_idx+1:self.current_chunk_idx+self.chunk_size+1]
                self.current_chunk_idx+=self.chunk_size
        
            yield data, target