import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        text = set()
        for t in texts:
            for w in t.split():
                text.add(w.lower())

        texts = sorted(text)

        self.word_to_id[self.pad_token] = 0
        self.id_to_word[0] = self.pad_token
        
        self.word_to_id[self.bos_token] = 2
        self.id_to_word[2] = self.bos_token
        
        self.word_to_id[self.unk_token] = 1
        self.id_to_word[1] = self.unk_token
        
        self.word_to_id[self.eos_token] = 3
        self.id_to_word[3] = self.eos_token

        id = 4
        for text in texts:
            self.word_to_id[text] = id
            self.id_to_word[id] = text
            id += 1

        self.vocab_size = len(texts) + 4
            
        
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        texts = text.lower().split()
        token_ids = []
        
        for t in texts:
            token_ids.append(self.word_to_id[t] if t in self.word_to_id else self.word_to_id[self.unk_token])
            
        return token_ids
        
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        special = {
            self.word_to_id[self.pad_token],
            self.word_to_id[self.bos_token],
            self.word_to_id[self.eos_token]
        }
        
        words = []
        for token_id in ids:
            if token_id not in special:
                words.append(self.id_to_word[token_id] if token_id in self.id_to_word else self.unk_token)
        
        return " ".join(words)
            
