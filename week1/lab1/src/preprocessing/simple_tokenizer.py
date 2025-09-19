import string
from typing import List
from lab1.src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        for p in string.punctuation:
            text = text.replace(p, f" {p} ")
        return text.split()
