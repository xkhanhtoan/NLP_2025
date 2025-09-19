import re
from typing import List
from lab1.src.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return self.pattern.findall(text)
