from lab1.src.preprocessing.simple_tokenizer import SimpleTokenizer
from lab1.src.preprocessing.regex_tokenizer import RegexTokenizer
from lab1.src.core.dataset_loaders import load_raw_text_data

def show(title: str, tokens: list[str], n: int = 20):
    print(f"{title}: {tokens[:n]}")

if __name__ == "__main__":
    # Task 1 & 2: 
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    tests = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!",
    ]

    print("\n=== Task 1 & 2: Evaluation ===")
    for s in tests:
        print(f"\nInput: {s}")
        show("SimpleTokenizer", simple_tokenizer.tokenize(s))
        show("RegexTokenizer", regex_tokenizer.tokenize(s))

    # Task 3: 
    dataset_path = r"C:\Users\Z\Desktop\2025-code\NLP\week1\lab1\src\data\en_ewt-ud-dev.txt"

    raw_text = load_raw_text_data(dataset_path)
    sample_text = raw_text[:500]

    print("\n=== Task 3: UD_English-EWT ===")
    print(f"Original Sample (first 100 chars): {sample_text[:100]}...")

    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20): {simple_tokens[:20]}")

    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20): {regex_tokens[:20]}")