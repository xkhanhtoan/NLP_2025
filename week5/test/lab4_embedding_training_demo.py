from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import multiprocessing, os

DATA_PATH = "C:/Users/Z/Desktop/2025-code/NLP/week5/data/en_ewt-ud-dev.txt"
sentences = [simple_preprocess(line) for line in open(DATA_PATH, encoding="utf-8") if line.strip() and not line.startswith("#")]
print(f" Loaded {len(sentences)} sentences from {DATA_PATH}")

model = Word2Vec(
    sentences=sentences,
    vector_size=200,       
    window=7,              
    min_count=3,         
    sg=1,              
    negative=10,         
    sample=1e-4,        
    epochs=15,         
    workers=multiprocessing.cpu_count() - 1,  
)

os.makedirs("results", exist_ok=True)
MODEL_PATH = "results/word2vec_ewt_optimized.model"
model.save(MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}\n")

if "king" in model.wv:
    print("\nVector for 'king' (first 10 dims):")
    print(model.wv["king"][:10])
else:
    print("'king' not in vocabulary.")

try:
    print("\nTop 5 words similar to 'king':")
    for w, s in model.wv.most_similar("king", topn=5):
        print(f"  {w:10s} -> {s:.4f}")
except KeyError:
    print("'king' not in vocabulary.")

try:
    print("\nAnalogy: king - man + woman ≈ ?")
    for w, s in model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=5):
        print(f"  {w:10s} -> {s:.4f}")
except KeyError:
    print("One of the words not in vocabulary.")

log_path = os.path.join(os.path.dirname(__file__), "..", "results", "lab4_bonus_summary.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"Vocabulary size: {len(model.wv)}\n")
    f.write(f"Vector size: {model.vector_size}\n")
    f.write(f"Window: {model.window}\n")
    f.write(f"Epochs: {model.epochs}\n")
    f.write(f"Training sentences: {len(sentences)}\n")
    f.write("\nSample vector for 'king':\n")
    if "king" in model.wv:
        f.write(str(model.wv["king"][:10]) + "\n")
    f.write("\nTop 5 similar to 'king':\n")
    try:
        for w, s in model.wv.most_similar("king", topn=5):
            f.write(f"  {w:10s} -> {s:.4f}\n")
    except KeyError:
        f.write("  'king' not in vocabulary\n")
print("\nSummary saved to results/lab4_bonus_summary.txt")