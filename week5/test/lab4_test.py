from src.representations.word_embedder import word_embedder

def test_word_embedder():
    embedder = word_embedder("glove-wiki-gigaword-50")
    
    king_vector = embedder.get_vector("king")
    print("Vector for 'king':", king_vector)

    king_queen_similarity = embedder.get_similarity("king", "queen")
    print("Similarity between 'king' and 'queen':", king_queen_similarity)

    king_top_similar = embedder.get_most_similar("computer", top_n=10)
    print("Top 10 words similar to 'computer':", king_top_similar)

    test_sequence = 'The queen rules the country.'
    document_vector = embedder.embed_document(test_sequence)
    print("Document vector (first 10 values):", document_vector[:10])

if __name__ == "__main__":
    test_word_embedder()
