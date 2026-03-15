from __future__ import annotations

import numpy as np

try:
    from dataset import SGNSDataset
    from model import Word2VecSGNS
    from preprocess import preprocess_corpus
except ImportError:
    from dataset import SGNSDataset
    from model import Word2VecSGNS
    from preprocess import preprocess_corpus


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)) + 1e-12
    return float(np.dot(vector_a, vector_b) / denominator)


def find_nearest_words(
    query_word: str,
    embeddings: np.ndarray,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    if query_word not in word_to_id:
        return []

    query_id = word_to_id[query_word]
    query_vector = embeddings[query_id]

    similarities: list[tuple[str, float]] = []

    for word_id in range(len(id_to_word)):
        if word_id == query_id:
            continue

        word = id_to_word[word_id]
        similarity = cosine_similarity(query_vector, embeddings[word_id])
        similarities.append((word, similarity))

    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_k]


def main() -> None:
    data_path = "The bridge on the Drina.txt"

    window_size = 2
    num_negative_samples = 5
    embedding_dim = 50
    learning_rate = 0.025
    num_epochs = 5
    seed = 28

    corpus = preprocess_corpus(data_path)

    if len(corpus.word_to_id) == 0:
        raise ValueError("Vocabulary is empty. Try lowering min_count or using a larger corpus.")

    dataset = SGNSDataset(
        corpus=corpus,
        window_size=window_size,
        num_negative_samples=num_negative_samples,
        seed=seed,
    )

    if len(dataset) == 0:
        raise ValueError("No training pairs were generated. Check preprocessing and corpus size.")

    model = Word2VecSGNS(
        vocabulary_size=len(corpus.word_to_id),
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        seed=seed,
    )

    rng = np.random.default_rng(seed)

    print("Vocabulary size:", len(corpus.word_to_id))
    print("Number of training pairs:", len(dataset))
    print()

    for epoch in range(num_epochs):
        shuffled_indices = rng.permutation(len(dataset))
        total_loss = 0.0

        for pair_index in shuffled_indices:
            center_word_id, context_word_id, negative_word_ids = dataset.get_training_example(pair_index)

            loss = model.train_step(
                center_word_id=center_word_id,
                context_word_id=context_word_id,
                negative_word_ids=negative_word_ids,
            )
            total_loss += loss

        average_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} - average loss: {average_loss:.4f}")

    print("\nNearest neighbors after training:")
    embeddings = model.get_word_embeddings()

    query_words = ['bridge', 'drina', 'river', 'town', 'village', 'woman', 'night', 'road']

    for query_word in query_words:
        neighbors = find_nearest_words(
            query_word=query_word,
            embeddings=embeddings,
            word_to_id=corpus.word_to_id,
            id_to_word=corpus.id_to_word,
            top_k=5,
        )

        print(f"\n{query_word}:")
        for neighbor_word, similarity in neighbors:
            print(f"  {neighbor_word:<15} {similarity:.4f}")


if __name__ == "__main__":
    main()