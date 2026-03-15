from __future__ import annotations
import numpy as np

try:
    from preprocess import PreprocessedCorpus
except ImportError:
    from preprocess import PreprocessedCorpus


class SGNSDataset:
    def __init__(
        self,
        corpus: PreprocessedCorpus,
        window_size: int = 2,
        num_negative_samples: int = 5,
        seed: int = 28,
    ) -> None:
        self.corpus = corpus
        self.window_size = window_size
        self.num_negative_samples = num_negative_samples
        self.rng = np.random.default_rng(seed)

        self.pairs = self._build_positive_pairs()
        self.negative_sampling_probs = self._build_negative_sampling_distribution()

    def _build_positive_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []

        for sentence in self.corpus.sentences_as_ids:
            sentence_length = len(sentence)

            for center_index in range(sentence_length):
                center_word_id = sentence[center_index]

                left = max(0, center_index - self.window_size)
                right = min(sentence_length, center_index + self.window_size + 1)

                for context_index in range(left, right):
                    if context_index == center_index:
                        continue

                    context_word_id = sentence[context_index]
                    pairs.append((center_word_id, context_word_id))

        return pairs

    def _build_negative_sampling_distribution(self) -> np.ndarray:
        vocabulary_size = len(self.corpus.word_to_id)
        counts = np.zeros(vocabulary_size, dtype=np.float64)

        for word, count in self.corpus.word_counts.items():
            word_id = self.corpus.word_to_id[word]
            counts[word_id] = count

        counts = counts ** 0.75
        probabilities = counts / counts.sum()
        return probabilities

    def sample_negative_words(self, center_word_id: int, context_word_id: int) -> np.ndarray:
        negative_word_ids: list[int] = []

        while len(negative_word_ids) < self.num_negative_samples:
            sampled_id = int(
                self.rng.choice(
                    len(self.negative_sampling_probs),
                    p=self.negative_sampling_probs,
                )
            )

            if sampled_id == center_word_id or sampled_id == context_word_id:
                continue

            negative_word_ids.append(sampled_id)

        return np.array(negative_word_ids, dtype=np.int64)

    def get_training_example(self, index: int) -> tuple[int, int, np.ndarray]:
        center_word_id, context_word_id = self.pairs[index]
        negative_word_ids = self.sample_negative_words(center_word_id, context_word_id)
        return center_word_id, context_word_id, negative_word_ids

    def __len__(self) -> int:
        return len(self.pairs)


if __name__ == "__main__":
    try:
        from .preprocess import preprocess_corpus
    except ImportError:
        from preprocess import preprocess_corpus

    corpus = preprocess_corpus("The bridge on the Drina.txt")
    dataset = SGNSDataset(
        corpus=corpus,
        window_size=2,
        num_negative_samples=4,
        seed=28,
    )

    print("Number of positive pairs:", len(dataset))

    for i in range(min(5, len(dataset))):
        center_word_id, context_word_id, negative_word_ids = dataset.get_training_example(i)

        center_word = corpus.id_to_word[center_word_id]
        context_word = corpus.id_to_word[context_word_id]
        negative_words = [corpus.id_to_word[word_id] for word_id in negative_word_ids]

        print()
        print("Center word:", center_word)
        print("Context word:", context_word)
        print("Negative samples:", negative_words)