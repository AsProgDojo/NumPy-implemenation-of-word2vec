from __future__ import annotations

import numpy as np


class Word2VecSGNS:
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 100,
        learning_rate: float = 0.025,
        seed: int = 28
            ,
    ) -> None:
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        rng = np.random.default_rng(seed)

        self.input_embeddings = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(vocabulary_size, embedding_dim),
        )

        self.output_embeddings = np.zeros(
            (vocabulary_size, embedding_dim),
            dtype=np.float64,
        )

    @staticmethod
    def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
        x = np.asarray(x, dtype=np.float64)

        positive_mask = x >= 0
        negative_mask = ~positive_mask

        result = np.empty_like(x, dtype=np.float64)
        result[positive_mask] = 1.0 / (1.0 + np.exp(-x[positive_mask]))

        exp_x = np.exp(x[negative_mask])
        result[negative_mask] = exp_x / (1.0 + exp_x)

        if result.ndim == 0:
            return float(result)

        return result

    def train_step(
        self,
        center_word_id: int,
        context_word_id: int,
        negative_word_ids: np.ndarray,
    ) -> float:
        """
        Perform one SGNS update step for:
            center word
            positive context word
            several negative words

        Returns:
            scalar loss for this training example
        """
        center_vector = self.input_embeddings[center_word_id].copy()
        positive_vector = self.output_embeddings[context_word_id].copy()
        negative_vectors = self.output_embeddings[negative_word_ids].copy()

        positive_score = np.dot(positive_vector, center_vector)
        negative_scores = negative_vectors @ center_vector

        positive_probability = self._sigmoid(positive_score)
        negative_probabilities = self._sigmoid(negative_scores)

        loss = -np.log(positive_probability + 1e-12)
        loss -= np.sum(np.log(1.0 - negative_probabilities + 1e-12))

        positive_score_gradient = positive_probability - 1.0
        negative_score_gradients = negative_probabilities

        center_gradient = positive_score_gradient * positive_vector
        center_gradient += negative_vectors.T @ negative_score_gradients

        positive_vector_gradient = positive_score_gradient * center_vector
        negative_vector_gradients = negative_score_gradients[:, np.newaxis] * center_vector

        self.input_embeddings[center_word_id] -= self.learning_rate * center_gradient
        self.output_embeddings[context_word_id] -= self.learning_rate * positive_vector_gradient

        np.add.at(
            self.output_embeddings,
            negative_word_ids,
            -self.learning_rate * negative_vector_gradients,
        )

        return float(loss)

    def get_word_embeddings(self) -> np.ndarray:
        """
        Return the learned word embeddings.
        For SGNS, we use the input embeddings as the main word vectors.
        """
        return self.input_embeddings