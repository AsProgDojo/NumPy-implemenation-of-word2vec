from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PreprocessedCorpus:
    sentences_as_ids: list[list[int]]
    word_to_id: dict[str, int]
    id_to_word: dict[int, str]
    word_counts: Counter


def read_text(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def split_into_sentences(text: str) -> list[str]:
    text = text.lower()
    parts = re.split(r"[.!?\n]+", text)
    return [part.strip() for part in parts if part.strip()]


def tokenize(sentence: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", sentence)


def preprocess_corpus(file_path: str) -> PreprocessedCorpus:
    raw_text = read_text(file_path)
    raw_sentences = split_into_sentences(raw_text)

    tokenized_sentences: list[list[str]] = []
    for sentence in raw_sentences:
        tokens = tokenize(sentence)
        if tokens:
            tokenized_sentences.append(tokens)

    all_tokens = [token for sentence in tokenized_sentences for token in sentence]
    full_counts = Counter(all_tokens)

    kept_words = [word for word, count in full_counts.items()]
    kept_words.sort(key=lambda word: (-full_counts[word], word))

    word_to_id = {word: idx for idx, word in enumerate(kept_words)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    sentences_as_ids: list[list[int]] = []
    filtered_counts = Counter()

    for sentence in tokenized_sentences:
        encoded_sentence = []

        for token in sentence:
            if token in word_to_id:
                token_id = word_to_id[token]
                encoded_sentence.append(token_id)
                filtered_counts[token] += 1

        if encoded_sentence:
            sentences_as_ids.append(encoded_sentence)

    return PreprocessedCorpus(
        sentences_as_ids=sentences_as_ids,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        word_counts=filtered_counts,
    )


if __name__ == "__main__":
    corpus = preprocess_corpus("The bridge on the Drina.txt")

    print("Vocabulary size:", len(corpus.word_to_id))
    print("Number of sentences:", len(corpus.sentences_as_ids))

    if corpus.sentences_as_ids:
        first_sentence_ids = corpus.sentences_as_ids[0]
        first_sentence_words = [corpus.id_to_word[word_id] for word_id in first_sentence_ids]

        print("First encoded sentence:", first_sentence_ids)
        print("First decoded sentence:", first_sentence_words)

    print("Top 10 words:")
    for word, count in corpus.word_counts.most_common(10):
        print(f"{word}: {count}")