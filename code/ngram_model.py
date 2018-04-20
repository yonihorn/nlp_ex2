#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv
import math

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                      index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)


def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0

    for sen in dataset:
        for i in range(2, len(sen)):
            unigram_counts[sen[i]] = unigram_counts.get(sen[i], 0) + 1
            # if i > 1:
            bigram = (sen[i - 1], sen[i])
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            # if i > 2:
            trigram = (sen[i - 2], sen[i - 1], sen[i])
            trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1
            token_count += 1

    return trigram_counts, bigram_counts, unigram_counts, token_count


def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    perplexity = 0
    lambda3 = 1 - lambda1 - lambda2

    logp = float(0)
    M = 0
    for sen in eval_dataset:
        for i in range(2, len(sen)):
            M += 1
            q1, q2, q3 = 0, 0, 0
            if (sen[i - 2], sen[i - 1], sen[i]) in trigram_counts and (sen[i - 2], sen[i - 1]) in bigram_counts:
                q1 = trigram_counts[(sen[i - 2], sen[i - 1], sen[i])] / float(bigram_counts[(sen[i - 2], sen[i - 1])])

            if (sen[i - 1], sen[i]) in bigram_counts and sen[i - 1] in unigram_counts:
                q2 = bigram_counts[(sen[i - 1], sen[i])] / float(unigram_counts[sen[i - 1]])

            if sen[i] in unigram_counts:
                q3 = unigram_counts[sen[i]] / float(train_token_count)

            logqi = np.log2(lambda1 * q1 + lambda2 * q2 + lambda3 * q3)
            logp += logqi

    l = logp / M
    perplexity = np.exp2(-l)

    return perplexity


def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    # Some examples of functions usage
    # test_train = [[1, 2, 3, 4, 5], [2, 3, 5], [2, 3, 4]]

    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)

    best_lambdas = (0.5, 0.4)
    min_perplexity = perplexity
    for lambda1 in np.arange(0, 1.01, 0.01):
        for lambda2 in np.arange(0, 1.01-lambda1, 0.01):
            perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, lambda1,
                                         lambda2)
            # print "#Calculating preplexity for (lambda1, lambda2) = " + str(
            #     (lambda1, lambda2)) + ", preplexity = " + str(perplexity)
            if perplexity < min_perplexity:
                min_perplexity, best_lambdas = perplexity, (lambda1, lambda2)
                # print "#update min perplexity: " + str(min_perplexity) + ", got for (lambda1, lambda2) = " + str(
                #     best_lambdas)

    print "#min perplexity: " + str(min_perplexity) + ", got for (lambda1, lambda2) = " + str(best_lambdas)


if __name__ == "__main__":
    test_ngram()
