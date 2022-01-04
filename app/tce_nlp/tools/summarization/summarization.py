import re

import nltk
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer

from sumy.nlp.stemmers import Stemmer

nltk.download('punkt')
nltk.download('stopwords')

language = 'portuguese'

stemmer = Stemmer(language)


def preprocess_sentence(sentence):
    sentence = sentence.strip()
    new_sentence = re.sub(r'(Considerando\sque,|Considerando\sque|Considerando)', '', sentence).strip().split()
    new_sentence[0] = new_sentence[0].capitalize()

    return ' '.join(new_sentence).replace(';', '.').replace('..', '.')


def py_summarize(document, n):
    """
       The function summarizes a text string
       :param document: a text string
       :param n: The number of sentences in the final summarization

       :return: the text summarized according to the number of sentences
       """
    # Object of automatic summarization.
    auto_abstractor = AutoAbstractor()
    # Set tokenizer.

    auto_abstractor.tokenizable_doc = SimpleTokenizer()
    # Set delimiter for making a list of sentence.
    auto_abstractor.delimiter_list = [";", r". "]
    # Object of abstracting and filtering document.
    auto_abstractor.set_top_sentences(n)
    abstractable_doc = TopNRankAbstractor()
    abstractable_doc.top_n = n
    # Summarize document.
    result_dict = auto_abstractor.summarize(document.strip(), abstractable_doc)

    sentences = result_dict["summarize_result"]

    # Output result.
    return list(map(preprocess_sentence, sentences))


def summarize_batch(raw_text_list, sentence_count):
    """
    Summarizer a batch of documents
    :param raw_text_list: a list of document strings
    :param sentence_count: The number of sentences in the final summarization
    :return: all the text summarized.
    """

    for raw_text in raw_text_list:
        summary = py_summarize(raw_text, sentence_count)
        yield ''.join(summary)
