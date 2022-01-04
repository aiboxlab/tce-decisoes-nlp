import json
import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
import spacy
from htmlBuilder.attributes import Style as InlineStyle
from htmlBuilder.tags import *
from spacy import displacy
from spacy.matcher import Matcher
from typing import List, Iterable
from ..tce_nlp.tools.ner.regex_ner import get_regex_dict
from ..tce_nlp.tools.preprocess.text_preprocess import preprocess_pipeline
from ..tce_nlp.tools.summarization import py_summarize
from ..tce_nlp.util.util import PATTERNS

nltk.download('rslp')

sentiment = open("app/tce_nlp/resources/SentimentAnalysis.sav", "rb")
sentiment_classifier = pickle.load(sentiment)

topic = open("app/tce_nlp/resources/SentimentAnalysis.sav", "rb")
topic_classifier = pickle.load(topic)


def get_ner_dict(text, from_regex=True, patterns=None):
    if patterns is None:
        patterns = []

    if from_regex:
        return get_regex_dict(text, patterns)

    return {}


def displacy_ner(text):
    regex_ner = get_ner_dict(text, patterns=PATTERNS)
    html = displacy.render(regex_ner, style="ent", manual=True)
    return html


class Document:
    nlp_model = spacy.load('pt_core_news_lg')
    word_dict = json.load(open('app/tce_nlp/resources/glossario.json'))
    sentiments_model = None
    topic_model = None

    def __init__(self, raw_text: str):
        self.topics = []
        self.sentiments = []
        self.sents = []
        self.summary = None
        self.text = self.pre_process(raw_text)
        self.spacy_doc = self.nlp_model(self.text.lower())
        self.word_dict = dict([(key.lower(), value) for key, value in self.word_dict.items()])
        self.spacy_matcher = self.get_glossary_matcher()
        self.words_matches = set()

    def pre_process(self, raw_text: str):

        self.sents = preprocess_pipeline(raw_text)
        self.text = ' '.join(self.sents)
        return self.text

    def get_text(self):
        return self.text

    def get_summary(self, sent_count: int = 3):
        """
        :return: The summarization of all the content.
        """

        self.summary = py_summarize(self.text, sent_count)

        return self.summary

    def get_legal_entities(self):
        raise NotImplemented

    def get_glossary_matcher(self):
        self.spacy_matcher = Matcher(self.nlp_model.vocab, validate=True)
        # Add match ID "HelloWorld" with no callback and one pattern

        for key in self.word_dict.keys():
            pattern = []
            for token in self.nlp_model(key.lower()):
                pattern.append({"LEMMA": token.text})

            self.spacy_matcher.add(key.lower(), [pattern])

        return self.spacy_matcher

    def get_dictionary(self):

        glossario_df = pd.DataFrame.from_dict(self.word_dict, orient='index').T

        strings_id = []

        for match_id, start, end in self.spacy_matcher(self.spacy_doc):
            span = self.spacy_doc[start:end]  # The matched span
            string_id = self.nlp_model.vocab.strings[match_id]  # Get string representation
            strings_id.append(string_id)
            self.words_matches.add(span.text)

        matched_dict = glossario_df.loc[:, strings_id]
        if matched_dict.size > 0:
            return matched_dict.to_dict('records')[0]
        return {}

    # def get_glossary_html(self):
    #     entities = {
    #         "text": self.spacy_doc.text,
    #         "ents": [],
    #         "title": None,
    #     }
    #     for match_id, start, end in self.spacy_matcher(self.spacy_doc):
    #         entities["ents"].append({
    #             "start": start,
    #             "end": end,
    #             "label": ""
    #         })
    #
    #     return displacy.render([entities], style="ent", manual=True)

    def get_glossary_html(self):

        v_starts = np.array([-1])
        html_sentences = []
        self.get_dictionary()

        for sentence in self.text.split(';'):
            sentence = sentence.strip().split()
            if len(sentence) > 0:
                sentence[0] = sentence[0].capitalize()
                html_text = ' '.join(sentence) + '.'

                for pattern in sorted(self.words_matches, reverse=True):

                    search = re.search(r'(\(|\s)' + pattern + r'(\s|\.|,|\)|-)', html_text, flags=re.IGNORECASE)
                    if search:

                        start, end = search.span()
                        if np.any(v_starts <= start) and start < end:
                            np.insert(v_starts, 0, start)
                            html_text = self.build_html(html_text, start, end)
                    else:
                        continue

                html_sentences.append(html_text)
        return ' '.join(html_sentences)

    def get_sentiments(self, sentences: Iterable[str]):

        for sentence in sentences:
            classification = sentiment_classifier.predict([sentence])

            self.sentiments.append({"sentence": sentence, "sentiments": classification[0]})

        return self.sentiments

    def get_topics(self, sentences: Iterable[str]):

        for sentence in sentences:
            classification = topic_classifier.predict([sentence])
            self.topics.append({"sentence": sentence, "topic": classification[0]})

        return self.topics

    @staticmethod
    def build_html(text_: str, start: int, end: int, background='#343a40', color='white'):
        start += 1
        end -= 1
        pre_text = text_[:start]
        pos_text = text_[end:]

        html = Div([
            InlineStyle(line_height='2.5', direction='ltr')],
            pre_text,
            Mark([
                InlineStyle(background=background, color=color,
                            padding='0.45em', margin='0.25em',
                            line_height=1, border_radius='0.35em')], text_[start:end]),
            pos_text,
        )

        return html.render()

    def get_features_dict(self):

        self.get_summary(3)

        html_summary = []
        for idx in range(len(self.summary)):
            html_summary.append(displacy_ner(self.summary[idx]))

        return {
            "text": self.get_text(),
            "dictionary": self.get_dictionary(),
            "html_summary": html_summary,
            "summary": self.summary,
            "html": self.get_glossary_html(),
            "sentiments": self.get_sentiments(self.summary),
            "topics": self.get_topics(self.summary)
        }
