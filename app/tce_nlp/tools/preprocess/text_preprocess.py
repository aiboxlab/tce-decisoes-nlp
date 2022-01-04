# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
from bs4 import BeautifulSoup
from cffi.backend_ctypes import unicode
import bleach

import re
import nltk
from nltk import word_tokenize
import spacy
from spacy import displacy
import warnings

warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')


def find_eos(raw_text):
    return re.finditer(r';\s+([A-Z]|-)|\.\s{1,}-', raw_text)


def replace_eos(raw_text):
    eos_iter = find_eos(raw_text)
    raw_text = list(raw_text)
    for eos in eos_iter:
        eos_symbol = re.search(r';\s+-|;|\.\s{1,}-', ''.join(raw_text))

        try:
            start, end = eos_symbol.span()
            raw_text[start:end] = '.'
        except AttributeError:
            print(''.join(raw_text))
            breakpoint()

    return ''.join(raw_text)


nlp = spacy.load('pt_core_news_lg')


def spacy_split_sentences(raw_text):
    return [sent.text.strip() for sent in nlp(raw_text).sents if sent.text.strip()]


def split_in_sentences(raw_text):
    raw_text = replace_eos(raw_text)
    return spacy_split_sentences(raw_text)


def replace_abbrv(sentence):
    abbrv_map = {r'Cons\.': 'conselheiro',
                 r'Rel\.': 'relator',
                 r'Min\.': 'ministro',
                 r'art\.': 'artigo',
                 r'Art\.': 'artigo',
                 r'arts\.': 'artigos',
                 r'docs\.': 'documentos',
                 r'doc\.': 'documento',
                 r'TC\.': 'TC',
                 r'inc\.': 'inciso',
                 r'\.–\s+': ' - ',
                 ', Tc': ', tc'}
    for abbrv in abbrv_map.keys():
        sentence = re.sub(abbrv, abbrv_map[abbrv], sentence, flags=re.IGNORECASE)
    return sentence


def capitalize_sent(sent):
    sent = sent.split()
    sent[0] = sent[0].capitalize()
    return ' '.join(sent)


def get_text_from_html(html_text, keep_tags=None):
    if keep_tags is None:
        keep_tags = ["p", "strong"]
    """

    :param keep_tags: tags containing the text
    :param html_text: html string
    :return:
    """
    clean = bleach.clean(html_text, tags=keep_tags, strip=True)
    soup = BeautifulSoup(clean.replace("&amp;", "&"), features="html.parser")

    for strg_tag in soup.find_all('strong'):
        try:
            strg_tag.string = strg_tag.string.capitalize()
        except AttributeError:
            Warning('No text in tag <strong>: ' + strg_tag.prettify())
            pass

    sentences = [capitalize_sent(unicode(prg.text)) for prg in soup.find_all('p')
                 if prg.get_text(strip=True)]

    return " ".join(sentences)


def preprocess_sentence(sentence):
    sentence = sentence.strip()

    new_sentence = re.sub(
        r'(Considernado|Considerando,\sno\sentanto,\sque|Considerando\sque,|Considerando\sque|Considerando,'
        r'|Considerando)',
        '', sentence).strip().split()

    try:
        new_sentence[0] = new_sentence[0].capitalize()
        if re.search(',', new_sentence[0].capitalize()):
            new_sentence.pop(0)
        new_sentence[0] = new_sentence[0].capitalize()
        new_sentence[-1] = new_sentence[-1].replace(';', '.')
    except:
        return None
    return ' '.join(new_sentence)


def preprocess_pipeline(text):

    text = replace_abbrv(text)
    text = replace_eos(text)
    sentences = filter(lambda sent: sent, map(preprocess_sentence, split_in_sentences(text)))

    return list(sentences)
