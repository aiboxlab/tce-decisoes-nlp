import spacy, re
import pandas as pd
from transformers import pipeline
import numpy as np

nlp = spacy.load('pt_core_news_sm')

POST_TAGS = ['VERB', 'NOUN', 'ADJ']

spacy2lex = {'ADJ': 'adj',
             'NOUN': 'nom',
             'VERB': 'ver',
             'SCONJ': 'sconj',
             'ADP': 'adp',
             'DET': 'det',
             'PUNCT': 'punct',
             'CCONJ': 'cconj',
             'PROPN': 'prop',
             'ADV': 'adv',
             'NUM': 'num'}

fill_mask = pipeline(
    "fill-mask",
    model='neuralmind/bert-base-portuguese-cased',
    tokenizer='neuralmind/bert-base-portuguese-cased'
)

lexporbr = pd.read_csv('lexporbr_alfa_txt.txt', sep='\t', error_bad_lines='ignore', encoding="ISO-8859-1")
lexporbr['zipf_scale'] = lexporbr['zipf_scale'].apply(lambda value: value.replace(',', '.'))
lexporbr['log10_freq_orto'] = lexporbr['log10_freq_orto'].apply(lambda value: value.replace(',', '.'))


def find2replace(sentence):
    doc = nlp(sentence)
    tokens2replace = []

    tokens = [(token.lemma_.lower(), spacy2lex[token.pos_], token) for token in doc if token.pos_ in POST_TAGS]

    for token_lemma, cat_gram, token in tokens:

        zipf_scale = lexporbr.loc[(lexporbr["ortografia"] == token_lemma)
                                  & (lexporbr["cat_gram"] == cat_gram),
                                  'zipf_scale'].astype(float).to_numpy()

        if len(zipf_scale) == 0 or zipf_scale < 4:
            tokens2replace.append(token)
    return tokens2replace


def simplify_sentence(sentence, token2mask):
    results = []

    for token in token2mask:
        try:
            masked_sentence = re.sub(token.text, '[MASK]', sentence)

            result = fill_mask(sentence + '[SEP]' + masked_sentence)
            df = pd.DataFrame(result).loc[:, ['score', 'token_str']]

            possible_tokens = [tok for tok in df['token_str'].to_numpy()
                               if tok.find('#')]

            prob = df['score'].to_numpy() / np.sum(df['score'].to_numpy())

            pred_token = np.random.choice(possible_tokens, p=prob)

            results.append({'sent': sentence,
                            'original_word': token.text,
                            'pred_word': pred_token
                            })
            sentence = masked_sentence.replace('[MASK]', pred_token)
        except Exception:
            pass
    return sentence, results
