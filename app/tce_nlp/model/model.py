import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

filename = '../resources/new_corpus.txt'

corpus = [line.strip() for line in open(filename, encoding='utf-8').readlines() if line.strip()]
print(corpus[:5])
print("Total of documents: ", len(corpus))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
features = vectorizer.get_feature_names()
print(len(features))

is_digit = [len(re.findall(r'\d', feat)) == 0 for feat in features]
print(len(is_digit))
features = np.asarray(features)[is_digit]
idf_document = X.toarray()[:, is_digit].sum(axis=1).flatten()
print(corpus[idf_document.argmax()])
idf_features = X.toarray()[:, is_digit].sum(axis=0).flatten()
print(features[idf_features.argsort()])
df = pd.DataFrame({
    "features": features[idf_features.argsort()],
    "idf": idf_features[idf_features.argsort()]
})

df.to_csv('features.csv', index_label=False, index=False)
