import numpy as np
import re

"""[{"text": "Mas Google está iniciando de trás.",
  "ents": [{"start": 4, "end": 10, "label": "ORG"}],
  "title": None}]"""


def get_regex_dict(text, patterns):
    """
    Transforma as entidades de um texto em labels.
    Exemplo: Ele adquiriu R$ 10.000,00 -> Ele adquiriu <Quantia Financeira_0> 0 é o id da label.
    :param patterns:
    :param text: any string
    :return: a dictionary mapping key (entities labels) to values (entities values)
    """

    entities = {
        "text": text,
        "ents": [],
        "title": None
    }
    v_starts = np.array([-1])
    for pattern, label in patterns:
        search_iter = re.finditer(pattern, text)
        for i, search in enumerate(search_iter):
            start, end = search.start(), search.end()

            if np.any(v_starts <= start) and start < end:
                np.insert(v_starts, 0, start)

                entities["ents"].append({
                    "start": start,
                    "end": end,
                    "label": ""
                })

    return [entities]
