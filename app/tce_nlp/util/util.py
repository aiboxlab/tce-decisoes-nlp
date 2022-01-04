from bs4 import BeautifulSoup
from cffi.backend_ctypes import unicode
import bleach

import re

money_pattern = re.compile(r'R[$] (\d+[.|,]){1,10}[0-9]{1,3}')
perc_pattern = re.compile(r'([0-9]{1,3})(,[0-9]{1,3}|)?(%)')
date_pattern = re.compile(r'[0-9][0-9]*[/][0-9][0-9]*[/][0-9][0-9]*')

law = r'(Lei|Decreto|Resoluções)([ \w+]*( nº|))(\s)(\d+|\d+\.\d+)([/]\d+)'
artigo = r"(§ 1º do |)(artigo \w+ \w+ )" + law

acordao = r"(TC|TCE) nº \d*\-\d"

l2 = r"(Lei|Decreto|Resolução) .+?(?={})\s".format(';|,')
artigo2 = r"(artigo|art.) (\w+\,) (inciso|§) (\w+\, \w+ )" + l2

PATTERNS = [(money_pattern, 'Quantia Financeira'),
            (perc_pattern, 'Percentagem'),
            (date_pattern, 'Data'),
            (law, 'Legislação'),
            (artigo, 'Artigo'),
            (acordao, 'Acórdão')]



