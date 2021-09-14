import pandas as pd
import numpy as np
import mediator

df = pd.read_csv('data.csv')
all_words = []


for ind in df.index:
    for word in mediator.keywords:
        all+=df[word][ind]

print(all_words)
