# %%
from pathlib import Path
import re

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
import pyopenjtalk
from tqdm.autonotebook import tqdm


text = Path('src_token/KWDLC.tsv').read_text(encoding = "utf-8")
text = text.split("\n")

label = []
data = []

for line in text:
    current_cnt = line.split("\t")[1]
    data.append(current_cnt.replace(" ", ""))
    label.append(current_cnt)

i = 0

ref = []
hyp = []
for i, d in enumerate(tqdm(data)):
    njd_features = pyopenjtalk.run_frontend(d)
    cur_text = []
    for content in njd_features:
        cur_text.append(content["string"])
    cur_text = " ".join(cur_text)
    
    hyp.append(list(cur_text))
    ref.append([list(label[i])])


smooth = SmoothingFunction()

corpus_score_1 = corpus_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
corpus_score_2 = corpus_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
corpus_score_3 = corpus_bleu(ref, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth.method1)
corpus_score_4 = corpus_bleu(ref, hyp, smoothing_function=smooth.method1)

print("### KWDLC tokenize score:")
print(corpus_score_1)
print(corpus_score_2)
print(corpus_score_3)
print(corpus_score_4)