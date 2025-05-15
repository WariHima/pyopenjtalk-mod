# %%
from pathlib import Path
import re
import pyopenjtalk

__PYOPENJTALK_G2P_PROSODY_A1_PATTERN = re.compile(r"/A:([0-9\-]+)\+")
__PYOPENJTALK_G2P_PROSODY_A2_PATTERN = re.compile(r"\+(\d+)\+")
__PYOPENJTALK_G2P_PROSODY_A3_PATTERN = re.compile(r"\+(\d+)/")
__PYOPENJTALK_G2P_PROSODY_E3_PATTERN = re.compile(r"!(\d+)_")
__PYOPENJTALK_G2P_PROSODY_F1_PATTERN = re.compile(r"/F:(\d+)_")
__PYOPENJTALK_G2P_PROSODY_P3_PATTERN = re.compile(r"\-(.*?)\+")

def __pyopenjtalk_g2p_prosody(
    text: str, drop_unvoiced_vowels: bool = True
) -> list[str]:
    """
    ESPnet の実装から引用、概ね変更点無し。「ん」は「N」なことに注意。
    ref: https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
    ------------------------------------------------------------------------------------------

    Extract phoneme + prosody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104
    """

    def _numeric_feature_by_regex(pattern: re.Pattern[str], s: str) -> int:
        match = pattern.search(s)
        if match is None:
            return -50
        return int(match.group(1))

    labels =  pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        cur_phone = __PYOPENJTALK_G2P_PROSODY_P3_PATTERN.search(lab_curr)
        if cur_phone != None:
            p3 = cur_phone.group(1)
            
            
            # type: ignore
            # deal unvoiced vowels as normal vowels
            if drop_unvoiced_vowels and p3 in "AEIOU":
                p3 = p3.lower()
            

            # deal with sil at the beginning and the end of text
            if p3 == "sil":
                assert n == 0 or n == N - 1
                if n == 0:
                    phones.append("^")
                elif n == N - 1:
                    # check question form or not
                    e3 = _numeric_feature_by_regex(
                        __PYOPENJTALK_G2P_PROSODY_E3_PATTERN, lab_curr
                    )
                    if e3 == 0:
                        phones.append("$")
                    elif e3 == 1:
                        phones.append("?")
                continue
            elif p3 == "pau":
                phones.append("_")
                continue
            else:
                phones.append(p3)

            # accent type and position info (forward or backward)
            a1 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A1_PATTERN, lab_curr)
            a2 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A2_PATTERN, lab_curr)
            a3 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A3_PATTERN, lab_curr)

            # number of mora in accent phrase
            f1 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_F1_PATTERN, lab_curr)

            a2_next = _numeric_feature_by_regex(
                __PYOPENJTALK_G2P_PROSODY_A2_PATTERN, labels[n + 1]
            )
            # accent phrase border
            if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
                phones.append("#")
            # pitch falling
            elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
                phones.append("]")
            # pitch rising
            elif a2 == 1 and a2_next == 2:
                phones.append("[")

    return phones

text = Path('tests/src_acc/jsut.tsv').read_text(encoding = "utf-8")
text = text.split("\n")

label = []
data = []


for line in text:
    accent = line.split("\t")[1]
    new_acc = ""
    for i in range(len(accent)):
        if accent[i] in ("[", "]", "#", " ", "$", "_", "?", "^", "$"):
            new_acc += accent[i]
        else:
            new_acc += "X"

    cur_text = line.split("\t")[2]

    data.append(cur_text)
    label.append(new_acc)

i = 0



# %%
from tqdm.autonotebook import tqdm

def is_dic_file(file: Path) -> bool:
    supported_extensions = [".dic"]
    return file.suffix.lower() in supported_extensions

input_dir = Path("./dic")
dict_files = [file for file in input_dir.rglob("*") if is_dic_file(file)]

if dict_files != []:
    for file in dict_files:
        pyopenjtalk.update_global_jtalk_with_user_dict(str(file))

ref = []
hyp = []
for i, d in enumerate(tqdm(data)):
    cur_text = __pyopenjtalk_g2p_prosody(d)
    cur_text = " ".join(cur_text)
    new_text = ""
    for i in range(len(cur_text)):
        if cur_text[i] in ("[", "]", "#", " ", "$", "_", "?", "^", "$"):
            new_text += cur_text[i]
        else:
            new_text += "X"
            
    hyp.append(list(new_text))
    ref.append([list(label[i])])

# %%
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

smooth = SmoothingFunction()

corpus_score_1 = corpus_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
corpus_score_2 = corpus_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
corpus_score_3 = corpus_bleu(ref, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth.method1)
corpus_score_4 = corpus_bleu(ref, hyp, smoothing_function=smooth.method1)

# %%
print("### score:")
print(corpus_score_1)
print(corpus_score_2)
print(corpus_score_3)
print(corpus_score_4)