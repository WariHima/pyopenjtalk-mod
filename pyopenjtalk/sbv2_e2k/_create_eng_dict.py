from pathlib import Path
from pyopenjtalk.sbv2_e2k.katakana_map import KATAKANA_MAP
import re

"""
nhk日本語アクセント辞書　カタカナのアクセント推定より抜粋
"""
ONLY_KATAKANA_PATTERN = re.compile("^[ァ-ワヰヱヲンヴー]+$")
KATAKANA_MAP_ITEM = KATAKANA_MAP.items()
youon_list = ["ァ", "ィ", "ゥ", "ェ" ,"ォ","ャ","ュ","ョ"]

out = []
for i in KATAKANA_MAP_ITEM:
    surface = i[0]
    pron = i[1]

    mora = pron 

    for youon in  youon_list:
        mora = mora.replace(youon, "")
    
    mora = len(mora)

    #4モーラ以上の時後ろから数えて三番目までを高くする
    if mora >= 4:
        accent = mora - 2

    else:
        accent = 1
    
    if ONLY_KATAKANA_PATTERN.fullmatch(pron):
        line = f"{pron},2,2,-3000,フィラー,*,*,*,*,*,{pron},{pron},{pron},{accent}/{mora},*"
        out.append(line)

out = "\n".join(out)
Path( "./pyopenjtalk/user_dictionary/english.csv" ).write_text(out, encoding = "utf-8")