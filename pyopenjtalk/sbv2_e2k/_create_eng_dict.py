from pathlib import Path
from pyopenjtalk.sbv2_e2k.katakana_map import KATAKANA_MAP

import mojimoji

"""
nhk日本語アクセント辞書　カタカナのアクセント推定より抜粋
"""

KATAKANA_MAP_ITEM = KATAKANA_MAP.items()
youon_list = ["ァ", "ィ", "ゥ", "ェ" ,"ォ","ャ","ュ","ョ", "ッ"]

out = []
for i in KATAKANA_MAP_ITEM:
    surface = i[0]
    surface = mojimoji.han_to_zen(surface)
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
    
    line = f"{surface},2,2,9000,フィラー,*,*,*,*,*,{surface},{pron},{pron},{accent}/{mora},*"
    out.append(line)

out = "\n".join(out)
Path( "english.csv" ).write_text(out, encoding = "utf-8")