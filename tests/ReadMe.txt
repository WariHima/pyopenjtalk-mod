voicevoxのpyopenjtalk-plus採用の議論を参考にして作られました。
https://github.com/VOICEVOX/voicevox_engine/issues/1486

カナ(発音)ベンチマーク

< rohan4600 >
pip install requirements.txt
python benchmark_rohan.py

pyopenjtalk-plusでのスコア
0.9537292046741687
0.9323544699007222
0.9127693257448863
0.8924055012180035

### トークナイズ(分割)ベンチマーク
pip install requirements.txt
python benchmark_token.py

tsvパスを差し替えることができます
次のファイルを使えます(KWDLCがハードコードされています)
WIKIPEDIA.tsv
FKC.tsv
KWDLC.tsv

pyopenjtalk-plusでのスコア
< WIKIPEDIA.tsv >
0.9215864940455851
0.8897225413460286
0.8651966763313014
0.8404199442183062

< FKC.tsv >
0.9635951027880545
0.9329357826807906
0.9060756510971438
0.8781504930636572

< KWDLC.tsv >
0.9624192730114766
0.935121343475862
0.9118425486523336
0.8879945857880588

アクセント指定ベンチマーク
< jsut.tsv >
0.6884325193892078
0.6835059613998141
0.6806199271379075
0.6704855044087938


各ファイルの入手先

rohan4600コーパスのトランスクリプトの入手先
https://github.com/mmorise/rohan4600


KWDLC.tsv 
KWDLCから作成
https://github.com/ku-nlp/KWDLC

FKC.tsv
Annotated FKC Corpusから作成
https://github.com/ku-nlp/AnnotatedFKCCorpus

WIKIPEDIA.tsv
wikipedia Annotated Corpusから作成
https://github.com/ku-nlp/WikipediaAnnotatedCorpus

jsut.tsv
jsutコーパスのラベルから作りました
https://github.com/sarulab-speech/jsut-label/

#規模が小さいため未使用
itaコーパスのトランスクリプトの入手先
https://github.com/mmorise/ita-corpus

jvsコーパスのカタカナでの読み(発音)の入手先
https://github.com/Hiroshiba/jvs_hiho

同作者の機械での読み推定スクリプトから漢字表記を入手
https://github.com/Hiroshiba/voiceactress100_ruby
