# pyopenjtalk-mod

[![PyPI](https://img.shields.io/pypi/v/pyopenjtalk-mod.svg)](https://pypi.python.org/pypi/pyopenjtalk-mod)
[![Python package](https://github.com/tsukumijima/pyopenjtalk-plus/actions/workflows/ci.yml/badge.svg)](https://github.com/tsukumijima/pyopenjtalk-plus/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

pyopenjtalk-modは[pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus)をベースに改造することを目的とした、pyopenjtalkの派生ライブラリです
pyopenjtalkの代替品として性能向上が期待できますが、開発頻度が高いため、pyopenjtalk-plusとの互換性は考えていません

ちなみにpyopenjtalk-plus は、各フォークでの改善を一つのコードベースにまとめ、さらなる改善を加えることを目的とした、[pyopenjtalk](https://github.com/r9y9/pyopenjtalk) の派生ライブラリらしいです。

user_dictionary以下の辞書のソースは[openjtalk-user-dict](https://github.com/WariHima/openjtalk-user-dict)にあります

## version log
[wiki](https://github.com/WariHima/pyopenjtalk-mod/wiki)にあります

## Installation

下記コマンドを実行して、ライブラリをインストールできます。

```bash
pip install pyopenjtalk-mod
```



## Supported platforms

- Linux
- Mac OSX
- Windows (MSVC) (see [this PR](https://github.com/r9y9/pyopenjtalk/pull/13))

## Build requirements

The python package relies on cython to make python bindings for open_jtalk and hts_engine_API. You must need the following tools to build and install pyopenjtalk:

- C/C++ compilers (to build C/C++ extentions)
- cmake
- cython

## Development

開発環境は macOS / Linux 、Python バージョンは 3.11 が前提です。

```bash
# submodule ごとリポジトリを clone
git clone --recursive https://github.com/tsukumijima/pyopenjtalk-mod.git

#or
git clone --recursive https://github.com/tsukumijima/pyopenjtalk-mod.git
git submodule update --recursive --init

cd pyopenjtalk-plus

# ライブラリ自身とその依存関係を .venv/ 以下の仮想環境にインストールし、開発環境を構築
pip install taskipy
task install

# コード整形
task lint
task format

# テストの実行
task test

# pyopenjtalk/dictionary/ 以下にある MeCab / OpenJTalk 辞書をビルド
## ビルド成果物は同ディレクトリに *.bin / *.dic として出力される
## ビルド後の辞書データは数百 MB あるバイナリファイルだが、取り回しやすいよう敢えて Git 管理下に含めている
task build-dictionary

# ライブラリの wheel と sdist をビルドし、dist/ に出力
task build

# ビルド成果物をクリーンアップ
task clean
```

[docs/](docs/) 以下のドキュメントは、[pyopenjtalk](https://github.com/r9y9/pyopenjtalk) 本家のドキュメントを改変なしでそのまま引き継いでいます。  


A python wrapper for [OpenJTalk](http://open-jtalk.sp.nitech.ac.jp/).

The package consists of two core components:

- Text processing frontend based on OpenJTalk
- Speech synthesis backend using HTSEngine

## Notice

- The package is built with the [modified version of OpenJTalk](https://github.com/r9y9/open_jtalk). The modified version provides the same functionality with some improvements (e.g., cmake support) but is technically different from the one from HTS working group.
- The package also uses the [modified version of hts_engine_API](https://github.com/r9y9/hts_engine_API). The same applies as above.

Before using the pyopenjtalk package, please have a look at the LICENSE for the two software.


## Quick demo

Please check the notebook version [here (nbviewer)](https://nbviewer.jupyter.org/github/r9y9/pyopenjtalk/blob/master/docs/notebooks/Demo.ipynb).

### TTS

```py
In [1]: import pyopenjtalk

In [2]: from scipy.io import wavfile

In [3]: x, sr = pyopenjtalk.tts("おめでとうございます")

In [4]: wavfile.write("test.wav", sr, x.astype(np.int16))
```

### Run text processing frontend only

```py
In [1]: import pyopenjtalk

In [2]: pyopenjtalk.extract_fullcontext("こんにちは")
Out[2]:
['xx^xx-sil+k=o/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:xx_xx#xx_xx@xx_xx|xx_xx/G:5_5%0_xx_xx/H:xx_xx/I:xx-xx@xx+xx&xx-xx|xx+xx/J:1_5/K:1+1-5',
'xx^sil-k+o=N/A:-4+1+5/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'sil^k-o+N=n/A:-4+1+5/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'k^o-N+n=i/A:-3+2+4/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'o^N-n+i=ch/A:-2+3+3/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'N^n-i+ch=i/A:-2+3+3/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'n^i-ch+i=w/A:-1+4+2/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'i^ch-i+w=a/A:-1+4+2/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'ch^i-w+a=sil/A:0+5+1/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'i^w-a+sil=xx/A:0+5+1/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'w^a-sil+xx=xx/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:xx+xx_xx/E:5_5!0_xx-xx/F:xx_xx#xx_xx@xx_xx|xx_xx/G:xx_xx%xx_xx_xx/H:1_5/I:xx-xx@xx+xx&xx-xx|xx+xx/J:xx_xx/K:1+1-5']
```

Please check `lab_format.pdf` in [HTS-demo_NIT-ATR503-M001.tar.bz2](http://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-ATR503-M001.tar.bz2) for more details about full-context labels.


### Grapheme-to-phoeneme (G2P)

```py
In [1]: import pyopenjtalk

In [2]: pyopenjtalk.g2p("こんにちは")
Out[2]: 'k o N n i ch i w a'

In [3]: pyopenjtalk.g2p("こんにちは", kana=True)
Out[3]: 'コンニチワ'
```

### Create/Apply user dictionary

1. Create a CSV file (e.g. `user.csv`) and write custom words like below:

```csv
ＧＮＵ,,,1,名詞,一般,*,*,*,*,ＧＮＵ,グヌー,グヌー,2/3,*
```

2. Call `mecab_dict_index` to compile the CSV file.

```python
In [1]: import pyopenjtalk

In [2]: pyopenjtalk.mecab_dict_index("user.csv", "user.dic")
reading user.csv ... 1
emitting double-array: 100% |###########################################|

done!
```

3. Call `update_global_jtalk_with_user_dict` to apply the user dictionary.

```python
In [3]: pyopenjtalk.g2p("GNU")
Out[3]: 'j i i e n u y u u'

In [4]: pyopenjtalk.update_global_jtalk_with_user_dict("user.dic")

In [5]: pyopenjtalk.g2p("GNU")
Out[5]: 'g u n u u'
```

### About `run_marine` option

After v0.3.0, the `run_marine` option has been available for estimating the Japanese accent with the DNN-based method (see [marine](https://github.com/6gsn/marine)). If you want to use the feature, please install pyopenjtalk as below;

```shell
pip install pyopenjtalk[marine]
```

And then, you can use the option as the following examples;

```python
In [1]: import pyopenjtalk

In [2]: x, sr = pyopenjtalk.tts("おめでとうございます", run_marine=True) # for TTS

In [3]: label = pyopenjtalk.extract_fullcontext("こんにちは", run_marine=True) # for text processing frontend only
```


## LICENSE

- pyopenjtalk: MIT license ([LICENSE.md](LICENSE.md))
- Open JTalk: Modified BSD license ([COPYING](https://github.com/r9y9/open_jtalk/blob/1.10/src/COPYING))
- htsvoice in this repository: Please check [pyopenjtalk/htsvoice/README.md](pyopenjtalk/htsvoice/README.md).
- marine: Apache 2.0 license ([LICENSE](https://github.com/6gsn/marine/blob/main/LICENSE))

## Acknowledgements

HTS Working Group for their dedicated efforts to develop and maintain Open JTalk.
