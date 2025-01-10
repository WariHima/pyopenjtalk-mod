
file = "./pyopenjtalk/suwad_dictionary/model.bin"

from pathlib import Path

def main(file: str):
    data = Path(file).read_bytes()
    datasize = len(data)

    data1 = data[:datasize//2]
    data2 = data[datasize//2:]
    Path(file+"1", ).write_bytes(data1)
    Path(file+"2", ).write_bytes(data2)


if __name__ == "__main__":
    main(file)