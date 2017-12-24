import tensorflow as tf
import scipy.io.wavfile as siowav
import os
import argparse
import tqdm


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_arguments():
    parser = argparse.ArgumentParser(description="Convert wav file to TFRecords file.")
    parser.add_argument("--wav_root", "-s", type=str, default="./wav", help="")
    parser.add_argument("--target_path", "-d", type=str, default="./wav.tfrecords", help="")
    return parser.parse_args()


def read_to_bytes(path):
    sr, wav = siowav.read(path)
    wav_raw = wav.tostring()
    del wav
    key = path.encode("utf-8")
    # create tf example feature
    example = tf.train.Example(features=tf.train.Features(feature={
        "sr": _int64_feature(sr),
        "key": _bytes_feature(key),
        "wav_raw": _bytes_feature(wav_raw)}))
    return example.SerializeToString()


def get_path_lst(root, cur_list=[]):
    for item in os.listdir(root):
        item_path = os.path.join(root, item)
        if os.path.isdir(item_path):
            get_path_lst(item_path, cur_list)
        if os.path.isfile(item_path):
            if item_path.split(".")[-1] == "wav":
                cur_list.append(item_path)
    return cur_list


def main():
    print(__file__)
    args = get_arguments()
    path_lst = get_path_lst(args.wav_root)
    assert path_lst, "[E] Path list is empty!"
    
    with tf.python_io.TFRecordWriter(args.target_path) as writer:
        for path in tqdm.tqdm(path_lst):
            example_str = read_to_bytes(path)
            writer.write(example_str)

    print("Congratulations!")

if __name__ == "__main__":
    main()
