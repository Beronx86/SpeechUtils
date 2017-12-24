import tensorflow as tf
import os
import argparse
import scipy.io.wavfile as siowav
import numpy as np
import tqdm


def get_arguments():
    parser = argparse.ArgumentParser(description="Extract wav from TFRecords file and save.")
    parser.add_argument("--tfrecord_path", "-s", type=str, default="./wav.tfrecords", help="")
    parser.add_argument("--wav_root", "-d", type=str, default="./wav_recover", help="")
    return parser.parse_args()


def parse_single_example(string_record):
    example = tf.train.Example()
    example.ParseFromString(string_record)

    sr = int(example.features.feature["sr"].int64_list.value[0])
    key = example.features.feature["key"].bytes_list.value[0].decode("utf-8")
    wav_raw = example.features.feature["wav_raw"].bytes_list.value[0]
    
    wav = np.fromstring(wav_raw, dtype=np.int16)

    return sr, key, wav


def main():
    args = get_arguments()
    
    record_iterator = tf.python_io.tf_record_iterator(path=args.tfrecord_path)

    for idx, string_record in tqdm.tqdm(enumerate(record_iterator)):
        sr, key, wav = parse_single_example(string_record)

        save_path = os.path.join(args.wav_root, key)
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        siowav.write(save_path, rate=sr, data=wav)

    print("Congratulations!")


if __name__ == "__main__":
    print(__file__)
    main()
