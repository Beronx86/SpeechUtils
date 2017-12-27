import subprocess

subprocess.call(["python", "../create_wav_mel_stftm_tfrecords.py", "-s", "./audios", "--sr", "48000"])