in= #输入音频名
out= #输出音频名
thresh_hold=0.2 #最低音量[0~1]，干净语音测试可设为0.2
max_silence_dur=0.25 #最长中间静音段长度


sox ${in}.wav ${out}.tmp0.wav silence -l 1 0.1 ${thresh_hold}% -1 ${max_silence_dur} ${thresh_hold}% \
&& sox ${out}.tmp0.wav ${out}.tmp1.wav reverse silence 1 0.1 ${thresh_hold}% \
&& sox ${out}.tmp1.wav ${out}.wav reverse \
&& rm ${out}.tmp0.wav ${out}.tmp1.wav