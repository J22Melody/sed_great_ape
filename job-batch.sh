# sbatch --output=models/chimp_waveform_lstm_0/train.log ./job.sh models/chimp_waveform_lstm_0/config.json
# sbatch --output=models/chimp_waveform_lstm_42/train.log ./job.sh models/chimp_waveform_lstm_42/config.json
# sbatch --output=models/chimp_waveform_lstm_3407/train.log ./job.sh models/chimp_waveform_lstm_3407/config.json

# sbatch --output=models/chimp_spectrogram_lstm_0/train.log ./job.sh models/chimp_spectrogram_lstm_0/config.json
# sbatch --output=models/chimp_spectrogram_lstm_42/train.log ./job.sh models/chimp_spectrogram_lstm_42/config.json
# sbatch --output=models/chimp_spectrogram_lstm_3407/train.log ./job.sh models/chimp_spectrogram_lstm_3407/config.json

# sbatch --output=models/chimp_wav2vec2_lstm_0/train.log ./job.sh models/chimp_wav2vec2_lstm_0/config.json
# sbatch --output=models/chimp_wav2vec2_lstm_42/train.log ./job.sh models/chimp_wav2vec2_lstm_42/config.json
# sbatch --output=models/chimp_wav2vec2_lstm_3407/train.log ./job.sh models/chimp_wav2vec2_lstm_3407/config.json

sbatch --output=models/chimp_wav2vec2_transformer_0/train.log ./job.sh models/chimp_wav2vec2_transformer_0/config.json
sbatch --output=models/chimp_wav2vec2_transformer_42/train.log ./job.sh models/chimp_wav2vec2_transformer_42/config.json
sbatch --output=models/chimp_wav2vec2_transformer_3407/train.log ./job.sh models/chimp_wav2vec2_transformer_3407/config.json