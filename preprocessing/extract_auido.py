# extract_audio.py
import os
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf

VIDEO_ROOT = "/home/jaesang02/MUTON_cpy/data/video"
AUDIO_ROOT = "/home/jaesang02/MUTON_cpy/data/audio"

TARGET_SR = 16000  # Whisper / WavLM 공통

def extract_audio(mp4_path, wav_path):
    try:
        clip = VideoFileClip(mp4_path)
        audio = clip.audio

        if audio is None:
            print(f"⚠️ No audio track: {mp4_path}")
            return

        tmp_wav = wav_path.replace(".wav", "_tmp.wav")
        audio.write_audiofile(tmp_wav, fps=44100, verbose=False, logger=None)

        # resample + mono
        y, sr = librosa.load(tmp_wav, sr=TARGET_SR, mono=True)
        sf.write(wav_path, y, TARGET_SR)

        os.remove(tmp_wav)
        print(f"✅ Extracted: {wav_path}")

    except Exception as e:
        print(f"❌ Failed {mp4_path}: {e}")

def main():
    for root, dirs, files in os.walk(VIDEO_ROOT):
        for file in files:
            if not file.endswith(".mp4"):
                continue

            mp4_path = os.path.join(root, file)

            # 상대 경로 유지
            rel_dir = os.path.relpath(root, VIDEO_ROOT)
            out_dir = os.path.join(AUDIO_ROOT, rel_dir)
            os.makedirs(out_dir, exist_ok=True)

            wav_name = file.replace(".mp4", ".wav")
            wav_path = os.path.join(out_dir, wav_name)

            extract_audio(mp4_path, wav_path)

if __name__ == "__main__":
    main()
