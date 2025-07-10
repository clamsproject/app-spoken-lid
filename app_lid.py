import argparse
from pathlib import Path
import csv  
import librosa
import torch
import whisper

from lid_util import detect_language_by_chunk


def load_audio_16k(path):
    # load any audio file as mono 16 kHz
    # need to lookup librosa documentation again
    wave, sr = librosa.load(path, sr=16000, mono=True) # --> what is mono?
    return wave, sr


def chunk_audio(wave, sr, chunk_sec=30):

    window = int(chunk_sec * sr)
    return [wave[i:i + window] for i in range(0, len(wave), window)]


def audio_to_mel(chunk, model):
    # adapt from whisper's audio.py
    # waveform --> Whisper supported Mel tensor [1, 80, 3000] or [1, 128, 3000]

    # pat or trim to 30s

    wav = torch.tensor(chunk, dtype=torch.float32).to(model.device)
    wav = whisper.pad_or_trim(wav)          # 30-second window
    mel = whisper.log_mel_spectrogram(wav, n_mels=model.dims.n_mels) # 80 or 128
    return mel.unsqueeze(0)                 # add batch dim to [1, 80, 3000] or [1, 128, 3000] as model.detect_language expects


# cli
def main():
    parser = argparse.ArgumentParser(
        description="Chunk-level language ID with Whisper")
    parser.add_argument("audio", help="audio file (wav/mp3, etc.)")
    parser.add_argument("--model", default="tiny",
                        choices=["tiny", "base", "small", "medium", "large", "turbo"],
                        help="Whisper model size (default: tiny)")
    parser.add_argument("--chunk", type=float, default=30,
                        help="chunk length in seconds (default: 30)")
    # parser.add_argument("--window," type=float, default=30, 
                        # help="sliding window in seconds(default 30)")
    parser.add_argument("--batch", type=int, default=1,
                        help="number of windows in one forward pass")
    parser.add_argument("--top", type=int, default=3, help="top languague scores")
    parser.add_argument("--out", default="preds.csv")
    args = parser.parse_args()

    model = whisper.load_model(args.model)
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual,
                                            language=None)


    wave, sr = load_audio_16k(args.audio)
    chunks = chunk_audio(wave, sr, args.chunk)
    rows = []
    for idx, chunk in enumerate(chunks):
        # print(type(idx), type(args.chunk))
        if len(chunk) == 0:
            continue

        mel = audio_to_mel(chunk, model)
        # probs, lang = detect_language_by_chunk(model, mel)
        probs, lang = detect_language_by_chunk(model, mel, tokenizer)


        idx_py   = int(idx)                     # tensor --> int
        chunk_dur = float(args.chunk)           # may be numpy/torch --> float
        start_s  = idx_py * chunk_dur
        end_s    = (idx_py + 1) * chunk_dur
        

        topN = list(probs.items())[:args.top]


        # print(
        #     f"Chunk {idx_py}  "
        #     f"[{start_s}-{end_s}s]: "
        #     f"{lang}  |  top-3 {top3}"
        # )
        rows.append({
            "file": Path(args.audio).name,
            "start": start_s,
            "end": end_s,
            "pred": topN[0][0],
            "topN": topN
            })

    with open(args.out, "w", newline="") as f:
        fieldnames = ["file", "start", "end", "pred", "topN"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()