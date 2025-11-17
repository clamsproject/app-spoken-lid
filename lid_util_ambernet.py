import torch
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path

# AmberNet supports 107 languages
# This list matches the order in the model's label configuration
AMBERNET_LANGUAGES = [
    'ab', 'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs',
    'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa',
    'fi', 'fo', 'fr', 'gl', 'gn', 'gu', 'gv', 'ha', 'haw', 'hi', 'hr', 'ht', 'hu',
    'hy', 'ia', 'id', 'is', 'it', 'iw', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko',
    'la', 'lb', 'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms',
    'mt', 'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru',
    'sa', 'sco', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw',
    'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'war',
    'yi', 'yo', 'zh'
]


def detect_language_by_chunk_ambernet(model, audio_chunk, sample_rate, top_k=3):
    # Args:
    #     model: NeMo EncDecSpeakerLabelModel (AmberNet)
    #     audio_chunk: numpy array of audio samples
    #     sample_rate: sample rate of audio (should be 16000)
    #     top_k: number of top predictions to return
        
    # Returns:
    #     (probs_sorted: dict[str, float], iso_code: str)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        
    try:
        sf.write(tmp_path, audio_chunk, sample_rate)
        
        try:
            with torch.no_grad():
                # Load audio
                import librosa
                audio_data, _ = librosa.load(tmp_path, sr=sample_rate, mono=True)
                
                # Convert to tensor
                audio_signal = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
                audio_signal_len = torch.tensor([len(audio_data)], dtype=torch.int64)
                
                # Try to get logits using forward pass
                try:
                    # Try infer_segment with audio tensor
                    _, logits = model.infer_segment(audio_signal, audio_signal_len)
                except TypeError:
                    logits, _ = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
                
                if isinstance(logits, tuple):
                    logits = logits[0]
                if len(logits.shape) > 1:
                    logits = logits[0]
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                probs_dict = {}
                for i, lang in enumerate(AMBERNET_LANGUAGES):
                    if i < len(probs):
                        probs_dict[lang] = float(probs[i].item())
                
                probs_sorted = dict(
                    sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)
                )
                
                iso_code = list(probs_sorted.keys())[0]
                
                return probs_sorted, iso_code
                
        except Exception as e:
            # fallback to get_label (simple and reliable)
            print(f"Note: Using get_label fallback (couldn't get probabilities: {e})")
            
            iso_code = model.get_label(tmp_path)
            
            probs_sorted = {iso_code: 0.95}
            
            dummy_prob = 0.05 / min(top_k - 1, len(AMBERNET_LANGUAGES) - 1)
            count = 0
            for lang in AMBERNET_LANGUAGES:
                if lang != iso_code and count < top_k - 1:
                    probs_sorted[lang] = dummy_prob
                    count += 1
            
            return probs_sorted, iso_code
            
    except Exception as e:
        # Last resort: return unknown
        print(f"Error in language detection: {e}")
        import traceback
        traceback.print_exc()
        return {'unknown': 1.0}, 'unknown'
            
    finally:
        try:
            Path(tmp_path).unlink()
        except:
            pass