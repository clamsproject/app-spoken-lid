import torch
from whisper.tokenizer import LANGUAGES  # 99 language codes

def _probs_to_dict(prob_obj):

    # dict 
    # list/tuple of floats
    # list/tuple of dicts {'language': str, 'probability': float}
    # torch.Tensor
    # Returns {lang_code: prob}

    # dict
    if isinstance(prob_obj, dict):
        return {k: float(v) for k, v in prob_obj.items()}

    # tensor --> list
    if isinstance(prob_obj, torch.Tensor):
        prob_obj = prob_obj.tolist()

    # list or tuple 
    if isinstance(prob_obj, (list, tuple)):
        first = prob_obj[0]
        # list of floats
        if isinstance(first, (float, int)):
            return {lang: float(p) for lang, p in zip(LANGUAGES, prob_obj)}
        # list of dicts with keys
        if isinstance(first, dict):
            out = {}
            for d in prob_obj:
                lang = d.get("language") or d.get("lang")
                prob = d.get("probability") or d.get("score") or d.get("prob")
                if lang is not None and prob is not None:
                    out[lang] = float(prob)
            return out
   
    raise TypeError("Unrecognized probability format from detect_language()")


def detect_language_by_chunk(model, mel, tokenizer):
    """
    returns (probs_sorted: dict[str,float], iso_code: str)
    for a single 30s Mel chunk.
    """
    
    mel = mel.to(next(model.parameters()).device)

    # whisper’s built-in detection
    lang_token, raw_probs = model.detect_language(mel)
    if isinstance(raw_probs, list) and len(raw_probs) == 1 and isinstance(raw_probs[0], dict):
        raw_probs = raw_probs[0]
    # print(raw_probs) 

    # decode the top‐lang token → ISO code
    if isinstance(lang_token, torch.Tensor):
        lang_token = int(lang_token.item())
        iso_code = tokenizer.decode([lang_token]).strip()
    elif isinstance(lang_token, int):
        iso_code = tokenizer.decode([lang_token]).strip()
    else:  # already a string
        iso_code = lang_token

    # normalize raw_probs into a {lang: float} dict
    probs_dict = {}

    # 1. if already a dict
    if isinstance(raw_probs, dict):
        for k, v in raw_probs.items():
            probs_dict[k] = float(v)

    # 2. if torch.Tensor
    elif isinstance(raw_probs, torch.Tensor):
        for i, p in enumerate(raw_probs.tolist()):
            probs_dict[LANGUAGES[i]] = float(p)

    # 3. if a list/tuple of dicts: [{"lang":..,"prob":..}, ...]
    elif isinstance(raw_probs, (list, tuple)) and isinstance(raw_probs[0], dict):
        for entry in raw_probs:
            lang = entry.get("language") or entry.get("lang")
            prob = entry.get("probability") or entry.get("score") or entry.get("prob")
            if lang is not None and prob is not None:
                probs_dict[lang] = float(prob)

    # 4. if a list/tuple of floats
    elif isinstance(raw_probs, (list, tuple)) and isinstance(raw_probs[0], (float, int)):
        for i, p in enumerate(raw_probs):
            probs_dict[LANGUAGES[i]] = float(p)

    else:
        raise TypeError(f"Unrecognized format for raw_probs: {type(raw_probs)}")

    probs_sorted = dict(
        sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)
    )

    return probs_sorted, iso_code

