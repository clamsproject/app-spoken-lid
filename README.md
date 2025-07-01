# AAPB-LID

Proof-of-concept pipeline for **language identification (LID)** on American Archive of Public Broadcasting (AAPB) audio.  
Built around OpenAI Whisper for rapid prototyping. 

## Quick start: 
configurable arguments in <>:

`python app_lid.py <fleurs_mixed_01.wav> 
    --model <tiny> 
    --window <30> 
    --batch <1> 
    --top <3> 
    --out <results/preds.csv>`

To do: revise batch inference


