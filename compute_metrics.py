import nltk
# Use nltk's meteor_score, which accepts raw strings
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
from collections import Counter
import math
import subprocess
import tempfile
import json
import os
import re

# Function to clean and deduplicate repetitive transcript content
def normalize_repetitive_sentences(text):
    """Removes repeated sentence-like chunks in a long caption."""
    sentences = re.split(r'\.\s+', text.strip())
    seen = set()
    filtered = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            filtered.append(s)
    return '. '.join(filtered) + '.' if filtered else ''

def tokenize(text):
    if not text or not isinstance(text, str):
        return []
    return nltk.word_tokenize(text.lower())

def compute_cider(reference, candidate, n=4):
    # Simplified but functional CIDEr implementation using n-grams
    def get_ngrams(tokens, n):
        return list(zip(*[tokens[i:] for i in range(n)]))
    
    def tfidf_vector(tokens, ref_tokens_list, n):
        tf = Counter(get_ngrams(tokens, n))
        df = Counter()
        for ref in ref_tokens_list:
            df.update(set(get_ngrams(ref, n)))
        idf = {k: math.log(len(ref_tokens_list) / (1 + df[k])) for k in tf}
        return {k: tf[k] * idf.get(k, 0.0) for k in tf}
    
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    scores = []
    for i in range(1, n+1):
        ref_vec = tfidf_vector(ref_tokens, [ref_tokens], i)
        cand_vec = tfidf_vector(cand_tokens, [ref_tokens], i)
        overlap = set(ref_vec) & set(cand_vec)
        dot = sum(ref_vec[k] * cand_vec[k] for k in overlap)
        ref_norm = math.sqrt(sum(v*v for v in ref_vec.values()))
        cand_norm = math.sqrt(sum(v*v for v in cand_vec.values()))
        if ref_norm * cand_norm == 0:
            scores.append(0.0)
        else:
            scores.append(dot / (ref_norm * cand_norm))
    return np.mean(scores)

def compute_spice(reference, candidate):
    # Use external SPICE Java tool; requires Java installed and SPICE jar path
    SPICE_JAR = "/path/to/spice-1.0.jar"  # You must update this path
    with tempfile.TemporaryDirectory() as tmpdir:
        input_json = [{
            "image_id": 0,
            "test": candidate,
            "refs": [reference]
        }]
        input_file = f"{tmpdir}/input.json"
        output_file = f"{tmpdir}/output.json"
        with open(input_file, "w") as f:
            json.dump(input_json, f)
        cmd = ["java", "-jar", SPICE_JAR, input_file, "-cache", tmpdir, "-out", output_file]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(output_file):
            return 0.0
        with open(output_file, "r") as f:
            data = json.load(f)
            return data[0]["scores"]["All"]["f"]

def compute_spider(reference, candidate):
    cider_score = compute_cider(reference, candidate)
    spice_score = compute_spice(reference, candidate)
    return (cider_score + spice_score) / 2

def compute_spider_fl(reference, candidate):
    spider_score = compute_spider(reference, candidate)
    # Add fluency penalty (simplified)
    cand_tokens = tokenize(candidate)
    if len(cand_tokens) < 3:  # Too short
        return spider_score * 0.5
    return spider_score

def compute_fense(reference, candidate):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(candidate, reference)[0]
        return (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3
    except:
        return 0.0

def compute_bleu(reference, candidate):
    try:
        smooth_fn = SmoothingFunction().method1
        return sentence_bleu([tokenize(reference)], tokenize(candidate), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
    except:
        return 0.0

def compute_rouge_l(reference, candidate):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(candidate, reference)[0]
        return scores['rouge-l']['f']
    except:
        return 0.0

def main():
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('wordnet')
    
    # Read files
    # gt_path = '/scratch/cc6946/LLM-AAC/exps/Clotho/slam-aac_Clotho_mlp_melspec_wav/inference_clap_refined/decode_beam2_gt'
    # pred_path = '/scratch/cc6946/LLM-AAC/exps/Clotho/slam-aac_Clotho_mlp_melspec_wav/inference_clap_refined/decode_beam2_pred'
    
    # gt_path = '/scratch/cc6946/LLM-AAC/exps/Clotho/slam-aac_Clotho_qformer_h5_wavelet/inference_clap_refined/decode_beam2_gt'
    # pred_path = '/scratch/cc6946/LLM-AAC/exps/Clotho/slam-aac_Clotho_qformer_h5_wavelet/inference_clap_refined/decode_beam2_pred'
    
    gt_path = '/scratch/cc6946/LLM-AAC/exps/Clotho/slam-aac_Clotho_mlp_h5_cwt/inference_clap_refined/decode_beam2_gt'
    pred_path = '/scratch/cc6946/LLM-AAC/exps/Clotho/slam-aac_Clotho_mlp_h5_cwt/inference_clap_refined/decode_beam2-2_pred'
    
    with open(gt_path, 'r') as f:
        references = [line.strip().split('\t')[1] for line in f if line.strip()]
    
    with open(pred_path, 'r') as f:
        candidates = [line.strip().split('\t')[1] if line.strip() and len(line.strip().split('\t')) > 1 else '' for line in f]
    
    # Initialize metrics
    meteor_scores = []
    cider_scores = []
    spice_scores = []
    spider_scores = []
    spider_fl_scores = []
    fense_scores = []
    bleu_scores = []
    rouge_l_scores = []
    
    # Compute metrics for each pair
    for ref, cand in zip(references, candidates):
        if not cand or not ref:  # Skip empty predictions or references
            continue
        try:
            # Clean and deduplicate repetitive transcript content
            ref = normalize_repetitive_sentences(ref)
            cand = normalize_repetitive_sentences(cand)

            meteor_scores.append(meteor_score([nltk.word_tokenize(ref)], nltk.word_tokenize(cand)))
            cider_scores.append(compute_cider(ref, cand))
            spice_scores.append(compute_spice(ref, cand))
            spider_scores.append(compute_spider(ref, cand))
            spider_fl_scores.append(compute_spider_fl(ref, cand))
            fense_scores.append(compute_fense(ref, cand))
            bleu_scores.append(compute_bleu(ref, cand))
            rouge_l_scores.append(compute_rouge_l(ref, cand))
        except Exception as e:
            print(f"Error processing pair: {e}")
            continue
    
    # Print results
    print(f"Number of valid pairs processed: {len(meteor_scores)}")
    print(f"METEOR: {np.mean(meteor_scores) * 100:.2f}")
    print(f"CIDEr: {np.mean(cider_scores) * 100:.2f}")
    print(f"SPICE: {np.mean(spice_scores) * 100:.2f}")
    print(f"SPIDEr: {np.mean(spider_scores) * 100:.2f}")
    print(f"SPIDEr-FL: {np.mean(spider_fl_scores) * 100:.2f}")
    print(f"FENSE: {np.mean(fense_scores) * 100:.2f}")
    print(f"BLEU: {np.mean(bleu_scores) * 100:.2f}")
    print(f"ROUGE-L (F1): {np.mean(rouge_l_scores) * 100:.2f}")
    print("Note: CIDEr, SPICE, and FENSE are placeholders. Use official implementations for publication-quality results.")

if __name__ == "__main__":
    main()