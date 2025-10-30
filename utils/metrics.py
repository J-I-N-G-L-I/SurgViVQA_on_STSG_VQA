import re
import evaluate

def get_nlp_metrics(references, predictions):
    """
    Computes corpus-level NLP evaluation metrics (BLEU, ROUGE, METEOR)
    for predicted text against reference text.
    
    Parameters:
    - references: list of strings (ground truth text)
    - predictions: list of strings (generated text by model)
    
    Prints:
    - Corpus-level metrics (overall scores)
    """
    
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    # ------------------------
    # Corpus-level metrics
    # ------------------------
    results_bleu = bleu.compute(predictions=predictions, references=references)
    results_rouge = rouge.compute(predictions=predictions, references=references)
    results_meteor = meteor.compute(predictions=predictions, references=references)

    print(f"Corpus-level metrics:")
    print(f"  BLEU-1: {results_bleu['precisions'][0]:.6f}, "
          f"BLEU-2: {results_bleu['precisions'][1]:.6f}, "
          f"BLEU-3: {results_bleu['precisions'][2]:.6f}, "
          f"BLEU-4: {results_bleu['precisions'][3]:.6f}, "
          f"Overall BLEU: {results_bleu['bleu']:.6f}")
    print(f"  ROUGE-1: {results_rouge['rouge1']:.6f}, "
          f"ROUGE-2: {results_rouge['rouge2']:.6f}, "
          f"ROUGE-L: {results_rouge['rougeL']:.6f}, "
          f"ROUGE-Lsum: {results_rouge['rougeLsum']:.6f}")
    print(f"  METEOR: {results_meteor['meteor']:.6f}")


def is_keyword_present(gen_ans, keyword):
    """
    Returns True if keyword appears as a full word or phrase in gen_ans.
    Works for single words at beginning, middle, or end.
    """

    gen_ans = gen_ans.lower().strip()
    keyword = keyword.lower().strip()
    
    # For multi-word phrases, don't use word boundaries around spaces
    if ' ' in keyword:
        pattern = re.escape(keyword)
    else:
        pattern = r'\b' + re.escape(keyword) + r'\b'
    
    return bool(re.search(pattern, gen_ans))

def keyword_accuracy(predictions, keyword_references):
    """
    Computes accuracy for single-keyword matching in generated answers.
    
    Parameters:
    - predictions: list of strings (generated answer per sample)
    - keyword_references: list of lists of strings (keywords per sample)
    
    Prints:
    - Single-keyword accuracy (fraction of correct matches)
    """

    single_correct = 0
    single_total = 0

    for gen_ans, ref_kw in zip(predictions, keyword_references):

        if len(ref_kw) == 1:
            single_total += 1
            if is_keyword_present(gen_ans, ref_kw[0]):
                single_correct += 1

    single_acc = single_correct / single_total if single_total else None

    print(f"Single-keyword accuracy: {single_acc:.4f}" if single_acc is not None else "No single-keyword samples")


def evaluate_model(references, predictions, keyword_references):
    """
    Complete evaluation pipeline combining:
    - NLP metrics (BLEU, ROUGE, METEOR)
    - Single-keyword accuracy
    
    Parameters:
    - references: list of reference texts
    - predictions: list of generated texts
    - keyword_references: list of lists of keywords per sample
    """

    get_nlp_metrics(references, predictions)
    keyword_accuracy(predictions, keyword_references)