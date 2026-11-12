from pathlib import Path
import evaluate
from eval_entity_extraction import obtain_entity_extraction_p_r_f
from eval_relation_extraction import obtain_relation_extraction_p_r_f
from sklearn.metrics import f1_score, recall_score, accuracy_score


rouge_evalutor = evaluate.load(str(Path(__file__).with_name('rouge.py')))
bleu_evalutor = evaluate.load(str(Path(__file__).with_name('sacrebleu_score.py')))



# 评估指标
def get_score(predictions, references, dataset_info, tokenizer=None) -> dict:
    results = {}
    for metric in dataset_info['metric']:
        if metric == 'rouge-L':
            rougeL = rouge_evalutor.compute(
                    predictions=predictions, 
                    references=references,
                    tokenizer=lambda x: x.split()
                )['rougeL']
            results[metric] = rougeL
        elif metric == 'bleu':
            bleu = bleu_evalutor.compute(
                predictions=predictions,
                references=references,
                tokenize='zh'
            )['score']
            results[metric] = bleu
        elif metric.startswith('entity'):
            exact_p_r_f, inexact_p_r_f = obtain_entity_extraction_p_r_f(predictions, references, dataset_info['label_list'])
            results[metric] = exact_p_r_f['全局']['f1']
        elif metric.startswith('relation'):
            exact_p_r_f, inexact_p_r_f = obtain_relation_extraction_p_r_f(predictions, references, dataset_info['label_list'])
            results[metric] = exact_p_r_f['全局']['f1']
        elif metric == 'acc':
            acc = accuracy_score(references, predictions)
            results[metric] = acc
        elif metric == 'recall':
            recall = recall_score(references, predictions, average="macro")
            results[metric] = recall
        elif 'F1' in metric:
            f1 = f1_score(references, predictions, average="micro" if 'micro' in metric else 'macro')
            results[metric] = f1
        else:
            raise NotImplementedError
    return results
