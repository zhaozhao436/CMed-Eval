import os
import json
import jsonlines
import argparse
import time
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from tqdm import tqdm
from models import load_model, generate_response
from dataset_cfg import *
from get_score import get_score



default_model_path = './models'
default_dataset_path = './data/test'
default_dataset_path_val = './data/val'

# 载入数据
def load_data(file_name: str) -> list:
    data = []
    with jsonlines.open(os.path.join(default_dataset_path, file_name), 'r') as f:
        for item in f:
            data.append(item)
    return data

def load_val_data(file_name: str) -> list:
    data = []
    with jsonlines.open(os.path.join(default_dataset_path_val, file_name.replace('test', 'val'))) as f:
        for item in f:
            data.append(item)
    return data


# 组织messages
def format_messages(data, data_name: str, few_shot: bool, data_val: list, k: int):
    messages = []
    labels = []
    for d in data:
        instruction = d['instruction'][0] if isinstance(d['instruction'], list) else d['instruction']
        if 'pubmedqa' in data_name:
            instruction = 'Answer the following question, only output "yes" or "no".'
        few_shot_promtp = '下面为参考示例：\n'
        if few_shot:
            # 组织 few_shot
            for i in range(k):
                # 单独处理CMID，去掉“句子：”、“标签：”等
                if 'CMID' in data_name:
                    q, a = data_val[i]["input"].replace("句子：", "").replace("标签：",""), data_val[i]["output"]
                # 处理没有input的
                elif any([key in data_name for key in ['NLPQE', 'CMTK', 'CDTE', 'CDTE', 'SRHEE', 'OTE', 'PMLS', 'PMLT']]):
                    q = data_val[i]['instruction'][0] if isinstance(data_val[i]['instruction'], list) else data_val[i]['instruction']
                    q = q.replace('只输出答案的选项字母。', '')
                    if data_val[i]['input'] != '':
                        q += ('\n' + data_val[i]['input'])
                    a = data_val[i]['output']
                # 有input的
                else:
                    q, a = data_val[i]["input"], data_val[i]["output"]
                few_shot_promtp += f'###参考{i+1}\ninput: {q}\noutput: {a}\n\n\n'
            # 组织 input
            if 'CMID' in data_name:
                input = instruction + few_shot_promtp + 'input: ' + d['input'].replace("句子：", "").replace("标签：","") + '\noutput: '
            elif any([key in data_name for key in ['NLPQE', 'CMTK', 'CDTE', 'CDTE', 'SRHEE', 'OTE', 'PMLS', 'PMLT']]):
                if d['input'] != '':
                    input = f'回答下边的问题。\n' + few_shot_promtp + 'input: ' + instruction + '\n' + d['input'] + '\noutput: '
                else:
                    input = f'回答下边的问题。\n' + few_shot_promtp + 'input: ' + instruction + '\noutput: '
            else:
                input = instruction + few_shot_promtp + 'input: ' + d['input'] + '\noutput: '
        else:
            input = instruction + d['input']
        labels.append(d['output'])
    return input, labels


def run(args):
    global default_model_path, default_dataset_path, default_dataset_path_val
    default_model_path = args.model_root
    default_dataset_path = args.data_dir
    default_dataset_path_val = args.val_dir
    if args.batch_size != 1:
        raise ValueError("This model interface supports --batch_size 1 only.")
    if args.k < 1 and args.few_shot:
        raise ValueError("--k must be positive when --few_shot is enabled.")
    if not any([non_knowledge_based, knowledge_based, ethics]):
        raise ValueError("The original dataset_cfg.py was absent from the source archive. "
                         "Populate dataset_cfg.py with the real task definitions before evaluation.")
    # models to eval
    model_list = args.models.split(',')
    # datasets to eval
    dataset_list = []
    for dataset in args.datasets.split(','):
        if dataset in ['medical_natural_language_understanding', 'medical_information_extraction']:
            dataset_list.extend([d for k, v in non_knowledge_based[dataset].items() for d in v])
        elif dataset in ['medical_knowledge', 'inference_of_complex_diagnosis_and_treatment', 'medical_text_generation']:
            dataset_list.extend([d for k, v in knowledge_based[dataset].items() for d in v])
        elif dataset in ['medical_ethics']:
            dataset_list.extend([d for k, v in ethics[dataset].items() for d in v])
        elif dataset == 'MedQA':
            dataset_list.append({"name": "MedQA", "metric": ["acc", "recall", "micro-F1", "macro-F1"], "file_name": "test.jsonl"})
        else:
            raise(NotImplementedError)

    # generation config
    generation_config = GenerationConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )
    batch_size = args.batch_size

    # print info
    print('=' * 100)
    print('evaluation models: \n', model_list)
    print('evaluation datasets: \n', '\n'.join([str(dataset_dict) for dataset_dict in dataset_list]))
    print('generation config: \n', generation_config.to_json_string())
    print('=' * 100)

    # begin eval
    final_results = ''
    for model_name in model_list:
        # load model
        print('<<<<<<--------' + model_name + '--------->>>>>>' + time.strftime("%Y-%m-%d %X", time.localtime()))
        if model_name == 'med_sft':
            model, tokenizer = load_model(default_model_path, 'llama-2-13b-med_sft_short_ehr_full_epoch2', adapter_path=args.adapter_path)
        elif model_name == 'med_sft_llama3.1':
            model, tokenizer = load_model(default_model_path, 'llama-3.1-8b-sft_full_epoch2_v7.1_all_lr-1e-5_llamafactory', adapter_path=args.adapter_path)
        else:
            model, tokenizer = load_model(default_model_path, model_name, adapter_path=args.adapter_path)

        model.generation_config.temperature=args.temperature
        model.generation_config.top_k=args.top_k
        model.generation_config.top_p=args.top_p
        model.generation_config.do_sample=args.do_sample
        model.generation_config.num_beams=args.num_beams
        model.generation_config.repetition_penalty=args.repetition_penalty
        model.generation_config.max_new_tokens=args.max_new_tokens
        tokenizer.padding_side = "left"
        # print('chat_template:\n', tokenizer.default_chat_template)

        for dataset_info in dataset_list:
            # load dataset
            print('*' * 100)
            print(time.strftime("%Y-%m-%d %X", time.localtime()))
            print('evaluating datasets: {}, metric: {}'.format(dataset_info['name'], dataset_info['metric']))
            data = load_data(dataset_info['file_name'])#[:2]     # test data
            data_val = load_val_data(dataset_info['file_name']) if args.few_shot else []     # val_data for few_shot
            
            with_few_shot = args.few_shot   # and (dataset_info['metric'][0] not in ['rouge-L', 'bleu'])    # few_shot
            if with_few_shot:
                print(f'Use few-shots {args.k}!!!')
            # inference 
            predictions, references = [], []
            for i in tqdm(range(len(data) // batch_size)):
                # format messages
                instruction, labels = format_messages(data[i * batch_size: (i + 1) * batch_size], dataset_info['name'], with_few_shot, data_val, args.k)
                # generate
                # results = generate_response(generation_config, tokenizer, model, messages)
                results = generate_response(model_name, tokenizer, model, instruction)
                # print('response: ', results)
                predictions.extend(results)
                references.extend(labels)
            if len(data) % batch_size != 0:
                # format messages
                instruction, labels = format_messages(data[-len(data) % batch_size: ], dataset_info['name'], with_few_shot, data_val, args.k)
                # generate
                # results = generate_response(generation_config, tokenizer, model, messages)
                results = generate_response(model_name, tokenizer, model, instruction)
                predictions.extend(results)
                references.extend(labels)

            # get score
            score = get_score(predictions, references, dataset_info, tokenizer)
            print('score: ', score)
            # save results
            for i in range(len(data)):
                data[i]['prediction'] = predictions[i]
            print('some examples:')
            for example in data[:2]:
                print(example)
            data.append({"score": score})
            save_path = os.path.join(args.output_dir, f'{model_name}_{dataset_info["name"]}')
            os.makedirs(save_path, exist_ok=True)
            save_name = time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + '-'.join([f'{k}_{v:.4f}' for k, v in score.items()]) + (f'_{args.k}_shot.jsonl' if with_few_shot else '.jsonl')
            with jsonlines.open(os.path.join(save_path, save_name), 'w') as writer:
                writer.write_all(data)
            print('saved to: ', os.path.join(save_path, save_name))
            final_results += (f'{model_name}, {dataset_info["name"]}, {str(score)}\n')
    # save final results
    # with open(os.path.join(args.output_dir, f'{model_name}_all_few_shot.txt' if with_few_shot else f'{model_name}_all.txt'), 'w') as f:
    #     f.write(final_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', default='./models', help='Directory containing model subdirectories')
    parser.add_argument('--data_dir', default='./data/test', help='Test JSONL directory')
    parser.add_argument('--val_dir', default='./data/val', help='Few-shot validation JSONL directory')
    parser.add_argument('--adapter_path', default=None, help='LoRA adapter directory, required for adapter models')
    parser.add_argument('--datasets', type=str, required=True, help='二级分类数据集名，多个数据集英文逗号隔开')
    parser.add_argument('--models', type=str, required=True, help='多个模型用英文逗号分开')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_false")
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.1)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--few_shot', action="store_true")
    parser.add_argument('--k', type=int, default=3, help='number of few-shots')
    args = parser.parse_args()

    run(args)
