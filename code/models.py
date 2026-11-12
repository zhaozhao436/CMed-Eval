import os
import re
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from peft import PeftModel



def load_model(model_path, model_name, adapter_path=None):
    if model_name in ['qwen_sft', 'med_sft_llama3.1_lora', 'zhongjing'] and not adapter_path:
        raise ValueError("Supply --adapter_path for this LoRA model.")
    # HuatuoGPT2载入需要修改代码：https://github.com/baichuan-inc/Baichuan2/issues/204#issuecomment-1756867868
    if model_name in ['Baichuan2-13B-Chat', 'Qwen1.5-14B-Chat', 'chatglm3-6b', 'Meta-Llama-3-8B-Instruct', 'HuatuoGPT2-13B', 'llama-3.1-8b-sft_full_epoch2_v7.1_all_lr-1e-5_llamafactory']:
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_path, model_name),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            # attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, model_name), 
            trust_remote_code=True,
        )
    elif model_name in ['llama3.1-8B']:
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_path, model_name),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, model_name), 
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name in ['ChatDoctor', 'ChiMed-GPT', 'ziya-llama-13b-medical-merged', 'llama-2-13b-med_sft_short_ehr_full_epoch2']:
        model = LlamaForCausalLM.from_pretrained(
            os.path.join(model_path, model_name),
            device_map="auto",
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
        tokenizer = LlamaTokenizer.from_pretrained(
            os.path.join(model_path, model_name),
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name in ['zhongjing']:
        model = LlamaForCausalLM.from_pretrained(
            os.path.join(model_path, model_name, 'base'),
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        tokenizer = LlamaTokenizer.from_pretrained(
            os.path.join(model_path, model_name, 'base'),
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = PeftModel.from_pretrained(model, adapter_path)
    elif model_name in ['qwen_sft']:
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_path, 'Qwen1.5-14B-Chat'),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            #  attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, 'Qwen1.5-14B-Chat'), 
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    elif model_name in ['med_sft_llama3.1_lora']:
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_path, 'Meta-Llama-3.1-8B'),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, 'Meta-Llama-3.1-8B'), 
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model, tokenizer


def generate_response(model_name, tokenizer, model, instruction):
    if model_name == 'Baichuan2-13B-Chat':
        # print(instruction)
        message = [{'role': 'user', 'content': instruction}]
        response = model.chat(tokenizer, message)
    elif model_name == 'Qwen1.5-14B-Chat' or model_name == 'qwen_sft':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    elif model_name == 'chatglm3-6b':
        model.generation_config.pad_token_ids = tokenizer.pad_token_id
        response, history = model.chat(tokenizer, instruction, history=[])
    elif model_name in ['Meta-Llama-3-8B-Instruct']:
        message = [{'role': 'user', 'content': instruction}]
        input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=tokenizer.chat_template
        ).to(model.device)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = model.generate(
            input_ids,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response, skip_special_tokens=True)
    elif model_name in ['llama3.1-8B']:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model, 
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        )
        response = pipeline(instruction, max_new_tokens=1)[0]['generated_text'][len(instruction):]
        # print(response)
    elif model_name == 'ChatDoctor':
        fulltext = "If you are a doctor, please answer the medical questions based on the patient's description. \n\nPatient: " +\
             instruction + "\n\nChatDoctor: "
        gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
        generated_ids = model.generate(
                gen_in,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
            )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?
        text_without_prompt = generated_text[len(fulltext):]
        response = text_without_prompt.split("Patient: ")[0].strip()
    elif model_name == 'HuatuoGPT2-13B':
        generation_config = GenerationConfig(
            temperature=0.3,
            # top_k=40,
            top_p=0.7,
            do_sample=True,
            # num_beams=1,
            repetition_penalty=1.1,
            max_new_tokens=16,
        )
        message = [{'role': 'user', 'content': instruction}]
        response = model.chat(tokenizer, message, generation_config=generation_config)
        # print(response)
    elif model_name == 'zhongjing':
        full_text = f'<human>:{instruction}\n<bot>:'    # https://github.com/SupritYoung/Zhongjing/blob/a105934245c265b84f77dbb070b568e2a79688b4/src/llmtuner/extras/template.py#L406
        input_ids = tokenizer(full_text, return_tensors="pt").input_ids.cuda()
        generated_ids = model.generate(
            input_ids=input_ids,
            # pad_token_id=tokenizer.eos_token_id,
        )[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # print(generated_text)
        response = generated_text[len(full_text):]
    elif model_name in ['ChiMed-GPT', 'ziya-llama-13b-medical-merged']:
        if model_name == 'ChiMed-GPT':
            query = f"[human]:{instruction}\n[bot]:"
        else:
            query = f"<human>:{instruction}\n<bot>:"
        input_ids = tokenizer(query, return_tensors="pt").input_ids.cuda()
        generate_ids = model.generate(
            input_ids,
            max_new_tokens=1, 
            do_sample=True, 
            top_p=0.9)[0]
        generated_text = tokenizer.decode(generate_ids, skip_special_tokens=True)
        if model_name == 'ChiMed-GPT':
            response = generated_text[len(query):]
        else:
            response = generated_text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):].strip()
        # print('generated text:', generated_text)
        # print('response:', response)
    elif model_name in ['med_sft']:
        PROMPT_TEMPLATE = (
            "[INST] <<SYS>>\n"
            "You are a helpful assistant. 你是一个乐于助人、尊重他人且诚实的助手。请始终以尽可能有帮助的方式回答问题，同时确保安全。您的回答不应包含任何有害、不道德、种族主义、性别歧视、有毒、危险或非法内容。请确保您的回应在社会上是中立和积极的。如果某个问题毫无意义，或者在事实上不连贯，解释原因而不是回答不正确的内容。如果您不知道某个问题的答案，请不要分享虚假信息。\n"
            "<</SYS>>\n\n{instruction} [/INST]"
        )
        full_text = PROMPT_TEMPLATE.format_map({'instruction': instruction})
        input_ids = tokenizer(full_text, return_tensors="pt").input_ids.cuda()
        generated_ids = model.generate(
            input_ids=input_ids,
            # pad_token_id=tokenizer.eos_token_id,
        )[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # print(generated_text)
        response = generated_text[len(full_text):]
        pattern = r"(\['(?:', ')*'\])"
        response = re.sub(pattern, '', response).strip()
    elif model_name in ['med_sft_llama3.1', 'med_sft_llama3.1_lora']:
        system_format='<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. 你是一个乐于助人、尊重他人且诚实的助手。请始终以尽可能有帮助的方式回答问题，同时确保安全。您的回答不应包含任何有害、不道德、种族主义、性别歧视、有毒、危险或非法内容。请确保您的回应在社会上是中立和积极的。如果某个问题毫无意义，或者在事实上不连贯，解释原因而不是回答不正确的内容。如果您不知道某个问题的答案，请不要分享虚假信息。\n<|eot_id|>'
        user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        input_text = system_format + user_format.format(content=instruction)
        inputs = tokenizer(input_text, return_tensors="pt")  #add_special_tokens=False ?
        generation_output = model.generate(
            input_ids = inputs["input_ids"].cuda(),
            attention_mask = inputs['attention_mask'].cuda(),
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=512
        )
        s = generation_output[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        # print(output)
        response = output.split("assistant\n\n")[-1].strip()
        # response = output[len(input_text):]
        # print(response)
    return [response.strip()]

