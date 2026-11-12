import json
import sys
# from Tools import import_applied_bilingual_entity_label_dict, import_applied_bilingual_relation_label_dict


scheme = 'IV'


def compute4re(reference: str, prediction: str, ch_relation_label_list: list):
    exact_result = {ch_relation_label: {'matched_reference_number': 0, 'reference_number': 0, 'matched_prediction_number': 0, 'prediction_number': 0} for ch_relation_label in ch_relation_label_list}
    inexact_result = {ch_relation_label: {'matched_reference_number': 0, 'reference_number': 0, 'matched_prediction_number': 0, 'prediction_number': 0} for ch_relation_label in ch_relation_label_list}

    reference_relation_list = {ch_relation_label: [] for ch_relation_label in ch_relation_label_list}
    if reference != '':
        try:    # json格式
            reference_list = eval(reference.replace('```json', '').replace('```', ''))
            reference = [f'{data["head_entity"]},{data["relation"]},{data["tail_entity"]}' for data in reference_list]
        except Exception as e:      # 非json
            reference = reference.strip('<').strip('>').replace('，', ',').split('>,<')
        for elem in reference:
            for ch_relation_label in ch_relation_label_list:
                mid = ',' + ch_relation_label + ','
                index = elem.find(mid)
                if index != -1:
                    head_entity_text = elem[:index]
                    tail_entity_text = elem[index+len(mid):]
                    reference_relation_list[ch_relation_label].append({'head_entity_text': head_entity_text, 'tail_entity_text': tail_entity_text})
                    break

    prediction_relation_list = {ch_relation_label: [] for ch_relation_label in ch_relation_label_list}
    if prediction != '':
        try:
            prediction_list = eval(prediction.replace('json```', '').replace('```', ''))
            prediction = [f'{data["head_entity"]},{data["relation"]},{data["tail_entity"]}' for data in prediction_list]
        except Exception as e:
            prediction = prediction.strip('<').strip('>').replace('，', ',').split('>,<')
        for elem in prediction:
            for ch_relation_label in ch_relation_label_list:
                mid = ',' + ch_relation_label + ','
                index = elem.find(mid)
                if index != -1:
                    head_entity_text = elem[:index]
                    tail_entity_text = elem[index+len(mid):]
                    prediction_relation_list[ch_relation_label].append({'head_entity_text': head_entity_text, 'tail_entity_text': tail_entity_text})
                    break
    
    for ch_relation_label in ch_relation_label_list:
        for ref_entity_text_pair in reference_relation_list[ch_relation_label]:
            ref_head_entity_text, ref_tail_entity_text = ref_entity_text_pair['head_entity_text'], ref_entity_text_pair['tail_entity_text']
            flag = False
            for pred_entity_text_pair in prediction_relation_list[ch_relation_label]:
                pred_head_entity_text, pred_tail_entity_text = pred_entity_text_pair['head_entity_text'], pred_entity_text_pair['tail_entity_text']
                if ref_head_entity_text == pred_head_entity_text and ref_tail_entity_text == pred_tail_entity_text:
                    flag = True
                    break
            if flag:
                exact_result[ch_relation_label]['matched_reference_number'] += 1
            exact_result[ch_relation_label]['reference_number'] += 1


        for ref_entity_text_pair in reference_relation_list[ch_relation_label]:
            ref_head_entity_text, ref_tail_entity_text = ref_entity_text_pair['head_entity_text'], ref_entity_text_pair['tail_entity_text']
            flag = False
            for pred_entity_text_pair in prediction_relation_list[ch_relation_label]:
                pred_head_entity_text, pred_tail_entity_text = pred_entity_text_pair['head_entity_text'], pred_entity_text_pair['tail_entity_text']
                if (ref_head_entity_text in pred_head_entity_text or pred_head_entity_text in ref_head_entity_text) and (ref_tail_entity_text in pred_tail_entity_text or pred_tail_entity_text in ref_tail_entity_text):
                    flag = True
                    break
            if flag:
                inexact_result[ch_relation_label]['matched_reference_number'] += 1
            inexact_result[ch_relation_label]['reference_number'] += 1


        for pred_entity_text_pair in prediction_relation_list[ch_relation_label]:
            pred_head_entity_text, pred_tail_entity_text = pred_entity_text_pair['head_entity_text'], pred_entity_text_pair['tail_entity_text']
            flag = False
            for ref_entity_text_pair in reference_relation_list[ch_relation_label]:
                ref_head_entity_text, ref_tail_entity_text = ref_entity_text_pair['head_entity_text'], ref_entity_text_pair['tail_entity_text']
                if ref_head_entity_text == pred_head_entity_text and ref_tail_entity_text == pred_tail_entity_text:
                    flag = True
                    break
            if flag:
                exact_result[ch_relation_label]['matched_prediction_number'] += 1
            exact_result[ch_relation_label]['prediction_number'] += 1


        for pred_entity_text_pair in prediction_relation_list[ch_relation_label]:
            pred_head_entity_text, pred_tail_entity_text = pred_entity_text_pair['head_entity_text'], pred_entity_text_pair['tail_entity_text']
            flag = False
            for ref_entity_text_pair in reference_relation_list[ch_relation_label]:
                ref_head_entity_text, ref_tail_entity_text = ref_entity_text_pair['head_entity_text'], ref_entity_text_pair['tail_entity_text']
                if (ref_head_entity_text in pred_head_entity_text or pred_head_entity_text in ref_head_entity_text) and (ref_tail_entity_text in pred_tail_entity_text or pred_tail_entity_text in ref_tail_entity_text):
                    flag = True
                    break
            if flag:
                inexact_result[ch_relation_label]['matched_prediction_number'] += 1
            inexact_result[ch_relation_label]['prediction_number'] += 1


    return exact_result, inexact_result


def obtain_relation_extraction_p_r_f(predictions, references, ch_relation_label_list):
    # reference_prediction_list = []
    # with open(path, 'r') as file:
    #     for line in file:
    #         data = json.loads(line)
    #         reference_prediction_list.append(data)
    # print(len(label_predict_list))

    # en2ch_relation_label_dict, _ = import_applied_bilingual_relation_label_dict()
    # ch_relation_label_list = [v for [k, v] in en2ch_relation_label_dict.items()]

    """id_list = [0, 1, 10, 11, 100, 101, 1000, 1001]
    for id in id_list:
        print(50 * '*', id, 50 * '*')
        print(test_data[id])
        print(reference_prediction_list[id])"""
    
    total_exact_result = {ch_relation_label: {'matched_reference_number': 0, 'reference_number': 0, 'matched_prediction_number': 0, 'prediction_number': 0} for ch_relation_label in ch_relation_label_list}
    total_inexact_result = {ch_relation_label: {'matched_reference_number': 0, 'reference_number': 0, 'matched_prediction_number': 0, 'prediction_number': 0} for ch_relation_label in ch_relation_label_list}
    for example_id in range(len(predictions)):
        # reference, prediction = reference_prediction_list[2 * example_id + 1]['label'], reference_prediction_list[2 * example_id + 1]['predict']
        reference, prediction = references[example_id], predictions[example_id]
        if reference == '' and prediction == '':
            continue
        exact_result, inexact_result = compute4re(reference, prediction, ch_relation_label_list)

        for ch_relation_label in ch_relation_label_list:
            total_exact_result[ch_relation_label]['matched_reference_number'] += exact_result[ch_relation_label]['matched_reference_number']
            total_exact_result[ch_relation_label]['reference_number'] += exact_result[ch_relation_label]['reference_number']
            total_exact_result[ch_relation_label]['matched_prediction_number'] += exact_result[ch_relation_label]['matched_prediction_number']
            total_exact_result[ch_relation_label]['prediction_number'] += exact_result[ch_relation_label]['prediction_number']
            total_inexact_result[ch_relation_label]['matched_reference_number'] += inexact_result[ch_relation_label]['matched_reference_number']
            total_inexact_result[ch_relation_label]['reference_number'] += inexact_result[ch_relation_label]['reference_number']
            total_inexact_result[ch_relation_label]['matched_prediction_number'] += inexact_result[ch_relation_label]['matched_prediction_number']
            total_inexact_result[ch_relation_label]['prediction_number'] += inexact_result[ch_relation_label]['prediction_number']

    exact_p_r_f = {ch_relation_label: {'precision': 0, 'recall': 0, 'f1': 0} for ch_relation_label in ch_relation_label_list}
    exact_p_r_f['全局'] = {}
    inexact_p_r_f = {ch_relation_label: {'precision': 0, 'recall': 0, 'f1': 0} for ch_relation_label in ch_relation_label_list}
    inexact_p_r_f['全局'] = {}
    
    for ch_relation_label in ch_relation_label_list:
        if total_exact_result[ch_relation_label]['prediction_number'] != 0:
            exact_p_r_f[ch_relation_label]['precision'] = total_exact_result[ch_relation_label]['matched_prediction_number'] / total_exact_result[ch_relation_label]['prediction_number']
        else:
            exact_p_r_f[ch_relation_label]['precision'] = 'None'
        if total_exact_result[ch_relation_label]['reference_number'] != 0:
            exact_p_r_f[ch_relation_label]['recall'] = total_exact_result[ch_relation_label]['matched_reference_number'] / total_exact_result[ch_relation_label]['reference_number']
        else:
            exact_p_r_f[ch_relation_label]['recall'] = 'None'
        if (exact_p_r_f[ch_relation_label]['precision'] != 'None') and (exact_p_r_f[ch_relation_label]['recall'] != 'None') and (exact_p_r_f[ch_relation_label]['precision'] + exact_p_r_f[ch_relation_label]['recall'] != 0):
            exact_p_r_f[ch_relation_label]['f1'] = (2 * exact_p_r_f[ch_relation_label]['precision'] * exact_p_r_f[ch_relation_label]['recall']) / (exact_p_r_f[ch_relation_label]['precision'] + exact_p_r_f[ch_relation_label]['recall'])
        else:
            exact_p_r_f[ch_relation_label]['f1'] = 'None'
        
        if total_inexact_result[ch_relation_label]['prediction_number'] != 0:
            inexact_p_r_f[ch_relation_label]['precision'] = total_inexact_result[ch_relation_label]['matched_prediction_number'] / total_inexact_result[ch_relation_label]['prediction_number']
        else:
            inexact_p_r_f[ch_relation_label]['precision'] = 'None'
        if total_inexact_result[ch_relation_label]['reference_number'] != 0:
            inexact_p_r_f[ch_relation_label]['recall'] = total_inexact_result[ch_relation_label]['matched_reference_number'] / total_inexact_result[ch_relation_label]['reference_number']
        else:
            inexact_p_r_f[ch_relation_label]['recall'] = 'None'
        if (inexact_p_r_f[ch_relation_label]['precision'] != 'None') and (inexact_p_r_f[ch_relation_label]['recall'] != 'None') and (inexact_p_r_f[ch_relation_label]['precision'] + inexact_p_r_f[ch_relation_label]['recall'] != 0):
            inexact_p_r_f[ch_relation_label]['f1'] = (2 * inexact_p_r_f[ch_relation_label]['precision'] * inexact_p_r_f[ch_relation_label]['recall']) / (inexact_p_r_f[ch_relation_label]['precision'] + inexact_p_r_f[ch_relation_label]['recall'])
        else:
            inexact_p_r_f[ch_relation_label]['f1'] = 'None'

    # for ch_relation_label in ch_relation_label_list:
    #     print(ch_relation_label + '的精准结果为：', 'Precision：', exact_p_r_f[ch_relation_label]['precision'], 'Recall：', exact_p_r_f[ch_relation_label]['recall'], 'F1-Score：', exact_p_r_f[ch_relation_label]['f1'])
        # print(ch_relation_label + '的非精准结果为：', 'Precision：', inexact_p_r_f[ch_relation_label]['precision'], 'Recall：', inexact_p_r_f[ch_relation_label]['recall'], 'F1-Score：', inexact_p_r_f[ch_relation_label]['f1'])

    exact_matched_reference_number, exact_reference_number, exact_matched_prediction_number, exact_prediction_number = 0, 0, 0, 0
    inexact_matched_reference_number, inexact_reference_number, inexact_matched_prediction_number, inexact_prediction_number = 0, 0, 0, 0
    for ch_relation_label in ch_relation_label_list:
        exact_matched_reference_number += total_exact_result[ch_relation_label]['matched_reference_number']
        exact_reference_number += total_exact_result[ch_relation_label]['reference_number']
        exact_matched_prediction_number += total_exact_result[ch_relation_label]['matched_prediction_number']
        exact_prediction_number += total_exact_result[ch_relation_label]['prediction_number']
        inexact_matched_reference_number += total_inexact_result[ch_relation_label]['matched_reference_number']
        inexact_reference_number += total_inexact_result[ch_relation_label]['reference_number']
        inexact_matched_prediction_number += total_inexact_result[ch_relation_label]['matched_prediction_number']
        inexact_prediction_number += total_inexact_result[ch_relation_label]['prediction_number']

    if exact_prediction_number == 0:
        exact_prediction_number = len(predictions)
    exact_precision, exact_recall = exact_matched_prediction_number / exact_prediction_number, exact_matched_reference_number / exact_reference_number
    try:    # 防止零除
        exact_f1 = (2 * exact_precision * exact_recall) / (exact_precision + exact_recall)
    except:
        exact_f1 = 0
    
    if inexact_prediction_number == 0:
        inexact_prediction_number = len(predictions)
    inexact_precision, inexact_recall = inexact_matched_prediction_number / inexact_prediction_number, inexact_matched_reference_number / inexact_reference_number
    try:
        inexact_f1 = (2 * inexact_precision * inexact_recall) / (inexact_precision + inexact_recall)
    except:
        inexact_f1 = 0
    # print('全局的精准结果为：', 'Precision：', exact_precision, 'Recall：', exact_recall, 'F1-Score：', exact_f1)
    # print('全局的非精准结果为：', 'Precision：', inexact_precision, 'Recall：', inexact_recall, 'F1-Score：', inexact_f1)

    exact_p_r_f['全局']['precision'], exact_p_r_f['全局']['recall'], exact_p_r_f['全局']['f1'] = exact_precision, exact_recall, exact_f1
    inexact_p_r_f['全局']['precision'], inexact_p_r_f['全局']['recall'], inexact_p_r_f['全局']['f1'] = inexact_precision, inexact_recall, inexact_f1



    return exact_p_r_f, inexact_p_r_f
