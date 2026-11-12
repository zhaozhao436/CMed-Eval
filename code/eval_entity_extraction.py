import json
import sys
# from Tools import import_applied_bilingual_entity_label_dict, import_applied_bilingual_relation_label_dict


scheme = 'III'
test_example_number = 3266


def compute4ee(reference, prediction, ch_entity_label_list):
    exact_result = {ch_entity_label: {'matched_reference_number': 0, 'reference_number': 0, 'matched_prediction_number': 0, 'prediction_number': 0} for ch_entity_label in ch_entity_label_list}
    inexact_result = {ch_entity_label: {'matched_reference_number': 0, 'reference_number': 0, 'matched_prediction_number': 0, 'prediction_number': 0} for ch_entity_label in ch_entity_label_list}

    reference_entity_list = {ch_entity_label: [] for ch_entity_label in ch_entity_label_list}
    if reference != '':
        reference = reference.strip('(').strip('（').strip('）').strip(')').replace('；', ',').replace('，', ',').replace('（', '(').replace('）', ')').split('),(')
        for elem in reference:
            if '@@' in elem:
                [text, ch_entity_label] = elem.split('@@')
                if ch_entity_label in reference_entity_list.keys():
                    reference_entity_list[ch_entity_label].append(text)

    prediction_entity_list = {ch_entity_label: [] for ch_entity_label in ch_entity_label_list}
    if prediction != '':
        prediction = prediction.strip('(').strip('（').strip('）').strip(')').replace('；', ',').replace('，', ',').replace('（', '(').replace('）', ')').split('),(')
        
        for elem in prediction:
            split_elem = elem.split('@@')
            if len(split_elem) != 2:
                continue
            [text, ch_entity_label] = split_elem
            if ch_entity_label in ch_entity_label_list:
                prediction_entity_list[ch_entity_label].append(text)
    
    for ch_entity_label in ch_entity_label_list:
        for reference_text in reference_entity_list[ch_entity_label]:
            flag = False
            for prediction_text in prediction_entity_list[ch_entity_label]:
                if reference_text == prediction_text:
                    flag = True
                    break
            if flag:
                exact_result[ch_entity_label]['matched_reference_number'] += 1
            exact_result[ch_entity_label]['reference_number'] += 1


        for reference_text in reference_entity_list[ch_entity_label]:
            flag = False
            for prediction_text in prediction_entity_list[ch_entity_label]:
                if reference_text in prediction_text or prediction_text in reference_text:
                    flag = True
                    break
            if flag:
                inexact_result[ch_entity_label]['matched_reference_number'] += 1
            inexact_result[ch_entity_label]['reference_number'] += 1

        
        for prediction_text in prediction_entity_list[ch_entity_label]:
            flag = False
            for reference_text in reference_entity_list[ch_entity_label]:
                if reference_text == prediction_text:
                    flag = True
                    break
            if flag:
                exact_result[ch_entity_label]['matched_prediction_number'] += 1
            exact_result[ch_entity_label]['prediction_number'] += 1


        for prediction_text in prediction_entity_list[ch_entity_label]:
            flag = False
            for reference_text in reference_entity_list[ch_entity_label]:
                if reference_text in prediction_text or prediction_text in reference_text:
                    flag = True
                    break
            if flag:
                inexact_result[ch_entity_label]['matched_prediction_number'] += 1
            inexact_result[ch_entity_label]['prediction_number'] += 1

    return exact_result, inexact_result


def obtain_entity_extraction_p_r_f(predictions, references, ch_entity_label_list):
    # reference_prediction_list = []
    # with open(path, 'r') as file:
    #     for line in file:
    #         data = json.loads(line)
    #         reference_prediction_list.append(data)
    # print(len(label_predict_list))

    # en2ch_entity_label_dict, _ = import_applied_bilingual_entity_label_dict()
    # ch_entity_label_list = [v  for [k, v] in en2ch_entity_label_dict.items()]

    """id_list = [0, 1, 10, 11, 100, 101, 1000, 1001]
    for id in id_list:
        print(50 * '*', id, 50 * '*')
        print(test_data[id])
        print(reference_prediction_list[id])"""
    
    total_exact_result = {ch_entity_label: {'matched_reference_number': 0, 'reference_number': 0, 'matched_prediction_number': 0, 'prediction_number': 0} for ch_entity_label in ch_entity_label_list}
    total_inexact_result = {ch_entity_label: {'matched_reference_number': 0, 'reference_number': 0, 'matched_prediction_number': 0, 'prediction_number': 0} for ch_entity_label in ch_entity_label_list}
    for example_id in range(len(predictions)):
        # reference, prediction = reference_prediction_list[2 * example_id]['label'], reference_prediction_list[2 * example_id]['predict']
        reference, prediction = references[example_id], predictions[example_id]
        if reference == '' and prediction == '':
            continue
        exact_result, inexact_result = compute4ee(reference, prediction, ch_entity_label_list)

        for ch_entity_label in ch_entity_label_list:
            total_exact_result[ch_entity_label]['matched_reference_number'] += exact_result[ch_entity_label]['matched_reference_number']
            total_exact_result[ch_entity_label]['reference_number'] += exact_result[ch_entity_label]['reference_number']
            total_exact_result[ch_entity_label]['matched_prediction_number'] += exact_result[ch_entity_label]['matched_prediction_number']
            total_exact_result[ch_entity_label]['prediction_number'] += exact_result[ch_entity_label]['prediction_number']
            total_inexact_result[ch_entity_label]['matched_reference_number'] += inexact_result[ch_entity_label]['matched_reference_number']
            total_inexact_result[ch_entity_label]['reference_number'] += inexact_result[ch_entity_label]['reference_number']
            total_inexact_result[ch_entity_label]['matched_prediction_number'] += inexact_result[ch_entity_label]['matched_prediction_number']
            total_inexact_result[ch_entity_label]['prediction_number'] += inexact_result[ch_entity_label]['prediction_number']
    
    exact_p_r_f = {ch_entity_label: {'precision': 0, 'recall': 0, 'f1': 0} for ch_entity_label in ch_entity_label_list}
    exact_p_r_f['全局'] = {}
    inexact_p_r_f = {ch_entity_label: {'precision': 0, 'recall': 0, 'f1': 0} for ch_entity_label in ch_entity_label_list}
    inexact_p_r_f['全局'] = {}
    
    for ch_entity_label in ch_entity_label_list:
        if total_exact_result[ch_entity_label]['prediction_number'] != 0:
            exact_p_r_f[ch_entity_label]['precision'] = total_exact_result[ch_entity_label]['matched_prediction_number'] / total_exact_result[ch_entity_label]['prediction_number']
        else:
            exact_p_r_f[ch_entity_label]['precision'] = 'None'

        if total_exact_result[ch_entity_label]['reference_number'] != 0:
            exact_p_r_f[ch_entity_label]['recall'] = total_exact_result[ch_entity_label]['matched_reference_number'] / total_exact_result[ch_entity_label]['reference_number']
        else:
            exact_p_r_f[ch_entity_label]['recall'] = 'None'

        if (exact_p_r_f[ch_entity_label]['precision'] != 'None') and (exact_p_r_f[ch_entity_label]['recall'] != 'None') and (exact_p_r_f[ch_entity_label]['precision'] + exact_p_r_f[ch_entity_label]['recall'] != 0):
            exact_p_r_f[ch_entity_label]['f1'] = (2 * exact_p_r_f[ch_entity_label]['precision'] * exact_p_r_f[ch_entity_label]['recall']) / (exact_p_r_f[ch_entity_label]['precision'] + exact_p_r_f[ch_entity_label]['recall'])
        else:
            exact_p_r_f[ch_entity_label]['f1'] = 'None'
        
        if total_inexact_result[ch_entity_label]['prediction_number'] != 0:
            inexact_p_r_f[ch_entity_label]['precision'] = total_inexact_result[ch_entity_label]['matched_prediction_number'] / total_inexact_result[ch_entity_label]['prediction_number']
        else:
            inexact_p_r_f[ch_entity_label]['precision'] = 'None'
        if total_inexact_result[ch_entity_label]['reference_number'] != 0:
            inexact_p_r_f[ch_entity_label]['recall'] = total_inexact_result[ch_entity_label]['matched_reference_number'] / total_inexact_result[ch_entity_label]['reference_number']
        else:
            inexact_p_r_f[ch_entity_label]['recall'] = 'None'
        if (inexact_p_r_f[ch_entity_label]['precision'] != 'None') and (inexact_p_r_f[ch_entity_label]['recall'] != 'None') and (inexact_p_r_f[ch_entity_label]['precision'] + inexact_p_r_f[ch_entity_label]['recall'] != 0):
            inexact_p_r_f[ch_entity_label]['f1'] = (2 * inexact_p_r_f[ch_entity_label]['precision'] * inexact_p_r_f[ch_entity_label]['recall']) / (inexact_p_r_f[ch_entity_label]['precision'] + inexact_p_r_f[ch_entity_label]['recall'])
        else:
            inexact_p_r_f[ch_entity_label]['f1'] = 'None'

    # for ch_entity_label in ch_entity_label_list:
    #     print(ch_entity_label + '的精准结果为：', 'Precision：', exact_p_r_f[ch_entity_label]['precision'], 'Recall：', exact_p_r_f[ch_entity_label]['recall'], 'F1-Score：', exact_p_r_f[ch_entity_label]['f1'])
        # print(ch_entity_label + '的非精准结果为：', 'Precision：', inexact_p_r_f[ch_entity_label]['precision'], 'Recall：', inexact_p_r_f[ch_entity_label]['recall'], 'F1-Score：', inexact_p_r_f[ch_entity_label]['f1'])



    exact_matched_reference_number, exact_reference_number, exact_matched_prediction_number, exact_prediction_number = 0, 0, 0, 0
    inexact_matched_reference_number, inexact_reference_number, inexact_matched_prediction_number, inexact_prediction_number = 0, 0, 0, 0
    for ch_entity_label in ch_entity_label_list:
        exact_matched_reference_number += total_exact_result[ch_entity_label]['matched_reference_number']
        exact_reference_number += total_exact_result[ch_entity_label]['reference_number']
        exact_matched_prediction_number += total_exact_result[ch_entity_label]['matched_prediction_number']
        exact_prediction_number += total_exact_result[ch_entity_label]['prediction_number']
        inexact_matched_reference_number += total_inexact_result[ch_entity_label]['matched_reference_number']
        inexact_reference_number += total_inexact_result[ch_entity_label]['reference_number']
        inexact_matched_prediction_number += total_inexact_result[ch_entity_label]['matched_prediction_number']
        inexact_prediction_number += total_inexact_result[ch_entity_label]['prediction_number']
    
    # 如果prediction都提取不出来，数目改为样本数防止零除
    if exact_prediction_number == 0:    
        exact_prediction_number = len(predictions)
    if inexact_prediction_number == 0:
        inexact_prediction_number = len(predictions)

    exact_precision, exact_recall = exact_matched_prediction_number / exact_prediction_number, exact_matched_reference_number / exact_reference_number
    try:    # 防止零除
        exact_f1 = (2 * exact_precision * exact_recall) / (exact_precision + exact_recall)
    except:
        exact_f1 = 0

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
