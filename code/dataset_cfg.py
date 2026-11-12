label_list = {
    "ELGP_entity": ['其他治疗', '手术治疗', '流行病学', '社会学', '部位', '预后', '疾病', '检查', '症状', '药物', '其他'],
    "ELGP_relation": ['阶段', '发病性别倾向', '并发症', '手术治疗', '病因', '传播途径', '多发地区', '治疗后症状', '化疗', '病理生理', '发病率', '病史', '辅助检查', '外侵部位', '转移部位', '就诊科室', '发病年龄', '预后生存率', '发病机制', '临床表现', '发病部位', '遗传因素', '病理分型', '预防', '鉴别诊断', '药物治疗', '组织学检查', '辅助治疗', '影像学检查', '死亡率', '内窥镜检查', '相关（导致）', '相关（转化）', '相关（症状）', '多发群体', '侵及周围组织转移的症状', '风险评估因素', '实验室检查', '放射治疗', '多发季节', '筛查', '预后状况', '高危因素', '同义词'],
    "EEP_entity": ['预后', '其他', '疾病', '检查', '手术治疗', '部位', '药物', '症状', '社会学', 'None', '流行病学', '其他治疗'],
    "EEP_relation": ['多发季节', '药物治疗', '多发群体', '相关（转化）', '预后生存率', '病理分型', '影像学检查', '实验室检查', '病史', '风险评估因素', '同义词', '多发地区', '相关（导致）', '病因', '鉴别诊断', '就诊科室', '临床表现', '死亡率', '发病部位', '发病年龄', '放射治疗', '高危因素', '预后状况', '辅助检查', '内窥镜检查', '组织学检查', '手术治疗', '治疗后症状', '合并症', '辅助治疗', '筛查', '预防', '并发症', '发病率', '发病性别倾向'],
    "TCMM": ['药品', '药物成分', '疾病', '症状', '证候', '疾病分组', '食物', '食物分组', '人群', '药品分组', '药物剂型', '药物性味', '中药功效'],
    "DiaKG_entity": ['疾病', '疾病的分期分型', '病因', '发病机制', '临床表现', '检查方法', '检查指标', '检查指标值', '药物名称', '用药频率', '用药剂量', '用药方法', '非药治疗', '手术', '不良反应', '部位', '程度', '持续时间'],
    "DiaKG_relation": ['检查指标-疾病', '非药治疗-疾病', '疾病的分期分型-疾病', '部位-疾病', '药物-疾病', '病因-疾病', '临床表现-疾病', '手术-疾病', '检查方法-疾病', '发病机制-疾病', '不良反应-药物', '用药剂量-药物', '用药方法-药物', '用药频率-药物', '持续时间-药物'],
    "Yidu-S4K_q1": ['疾病和诊断', '影像检查', '实验室检验', '手术', '药物', '解剖部位'],
    "Yidu-S4K_q2": ['肿瘤原发部位', '原发病灶大小', '转移部位'],
}



non_knowledge_based = {
    "medical_natural_language_understanding": {
        "medical_intent_recognition": [
            {"name": "CMID", "metric": ["acc", "recall", "micro-F1", "macro-F1"], "file_name": "23_CMIDq2_instruction_task197-12k_structured_test.jsonl"}
        ],
        "clinical_terminology_standardization": [
            {"name": "CTS", "metric": ["acc", "recall", "micro-F1", "macro-F1"], "file_name": "28_ClinicalTerminology_instruction_task201-7k_structured_test.jsonl"}
        ]
    },
    "medical_information_extraction": {
        "medical_entity_relation_extraction": [
            {"name": "ELGP_entity", "metric": ["entity-micro-F1"], "label_list": label_list['ELGP_entity'], "file_name": "11_liver_gallbladder_pancreas_all_instruction_entity_structured_test.jsonl"},
            {"name": "ELGP_relation_json", "metric": ["relation-micro-F1"], "label_list": label_list['ELGP_relation'], "file_name": "11_liver_gallbladder_pancreas_all_instruction_relation_json_structured_test.jsonl"},
            {"name": "ELGP_relation", "metric": ["relation-micro-F1"], "label_list": label_list['ELGP_relation'], "file_name": "11_liver_gallbladder_pancreas_all_instruction_relation_structured_test.jsonl"},
            {"name": "EEP_entity", "metric": ["entity-micro-F1"], "label_list": label_list['EEP_entity'], "file_name": "12_odls_1abs_all_instruction_entity_structured_test.jsonl"},
            {"name": "EEP_relation_json", "metric": ["relation-micro-F1"], "label_list": label_list['EEP_relation'], "file_name": "12_odls_1abs_all_instruction_relation_json_structured_test.jsonl"},
            {"name": "EEP_relation", "metric": ["relation-micro-F1"], "label_list": label_list['EEP_relation'], "file_name": "12_odls_1abs_all_instruction_relation_structured_test.jsonl"},
            {"name": "TCMM", "metric": ["entity-micro-F1"], "label_list": label_list['TCMM'], "file_name": "22_PhysicalIdentification_instruction_task196_structured_test.jsonl"},
            {"name": "DiaKG_entity", "metric": ["entity-micro-F1"], "label_list": label_list['DiaKG_entity'], "file_name": "21_DiaKG_instruction_task195_entity_structured_test.jsonl"},
            {"name": "DiaKG_relation", "metric": ["relation-micro-F1"], "label_list": label_list['DiaKG_relation'], "file_name": "21_DiaKG_instruction_task195_relation_structured_test.jsonl"},
            {"name": "Yidu-S4K_q1", "metric": ["entity-micro-F1"], "label_list": label_list['Yidu-S4K_q1'], "file_name": "29_Yidu-S4Kq1_instruction_task202-1k_structured_test.jsonl"},
            {"name": "Yidu-S4K_q2", "metric": ["entity-micro-F1"], "label_list": label_list['Yidu-S4K_q2'], "file_name": "29_Yidu-S4Kq2_instruction_task202-1k_structured_test.jsonl"},
        ],
        "positivity_and_negativity_determination_of_clinical_finding": [
            {"name": "MDCFNPC", "metric": ["acc", "recall", "micro-F1", "macro-F1"], "file_name": "18_CHIP2021_instruction_task192_yinyang_structured_test.jsonl"}
        ],
    }
}


knowledge_based = {
    "medical_knowledge": {
        "fundamental_medical_knowledge_QA": [
            {"name": "CMTK", "metric": ["rouge-L", "bleu"], "file_name": "13_book_exercises_instruction-1k_test.jsonl"},
        ],
        "MedicineExam": [
            {"name": "NLPQE", "metric": ["acc", "recall", "micro-F1", "macro-F1"], "file_name": "35_med_exam_7000_instruction_task010190_4.1k_structured_test.jsonl"},
            {"name": "NLPEC", "metric": ["acc", "recall", "micro-F1", "macro-F1"], "file_name": "16_NLPEC_test_result-0.5k_structured.jsonl"}
        ]
    },
    "inference_of_complex_diagnosis_and_treatment": {
        "specialized_disease_diagnosis_and_treatment": [
            {"name": "DTRC", "metric": ["rouge-L", "bleu"], "file_name": "05_diagnosis_guideline_instruction_all-9k_test.jsonl"}
        ]
    },
    "medical_text_generation": {
        "professional_medical_literature_summary_generation": [
            {"name": "PMLS", "metric": ["rouge-L", "bleu"], "file_name": "01_2020_3_cn_en_instruction-4k_test.jsonl"},
        ],
        "professional_medical_literature_translation": [
            {"name": "PMLT", "metric": ["rouge-L", "bleu"], "file_name": "02_2020_3_cn_en_trans_instruction_task176-0_test.jsonl"}
        ]
    }
}

ethics = {
    "medical_ethics": {
        "clinical_diagnosis_and_treatment_ethics": [
            {"name": "CDTE", "metric": ["rouge-L", "bleu"], "file_name": "50_clinical_diagnosis_treatment_ethics_instruction_test_0.2k.jsonl"}
        ],
        "End-of-Life_care_ethics": [
            {"name": "ELCE", "metric": ["rouge-L", "bleu"], "file_name": "50_end_of_life_care_ethics_instruction_test_0.2k.jsonl"}
        ],
        "organ_transplantation_ethics": [
            {"name": "OTE", "metric": ["rouge-L", "bleu"], "file_name": "50_organ_transplantation_ethics_instruction_test_0.22k.jsonl"}
        ],
        "medical_scientific_research_and_human_experimentation_ethics": [
            {"name": "SRHEE", "metric": ["rouge-L", "bleu"], "file_name": "50_medical_scientific_research_human_experimentation_ethics_instruction_test_0.25k.jsonl"}
        ]
    }
}


google = {
    "google": {
        "multimedQA": [
            {"name": "medqa", "metric": ["acc", "recall", "micro-F1", "macro-F1"], "file_name": "40_multimedQA_medqa_US_task010246_Taiwan_task010247_test.jsonl"},
            {"name": "medmcqa", "metric": ["acc", "recall", "micro-F1", "macro-F1"], "file_name": "40_multimedQA_medmcqa_task010249_187k_test.jsonl"},
            {"name": "pubmedqa", "metric": ["acc", "recall", "micro-F1", "macro-F1"], "file_name": "40_multimedQA_pubmedqa_task010250_test.jsonl"},
            {"name": "mmlu", "metric": ["acc", "recall", "micro-F1", "macro-F1"], "file_name": "40_multimedQA_mmlu_task010251_test.jsonl"},
            {"name": "MedInfo", "metric": ["rouge-L", "bleu"], "file_name": "40_multimedQA_MedInfo2019-QA-Medications_task010252_0_test.jsonl"},
            {"name": "LiveQA", "metric": ["rouge-L", "bleu"], "file_name": "40_multimedQA_LiveQA_MedicalTask_TREC2017-master_task010253_test.jsonl"}
        ]
    }
}
