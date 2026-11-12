"""Task configuration placeholder, NOT the original benchmark configuration.

The original dataset_cfg.py was not included in the supplied source archive.
Restore the real task-to-dataset mappings before running main.py. No benchmark
labels, file names or metrics have been invented here.

Each mapping has the shape:
    {task_group: {task_name: [dataset_info, ...]}}
Each dataset_info requires name, file_name and metric (a list of metric names).
Entity/relation extraction also requires label_list.

Supported outer groups in main.py:
non_knowledge_based: medical_natural_language_understanding,
                     medical_information_extraction
knowledge_based: medical_knowledge, inference_of_complex_diagnosis_and_treatment,
                 medical_text_generation
ethics: medical_ethics
"""

non_knowledge_based = {}
knowledge_based = {}
ethics = {}
