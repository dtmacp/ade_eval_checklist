# Adaptation of CheckList for ADE Detection

This project adapts [CheckList](https://github.com/marcotcr/checklist)<sup>1</sup>, a behavioural testing approach, to the task of Adverse Drug Effect (ADE) detection. 

An ADE is any harmful consequence from medical drug intake. This project focusses on detecting ADEs in user-generated reports as can be found on social media. The ADE detection task is framed as a binary classification task (ADE/no ADE). CheckList was used to inspire capability-based testing of a model for ADE detection. Tested capabilities include temporal order, positive sentiment, beneficial effect, and negation. Templates are created and used to prepare test cases that cover the four capabilities. An example is given below:

> After I was medicated with {drug}, I suffered from {ade}.

The placeholders for `{drug}` and `{ade}` can be filled with drug names and ADE entities from the annotated Psychiatric Treatment Adverse Reactions (PsyTAR)<sup>2</sup> corpus. The data is not part of this repository.

## Get Started

Create an environment using python version 3.8.10 
```
$ pip install -r requirements.txt
```

## CheckList Adaptation for ADE Detection
You can use the tests for ADE detection evaluation with your own fine-tuned BertforSequenceClassification (Huggingface) model.

Run `checklist_work/checklist_tests.py` which uses a customized CheckList test suite (`checklist_work/checklist_customized.py`). The code in this project uses parts of the [original CheckList code](https://github.com/marcotcr/checklist). Add the paths to extracted entities for ADEs, drug names and the path to the fine-tuned model.

Run all tests:
```
$ python checklist_tests.py \
    --temporal_order \
    --positive_sentiment \
    --beneficial_effect \
    --true_beneficial_effect_gold_label 0 \
    --negation \
    --ade_source PATH_TO_ADE_ENTITIES \
    --mild_ade_source PATH_TO_MILD_ADE_ENTITIES \
    --drug_source PATH_TO_DRUG_ENTITIES \
    --model PATH_TO_FINETUNED_MODEL
```
The Positive Sentiment test will use a ADE fill-ins from a list of less severe ADEs. Deactivate this behaviour if needed:
```
$ python checklist_tests.py \
    --positive_sentiment \
    --mild_ade_source None
```
Inspect default values for sampling of templates and entities as well as other arguments:
```
$ python checklist_tests.py -h
```

Entities to fill the CheckList templates are extracted from the PsyTAR corpus using the steps in `entity_extraction/extract_entities.ipynb`. 

## Model Fine-tuning
Code for fine-tuning a BertforSequenceClassification model such as [BioRedditBERT](https://huggingface.co/cambridgeltl/BioRedditBERT-uncased)<sup>3</sup> can be found in the `model/` folder. The fine-tuning data and the fine-tuned model are not included. Set up the config file for fine-tuning by adapting the arguments in `model/finetuner_config_bioredditbert.ini`.

Fine-tune by running
```
$ python finetune.py --configfile finetuner_config_bioredditbert.ini
```
## References
<sup>1</sup> Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://aclanthology.org/2020.acl-main.442/). In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4902–4912, Online. Association for Computational Linguistics. \
<sup>2</sup> Zolnoori, Maryam et al. 2019. [A systematic approach for developing a corpus of patient reported adverse drug events: A case study for SSRI and SNRI medications.](https://pubmed.ncbi.nlm.nih.gov/30611893/). Journal of biomedical informatics vol. 90 (2019): 103091. \
<sup>3</sup> Marco Basaldella, Fangyu Liu, Ehsan Shareghi, and Nigel Collier. 2020. [COMETA: A Corpus for Medical Entity Linking in the Social Media](https://aclanthology.org/2020.emnlp-main.253/). In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3122–3137, Online. Association for Computational Linguistics.