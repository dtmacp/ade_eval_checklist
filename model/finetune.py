import numpy as np
import argparse
import configparser
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import pdb

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, set_seed, PretrainedConfig


class FineTuner():
    def __init__(self, configpath, seed, minidebug=False) -> None:

        # link to finetuning config file
        configfile = configparser.ConfigParser()
        configfile.read(configpath)
        self.config = configfile["default"]

        # seed
        self.seed = seed

        self.truncation_setting = True if self.config["tokenizer_truncation"] == "True" else False

        # use minidebug mode if given as a flag
        if minidebug:
            self.minidebug = True
        else:
            self.minidebug = False

    def load_data(self):
        """
        Loads the dataset from path specified in config file and returns as HF dataset.
        Check HF for supported data formats.
        """
        # load dataset in HF format splits from csv
        dataset = load_dataset(self.config["data_raw_format"],
                                data_files={"train": self.config["traindata"],
                                            "val": self.config["valdata"],
                                            "test": self.config["testdata"]})
    
        return dataset
    
    def tokenization(self, dataset):

        """
        Tokenizes the dataset splits and removes columns not needed for fine-tuning.
        """
        
        # load the tokenizer corresponding to the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["modelcheckpoint"])

        # use map to add tokenizer output to dataset in order to create a tokenized input
        def tokenize_function(sample):
            return self.tokenizer(sample[self.config["text_column"]], truncation=self.truncation_setting)
        
        # tokenize the dataset using map. using batched=True for faster tokenization.
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # delete columns not needed from dataset
        cols_keep = [c.strip() for c in self.config["data_columns"].split(",")]  # read important columns from config and add to list
        # iterate over splits to check for unneeded columns
        cols_to_delete = set()
        for split, v1 in dataset.items():
            split_info = dataset[split].features
            for column, v2 in split_info.items():
                if column not in cols_keep:
                    cols_to_delete.add(column)
        # remove unneeded columns
        tokenized_datasets = tokenized_datasets.remove_columns(cols_to_delete)

        if self.minidebug:
            # cut the dataset splits to 10 samples each
            tokenized_datasets["train"] = tokenized_datasets["train"].select(range(100))
            tokenized_datasets["val"] = tokenized_datasets["val"].select(range(100))
            tokenized_datasets["test"] = tokenized_datasets["test"].select(range(100))
        
        if self.config["tokenizer_delete_long_inputs"] == "True":
            # inputs that exceed the max input length are not truncated but removed

            # get max input length: either read from fine-tuning config file or read from model config
            if self.config["max_input_length"] in [None, "None"]:
                model_config = PretrainedConfig.get_config_dict(self.config["modelcheckpoint"])[0]
                max_inp_len = int(model_config["max_position_embeddings"])
            else:
                max_inp_len = int(self.config["max_input_length"])

            # remove input samples that exceed the max input length
            for settype in ["train", "val", "test"]:
                old_set = tokenized_datasets[settype]
                new_set = old_set.filter(lambda sample: len(sample["attention_mask"]) <= max_inp_len) 
                diff_sets = len(old_set) - len(new_set)
                print("{} input sample(s) in {} split exceed(s) the maximum length and were removed from the dataset.".format(diff_sets, settype))

                if diff_sets != 0:
                    tokenized_datasets[settype] = new_set
                

        return tokenized_datasets
    
    def finetune(self, tokenized_datasets):

        """
        Load the pre-trained model from checkpoint, performs fine-tuning.
        """
        set_seed(self.seed)

        # define data collator. used for creating batches from tokenized input and padding to max batch length
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # load pre-trained model from checkpoint
        # model is loaded within Trainer
        
        # define evaluation after every epoch
        training_args = TrainingArguments(output_dir=self.config["output_dir"],
                                          num_train_epochs=float(self.config["num_train_epochs"]),
                                          evaluation_strategy="epoch",
                                          seed=self.seed,
                                          per_device_train_batch_size=int(self.config["train_batch_size"])) 
        
        # define the HF trainer
        if self.config["batch_sampling"] == "True": # will use weightedrandomsampling
            print("Training with CustomTrainer.")
            self.trainer = CustomTrainer(
                model_init=self._get_model, # use model_init instead of model directly
                args = training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["val"],
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics
            )
        
        else:
            self.trainer = Trainer(
                model_init=self._get_model, # use model_init instead of model directly
                args = training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["val"],
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics
            )
        
        print()
        print("Start training ...")
        print()

        # start the training 
        self.trainer.train()

        print()
        print("Training completed.")
        print()

    def _compute_metrics(self, eval_preds):
        """
        Helper function used for evaluation during training.
        Used in finetune().
        """
        
        metric = evaluate.load(self.config["eval_metric_training"])
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        return metric.compute(predictions=predictions, references=labels)
    
    def _get_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.config["modelcheckpoint"],
                                                                    num_labels=int(self.config["num_labels"]))
        return model
    
    def predict_test(self, data_to_evaluate):
        """
        Get the predicted labels and the prediction values on a dataset to evaluate.
        To use after fine-tuning.
        """
        predictions = self.trainer.predict(data_to_evaluate)
        prediction_labels = np.argmax(predictions.predictions, axis=-1)

        return prediction_labels, predictions

    def eval_testset(self, data_to_evaluate):
        """
        Get evaluation metrics on a dataset to evaluate.
        To use after fine-tuning.
        """

        print("--- Test set evaluation: ---")
        print(self.trainer.evaluate(data_to_evaluate))

        # get predictions on test set
        predict_traineroutput = self.trainer.predict(data_to_evaluate) 
        predictions = predict_traineroutput.predictions
        # take max over classes
        assert predictions.shape[1] == int(self.config["num_labels"])
        preds = np.argmax(predictions, axis=1)
        assert preds.shape == (predictions.shape[0],)

        # read metrics from config
        eval_metrics = [c.strip() for c in self.config["eval_metric_testset"].split(",")]
        metric_scores = dict()

        for metric_name in eval_metrics:
            try:
                metric = evaluate.load(metric_name)
                metric_name_hf_name = metric_name
                metric_exists = True
            except:
                try: # metric name includes the averaging type e.g. f1_weighted
                    specific_metric = [m for m in metric_name.split("_")]
                    assert len(specific_metric) == 2
                    
                    metric_name_hf_name = specific_metric[0]
                    
                    metric = evaluate.load(metric_name_hf_name, average=specific_metric[1])
                    metric_exists = True
                except:
                    print("Cannot evaluate with metric {} on test set. {} is not a Huggingface evaluate metric.".format(metric_name, metric_name))
                    metric_exists = False
            
            if metric_exists:
                metric_scores[metric_name] = metric.compute(predictions=preds, references=data_to_evaluate["label"])[metric_name_hf_name]
        
        if self.config["eval_confusion_matrix"] == "True":
            # print and save a confusion matrix on the test set predictions

            label_names = [str(l) for l in self.config["eval_confusion_matrix_label_names"].split(",")]
            print(classification_report(y_true=data_to_evaluate["label"], y_pred=preds, target_names=label_names))

            ax = plt.gca()
            plt.title(str(self.config["eval_confusion_matrix_title"]))
            disp = ConfusionMatrixDisplay.from_predictions(y_true=data_to_evaluate["label"], y_pred=preds,
                labels=[i for i in range(int(self.config["num_labels"]))], display_labels=label_names,
                cmap = "OrRd", ax = ax)
            
            disp.figure_.savefig(str(self.config["output_dir"] + "_confusion_matrix.png"), format="png")

        return metric_scores

    def save_finetuned(self):
        """
        Save the fine-tuned model. Saved model can be loaded via from_pretrained().
        """
        
        self.trainer.save_model()

        print("Fine-tuned model was saved to {}.".format(self.config["output_dir"]))

        return None
    
class CustomTrainer(Trainer):
    def get_train_dataloader(self):

        sampler = self.get_balancing_sampler(dataset=self.train_dataset)

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            sampler=sampler,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator
        )

        return train_dataloader
    
    def get_balancing_sampler(self, dataset):
        
        # get number of instances per label
        attr_counts_unsorted = Counter(dataset["label"])
        attr_counts = dict(sorted(attr_counts_unsorted.items()))

        # compute weights using the number of instances per label
        weights = dict()
        for attr, n in attr_counts.items():
            weights[attr] = 1/n
        
        # assign a weight to each instance in the dataset
        sample_weights = [weights[l] for l in dataset["label"]]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            )
    

if __name__== "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--configfile', type=str, default="finetuner_config.ini",
        help='Add path to config file for fine-tuning.'
    )

    parser.add_argument(
        '--minidebug',
        help='When using flag minidebug, a small subset of the data is used.',
        action='store_true'
    )
    args = parser.parse_args()

    # seed
    seed = 76
    set_seed(seed)

    if args.minidebug:
        print("Starting in minidebug mode with data subset.")
        finetuner = FineTuner(configpath=args.configfile, minidebug=True, seed=seed)
    else:
        finetuner = FineTuner(configpath=args.configfile, seed=seed)

    # load dataset specified in config file and tokenize
    dataset = finetuner.load_data()
    tokenized_datasets = finetuner.tokenization(dataset=dataset)

    # fine-tune and save the model
    finetuner.finetune(tokenized_datasets=tokenized_datasets)
    finetuner.save_finetuned()

    # get test set accuracy
    metric_scores = finetuner.eval_testset(data_to_evaluate=tokenized_datasets["test"])
    
    if len(metric_scores) == 0:
        pass
    else:
        for m, v in metric_scores.items():
            print("Test set {} score: {:.3f}".format(m, v))
