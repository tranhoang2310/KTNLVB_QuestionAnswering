import evaluate
from tqdm.auto import tqdm
import collections
import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering
from datasets import load_dataset
import data
import config
import ast

n_best = 20
max_answer_length = 30
metric = evaluate.load("squad")

def compute_f1(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": str(example_id), "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": str(ex["id"]), "answers": ex["answers"]} for ex in examples]


    list_f1 = []

    for pred, truth in zip(predicted_answers, theoretical_answers):
        truth_answer = ast.literal_eval(truth['answers'])
        f1 = compute_f1(pred['prediction_text'], truth_answer['text'][0])
        list_f1.append(f1)
    mean_f1 = np.mean(list_f1)

    return mean_f1

    # return metric.compute(predictions=predicted_answers, references=theoretical_answers)

if __name__ == "__main__":
    ## LOAD CONFIG
    cfg = config.Config(
            # data_path="data_qa_law.csv",
            train_data_path="data_qa_law_train.csv",
            test_data_path="data_qa_law_test.csv",
            model_url="xlm-roberta-base", # "hogger32/xlmRoberta-for-VietnameseQA"
            output_dir="trained_model",
            saved_model_path="trained_model/final_model/",
            optim="adamw_torch",
            test_size=0.2,
            train_size=0.8,
            learning_rate=2e-5,
            weight_decay=0.01,
            batch_size=16,
            n_epoch=100,
            )
    print(cfg)

    test_dataset = load_dataset("csv", data_files=cfg.test_data_path, split="all")

    map_test_dataset = test_dataset.map(data.preprocess_validation_data,
                                             batched=True,
                                             remove_columns=test_dataset.column_names)

    eval_set_for_model = map_test_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_set_for_model.set_format("torch")
    model_url = "./trained_model/final_model/"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
    trained_model = AutoModelForQuestionAnswering.from_pretrained(model_url).to(
        device
    )
    with torch.no_grad():
        outputs = trained_model(**batch)
    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy()
    f1 = compute_metrics(start_logits, end_logits, map_test_dataset, test_dataset)
    print(f"Final F1: {f1 * 100:.2f}%")

