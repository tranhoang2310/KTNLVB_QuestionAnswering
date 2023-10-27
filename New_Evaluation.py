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
from nltk.translate import bleu_score, meteor_score
from rouge import Rouge

n_best = 20
max_answer_length = 30
metric = evaluate.load("squad")

def compute_f1(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
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

    predicted_texts = [p['prediction_text'] for p in predicted_answers]
    ground_truth_texts = [ast.literal_eval(t['answers'])['text'][0] for t in theoretical_answers]

    # BLEU
    references = [[word.split() for word in ground_truth_texts]]
    candidates = [word.split() for word in predicted_texts]
    bleu = bleu_score.corpus_bleu(references, candidates)

    # METEOR
    meteor_scores = [meteor_score.single_meteor_score(ref, cand) for ref, cand in zip(ground_truth_texts, predicted_texts)]
    avg_meteor = np.mean(meteor_scores)

    # ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(predicted_texts, ground_truth_texts, avg=True)
    avg_rouge_l = rouge_scores["rouge-l"]["f"]

    # Tính trung bình F1 như trước
    list_f1 = [compute_f1(pred, truth) for pred, truth in zip(predicted_texts, ground_truth_texts)]
    mean_f1 = np.mean(list_f1)

    return {
        "f1": mean_f1,
        "bleu": bleu,
        "meteor": avg_meteor,
        "rouge-l": avg_rouge_l
    }

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
    map_test_dataset = test_dataset.map(data.preprocess_validation_data, batched=True, remove_columns=test_dataset.column_names)
    eval_set_for_model = map_test_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_set_for_model.set_format("torch")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
    trained_model = AutoModelForQuestionAnswering.from_pretrained(model_url).to(device)
    
    with torch.no_grad():
        outputs = trained_model(**batch)
    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy()

    metrics = compute_metrics(start_logits, end_logits, map_test_dataset, test_dataset)
    print(f"Final F1: {metrics['f1'] * 100:.2f}%")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"METEOR: {metrics['meteor']:.4f}")
    print(f"ROUGE-L: {metrics['rouge-l']:.4f}")
