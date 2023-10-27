import torch
import os
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_url = "./trained_model/final_model/"
    # model_url = "xlm-roberta-base"

    context = "Xử phạt người điều khiển xe mô tô, xe gắn máy (kể cả xe máy điện), các loại xe tương tự xe mô tô và các loại xe tương tự xe gắn máy vi phạm quy tắc giao thông đường bộ. Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây: Không chấp hành hiệu lệnh của đèn tín hiệu giao thông"
    question = "Nếu người điều khiển xe mô tô, xe gắn máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông thì bị xử phạt bao nhiêu tiền?"

    # Tokenize the text and return PyTorch tensors
    tokenizer = AutoTokenizer.from_pretrained(model_url)
    inputs = tokenizer(question, context, return_tensors="pt")

    # Pass your inputs to the model and return the logits
    model = AutoModelForQuestionAnswering.from_pretrained(model_url)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the highest probability from the model output for the start and end positions
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    # Decode the predicted tokens to get the answer
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

    # print answer
    answer = tokenizer.decode(predict_answer_tokens)
    print(answer)
