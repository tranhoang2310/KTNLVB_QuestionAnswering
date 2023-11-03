from flask import Flask, render_template, request, jsonify
import torch
import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from underthesea import sent_tokenize
import pandas as pd
import google_search
import time

app = Flask(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_url = "./trained_model/final_model/"

# Load model and tokenizer just once for efficiency
tokenizer = AutoTokenizer.from_pretrained(model_url)
model = AutoModelForQuestionAnswering.from_pretrained(model_url)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-answer', methods=['POST'])
def get_answer():
    start_time = time.time()  # Khởi tạo bộ đếm thời gian

    question = request.json['query']

    try:
        # Tìm kiếm thông qua Google và lưu kết quả vào output.csv
        google_search.perform_search(question)

        # Đọc file output.csv
        df = pd.read_csv('output.csv')

        # Lấy dữ liệu từ cột 'Context'
        contexts = df['Context'].tolist()

        best_answer = None
        best_confidence = float('-inf')

        for context in contexts:
            # Tách context ra thành các câu bằng tiếng Việt
            sentences = sent_tokenize(context)

            # Ghép các câu lại cho đến khi tổng số token đạt gần 512
            context_segments = []
            segment = ""
            for sentence in sentences:
                new_segment = (segment + " " + sentence).strip()
                if len(tokenizer(new_segment)["input_ids"]) <= 512:
                    segment = new_segment
                else:
                    context_segments.append(segment)
                    segment = sentence
            if segment:
                context_segments.append(segment)

            for segment in context_segments:
                print('Đang xử lý segment')
                inputs = tokenizer(question, segment, return_tensors="pt", truncation=True, max_length=512, padding='max_length')
                with torch.no_grad():
                    outputs = model(**inputs)

                # Confidence score
                start_confidence = outputs.start_logits.argmax()
                end_confidence = outputs.end_logits.argmax()
                total_confidence = end_confidence - start_confidence

                # Lấy câu trả lời tốt nhất dựa trên độ tin cậy
                if total_confidence > best_confidence:
                    best_confidence = total_confidence
                    answer_start_index = outputs.start_logits.argmax()
                    answer_end_index = outputs.end_logits.argmax()
                    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
                    best_answer = tokenizer.decode(predict_answer_tokens)
                    print('================================================================')
                    print('segment', segment)
                    print('Best Answer', best_answer)
                    print('================================================================')

                elapsed_time = time.time() - start_time
                if elapsed_time > 10:
                    print("Đã quá 20 giây! Trả về câu trả lời tốt nhất cho đến nay.")
                    return jsonify({'answer': best_answer if best_answer else "Không tìm thấy câu trả lời."})

        return jsonify({'answer': best_answer if best_answer else "Không tìm thấy câu trả lời."})

    except Exception as e:
        return jsonify({'error': f"Lỗi xảy ra: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
