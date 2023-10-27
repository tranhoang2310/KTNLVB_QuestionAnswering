import pandas as pd

def find_start_position_and_save(csv_file_path, output_csv_path):
    # Đọc file CSV vào DataFrame với encoding 'utf-16-le'
    df = pd.read_csv(csv_file_path, encoding='utf-8')

    # Tạo danh sách để lưu kết quả
    results = []

    # Duyệt qua từng hàng của DataFrame
    for index, row in df.iterrows():
        content = str(row['context'])
        answer = str(row['answers'])

        # Tìm vị trí xuất hiện của answer trong nội dung
        start_position = content.find(answer)

        formatted_string = f"{{'text':['{answer}'],'answer_start':[ {start_position} ]}}"

        # Thêm kết quả vào danh sách
        result_row = {'nội dung': content, 'answer': answer, 'start_position': formatted_string}
        results.append(result_row)

    # Tạo DataFrame mới từ danh sách kết quả
    result_df = pd.DataFrame(results)

    # Lưu DataFrame mới vào file CSV với encoding 'utf-16-le'
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8')

    print(f"Kết quả đã được lưu vào file CSV: {output_csv_path}")

# Đường dẫn đến file CSV của bạn
input_csv_file_path = 'Train_Data.csv'

# Đường dẫn đến file CSV đầu ra
output_csv_file_path = 'Train_Data_OUT.csv'

# Gọi hàm để thực hiện tìm kiếm và lưu kết quả
find_start_position_and_save(input_csv_file_path, output_csv_file_path)
