# -*- coding: utf-8 -*-
from pathlib import Path
C = Path(r"C:\DATA\TieuLuan01\build\chapters")

DEMO = {
"01_chuong1.md": ("## 1.6. Tổng kết chương", "## 1.7. Tổng kết chương", """## 1.6. Demo: perceptron và XOR trong trình duyệt

Để người đọc tự thử thay vì chỉ xem hình, thí nghiệm ở mục 1.5.1 được đóng gói thành một trang web tĩnh (`demo/ch1_perceptron.html`, không cần máy chủ). Người dùng chọn AND, OR hoặc XOR rồi bấm huấn luyện perceptron hay MLP 2-4-1; trang vẽ vùng quyết định ngay trên canvas. Toàn bộ thuật toán (quy tắc sửa lỗi của Rosenblatt và lan truyền ngược cho mạng hai tầng) được viết lại bằng JavaScript, cùng thuật toán với mã Python ở trên, nên kết quả khớp: perceptron hội tụ với AND và OR, không hội tụ với XOR (đúng 2/4 điểm), MLP đúng 4/4.

{{FIGW demo_c1.png | 62 | Demo Chương 1: perceptron chỉ đạt 2/4 điểm trên XOR (trên), MLP 2-4-1 đạt 4/4 (dưới)}}

"""),
"02_chuong2.md": ("## 2.6. So sánh và nhận xét", "## 2.7. So sánh và nhận xét", """## 2.6. Demo: sàng lọc tiểu đường bằng MLP chạy trong trình duyệt

Mô hình scratch của mục 2.4.1 (8 → 32 → 16 → 8 → 1, huấn luyện bằng NumPy thuần theo cấu hình ở Bảng cấu hình, hạt giống 42) được xuất thành tệp JSON gồm trọng số, độ lệch và thống kê chuẩn hóa. Trang `demo/ch2_diabetes.html` đọc các trọng số đó và tính lại toàn bộ lượt lan truyền tiến bằng JavaScript, nên không cần máy chủ hay thư viện học máy nào. Người dùng chỉnh tám chỉ số của một hồ sơ, xem xác suất mắc bệnh, và kéo thanh ngưỡng quyết định để thấy đánh đổi ở mục 2.5: ở ngưỡng 0,5 mô hình dè dặt (recall khoảng 0,65), hạ ngưỡng xuống thì phát hiện nhiều ca bệnh hơn nhưng báo nhầm nhiều hơn.

Để chắc rằng bản JavaScript tính đúng, tôi đối chiếu với đầu ra NumPy trên một hồ sơ kiểm tra: trang cho 5,1% và NumPy cho 5,09%. Trên toàn tập kiểm tra mô hình này đạt accuracy 0,9620 và F1 0,752 ở ngưỡng 0,5, trùng với kết quả hạt giống 42 ở mục 2.5. Trang ghi rõ đây là công cụ minh họa cho môn học, không phải công cụ chẩn đoán.

{{FIGW demo_c2.png | 80 | Demo Chương 2: nhập hồ sơ, xem xác suất và chỉnh ngưỡng quyết định}}

"""),
"03_chuong3.md": ("## 3.6. So sánh và nhận xét", "## 3.7. So sánh và nhận xét", """## 3.6. Demo: phân loại cảm xúc bằng Conv1D chạy trong trình duyệt

Mạng văn bản của mục 3.3 (Embedding 15.000 × 32 → Conv1D 64 → MaxPool → Conv1D 64 → GlobalMaxPool → Dense 3, tức 498.755 tham số) được huấn luyện lại bằng PyTorch với hạt giống 42 và 4 epoch, rồi xuất sang JSON (khoảng 3 MB: từ điển, ma trận embedding, các bộ lọc). Trang `demo/ch3_sentiment.html` tách token bằng cùng biểu thức chính quy `[a-z]+`, ánh xạ sang chỉ số từ điển, đệm về 60 token, rồi tự tính embedding, hai phép convolution, pooling và softmax bằng JavaScript. Người dùng gõ một bình luận và thấy xác suất ba lớp, kèm danh sách token để biết từ nào ngoài từ điển (bị thay bằng `<unk>`).

Kiểm chứng: xác suất do JavaScript tính trùng với PyTorch trên câu mẫu. Mô hình dùng cho demo đạt accuracy {{V f(J('demo/data/ch3_acc.json')['acc'], 3)}} trên 3.672 bình luận kiểm tra, thấp hơn một chút so với bảng ở mục 3.4 vì vòng lặp huấn luyện của demo được viết gọn hơn (kích thước lô 64, tốc độ học 1,5e-3, không dừng sớm). Điều này không ảnh hưởng đến mục đích minh họa, nhưng nghĩa là con số của demo và con số ở mục 3.4 là hai lần huấn luyện khác nhau. Giới hạn đáng chú ý: mô hình chỉ biết tiếng Anh và tập huấn luyện lệch về lớp tích cực, nên câu ngắn hoặc ngoại ngữ thường bị đẩy về trung tính hoặc tích cực.

{{FIGW demo_c3.png | 80 | Demo Chương 3: nhập bình luận, xem xác suất ba lớp và các token}}

"""),
"04_chuong4.md": ("## 4.6. So sánh và nhận xét", "## 4.7. So sánh và nhận xét", """## 4.6. Demo: dự báo biến động AMZN bằng GRU chạy trong trình duyệt

Mô hình GRU 32 đơn vị ẩn của mục 4.3.1 (NumPy thuần, lan truyền ngược theo thời gian tự viết, hạt giống 42) được xuất cùng 736 cửa sổ kiểm tra. Trang `demo/ch4_volatility.html` cài đặt lại phép tính GRU bằng JavaScript theo đúng công thức cổng đặt lại, cổng cập nhật và ứng viên của PyTorch, đọc cửa sổ 60 phiên đã chọn và trả về biến động 5 phiên tới. Người dùng kéo thanh trượt để chọn phiên (2021 - 2023) và so ba con số: dự báo của GRU, giá trị thực và đường cơ sở "biến động 20 phiên qua". Thanh trượt thứ hai là một kịch bản giả định: nhân độ lớn lợi suất và biên độ của 5 phiên cuối với một hệ số để xem dự báo phản ứng thế nào khi thị trường đột ngột biến động mạnh hay yên ắng.

Kiểm chứng: ở phiên 21/05/2021 trang cho dự báo 1,38%, trùng với đầu ra NumPy (1,38%). Trên cả 736 phiên, mô hình này có R² = 0,288, ngang hồi quy Ridge ở mục 4.4.3. Cần nhớ kết luận của chương: đây là dự báo độ lớn của biến động, bài toán dự báo được, còn chiều của giá thì không; trang chỉ phục vụ minh họa học thuật.

{{FIGW demo_c4.png | 80 | Demo Chương 4: chọn phiên kiểm tra, xem dự báo của GRU, giá trị thực và đường cơ sở}}

"""),
}
for fn, (old, new, block) in DEMO.items():
    p = C / fn
    t = p.read_text(encoding="utf-8")
    assert old in t, fn
    t = t.replace(old, block + new)
    p.write_text(t, encoding="utf-8")
print("ok")
