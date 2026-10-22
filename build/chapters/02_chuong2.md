# Chương 2. Các kỹ thuật học máy cơ bản

## 2.1. Cơ sở lý thuyết

### 2.1.1. Bài toán học có giám sát

Cho tập dữ liệu $\{(\mathbf x_i, y_i)\}_{i=1}^N$, học có giám sát tìm hàm $f_\theta$ sao cho $f_\theta(\mathbf x) \approx y$ bằng cách cực tiểu hóa một hàm mất mát trung bình trên tập huấn luyện. Ba dạng bài toán trong chương này tương ứng ba cặp hàm kích hoạt đầu ra và mất mát:

| Bài toán | Đầu ra | Mất mát |
|---|---|---|
| Phân loại nhị phân | $\hat y = \sigma(z)$ | $-\tfrac1N\sum y\ln\hat y + (1-y)\ln(1-\hat y)$ (BCE) |
| Hồi quy | $\hat y = z$ | $\tfrac{1}{2N}\sum(\hat y - y)^2$ (MSE) |
| Phân loại nhiều lớp | $\hat{\mathbf y} = \operatorname{softmax}(\mathbf z)$ | $-\tfrac1N\sum\sum y_k\ln\hat y_k$ (cross-entropy) |

Một điểm đẹp: ở cả ba cặp, đạo hàm của mất mát theo $z$ ở tầng cuối đều rút gọn thành $(\hat y - y)/N$, nên cùng một hàm lan truyền ngược dùng được cho cả ba bài toán.

Để đánh giá trung thực, dữ liệu được chia thành tập huấn luyện và tập kiểm tra (ở đây 80/20, phân tầng theo nhãn), mọi phép biến đổi có tham số (chuẩn hóa, từ điển TF-IDF, mã hóa theo vị trí) chỉ được `fit` trên tập huấn luyện để tránh rò rỉ thông tin. Với dữ liệu mất cân bằng, accuracy dễ gây hiểu lầm nên ta xem thêm precision, recall, F1 và ROC-AUC.

### 2.1.2. Các mô hình cổ điển

**Hồi quy tuyến tính và hồi quy logistic** là mô hình tuyến tính: $z = \mathbf w^\top\mathbf x + b$, dễ giải thích, huấn luyện nhanh nhưng chỉ cắt không gian đặc trưng bằng một siêu phẳng. **k láng giềng gần nhất (KNN)** không có pha huấn luyện: dự đoán bằng đa số nhãn của $k$ điểm gần nhất, nên chi phí dồn vào lúc dự đoán. **SVM** tìm siêu phẳng cực đại hóa lề; với nhân RBF nó cắt được ranh giới cong, nhưng huấn luyện tốn bình phương số mẫu. **Cây quyết định** chia dữ liệu bằng chuỗi câu hỏi có/không, dễ đọc nhưng dễ quá khớp; **rừng ngẫu nhiên** lấy trung bình nhiều cây huấn luyện trên các mẫu bootstrap để giảm phương sai. Naive Bayes đa thức, mô hình xác suất đơn giản với giả định các từ độc lập, là đường cơ sở truyền thống cho văn bản.

### 2.1.3. Mạng nơ-ron nhiều tầng (MLP)

MLP xếp nhiều tầng $\mathbf h^{(l)} = \operatorname{ReLU}(\mathbf h^{(l-1)}W^{(l)} + \mathbf b^{(l)})$. Nếu bỏ hàm kích hoạt phi tuyến thì cả mạng chỉ là một phép biến đổi tuyến tính, nên ReLU là điều kiện để mạng nhiều tầng mạnh hơn mô hình tuyến tính. Gradient được tính bằng quy tắc dây chuyền, duyệt ngược từ tầng ra: $dW^{(l)} = (\mathbf h^{(l-1)})^\top d\mathbf z^{(l)}$ và $d\mathbf h^{(l-1)} = d\mathbf z^{(l)} (W^{(l)})^\top$. Trọng số cập nhật theo gradient descent $W \leftarrow W - \eta\,dW$ (toàn bộ dữ liệu hoặc từng lô nhỏ), khởi tạo He ($\mathcal N(0, 2/n_{in})$) để phương sai tín hiệu ổn định qua ReLU. Đây là thuật toán 1986 đã nhắc ở Chương 1.

## 2.2. Dữ liệu

Chương này dùng ba bộ dữ liệu, mỗi bộ ứng với một dạng bài toán ở bảng trên. Quy trình làm sạch được ghi lại từng bước (kiểm toán số bản ghi bị loại) trong bài thực hành Assignment 03; số liệu dưới đây lấy lại từ kết quả của bài đó.

{{TABL t_c2_datasets}}

### Diabetes

Cùng nguồn với Chương 1 nhưng làm sạch kỹ hơn: bỏ bản ghi trùng lặp ({{V n(J('ch2/results/a03_system1.json')['audit'][1]['Loại bỏ'])}} dòng), nhóm giới tính "Other", tuổi dưới 1 và BMI ngoài [12, 80], còn {{V n(J('ch2/results/a03_system1.json')['n_clean'])}} hồ sơ ({{V pc(J('ch2/results/a03_system1.json')['pos_rate'],1)}} dương tính). Tám biến đầu vào, trong đó giới tính và tiền sử hút thuốc được mã hóa thứ tự. Năm hồ sơ đầu của bộ dữ liệu:

{{TABL t_c1_diabetes_sample}}

{{FIG s1_eda.png | Diabetes: phân bố nhãn và các biến liên tục theo nhãn}}

**Nhận xét.** Tương quan với nhãn cao nhất là đường huyết ({{V f(J('ch2/results/a03_system1.json')['corr_with_target']['blood_glucose_level'],2)}}) và HbA1c ({{V f(J('ch2/results/a03_system1.json')['corr_with_target']['HbA1c_level'],2)}}), kế đó là tuổi ({{V f(J('ch2/results/a03_system1.json')['corr_with_target']['age'],2)}}) và BMI ({{V f(J('ch2/results/a03_system1.json')['corr_with_target']['bmi'],2)}}). Hai chỉ số xét nghiệm gần như quyết định nhãn nên mô hình nào cũng đạt accuracy cao, và phân biệt chất lượng phải dựa vào recall và F1 của lớp bệnh.

### House Price (Hà Nội)

Hơn 82 nghìn tin rao bán nhà đất Hà Nội từ 08/2019 đến 08/2020, mọi trường số ở dạng chuỗi tiếng Việt ("86,96 triệu/m²") cần bóc tách. Sau khi bỏ dòng thiếu diện tích hoặc đơn giá, diện tích ngoài [10, 1.000] m², giá ngoài [0,3, 100] tỷ và tin trùng, còn {{V n(J('ch2/results/a03_system2.json')['n_clean'])}} tin. Mục tiêu là $\ln(\text{giá})$ chuẩn hóa. Mười lăm đặc trưng gồm $\ln$(diện tích), số phòng ngủ và tầng, các tỷ số, tháng đăng, loại nhà, pháp lý, cộng hai đặc trưng vị trí tạo bằng mã hóa theo mục tiêu có làm mượt (giá trung bình theo quận, theo phường, tính chỉ trên tập huấn luyện).

{{TABL t_c2_housing_sample}}

{{FIG s2_eda.png | House Price: phân bố giá trước và sau khi lấy logarit, quan hệ log-log với diện tích}}

**Nhận xét.** Giá lệch phải mạnh (skewness {{V f(J('ch2/results/a03_system2.json')['skew_raw'],2)}}) và về gần chuẩn sau $\ln$ ({{V f(J('ch2/results/a03_system2.json')['skew_log'],2)}}), nên huấn luyện trên thang log. Tương quan log-log giữa diện tích và giá là {{V f(J('ch2/results/a03_system2.json')['corr_logarea_logprice'],2)}}. Giá trung vị theo đơn vị m² chênh nhau {{V f(J('ch2/results/a03_system2.json')['district_ratio'],1)}} lần giữa quận đắt nhất và rẻ nhất, vì vậy vị trí là đặc trưng quan trọng nhất và mã hóa theo mục tiêu mang lại phần lớn độ chính xác.

### Comments (YouTube)

18.408 bình luận gắn nhãn cảm xúc negative, neutral, positive. Sau khi bỏ dòng thiếu, bình luận trùng và bình luận dưới 3 từ còn {{V n(J('ch2/results/a03_system3.json')['n_clean'])}}. Văn bản được biến thành vectơ TF-IDF 1.000 chiều (unigram và bigram, bỏ từ dừng tiếng Anh), nên ma trận đầu vào thưa {{V pc(J('ch2/results/a03_system3.json')['sparsity'],1)}}.

{{TABL t_c2_comments_sample}}

{{FIG s3_eda.png | Comments: phân bố lớp và độ dài bình luận}}

**Nhận xét.** Ba lớp lệch: positive {{V pc(J('ch2/results/a03_system3.json')['class_counts']['positive']/J('ch2/results/a03_system3.json')['n_clean'],0)}}, neutral {{V pc(J('ch2/results/a03_system3.json')['class_counts']['neutral']/J('ch2/results/a03_system3.json')['n_clean'],0)}}, negative {{V pc(J('ch2/results/a03_system3.json')['class_counts']['negative']/J('ch2/results/a03_system3.json')['n_clean'],0)}}. Các từ nổi bật theo lớp rất khác nhau: lớp positive là "love", "thank", "great"; lớp negative là "bad", "wrong", "hate"; lớp neutral lại chủ yếu là từ dừng tiếng Tây Ban Nha và Bồ Đào Nha ("que", "la", "para"), tức nhãn neutral bị nhiễm bởi bình luận ngoại ngữ. Điều này giới hạn trần chất lượng với đặc trưng từ đơn giản.

## 2.3. Các kỹ thuật học máy cổ điển

Các mô hình dưới đây chạy bằng scikit-learn trên đúng cùng $X$, $y$ (cùng chia tập, cùng chuẩn hóa) mà ba cài đặt MLP ở mục 2.4 sẽ dùng, để so sánh trực tiếp ở mục 2.5. Siêu tham số không dò kỹ (đây là đường cơ sở); SVM và SVR nhân RBF chỉ huấn luyện trên 20.000 mẫu do độ phức tạp bậc hai.

{{TAB t_c2_sk_diabetes}}

{{TAB t_c2_sk_housing}}

{{TAB t_c2_sk_comments}}

{{V c2_sk_comment()}}

## 2.4. MLP: ba cách cài đặt

Cùng một MLP được cài ba lần với cùng kiến trúc, khởi tạo He, thuật toán tối ưu SGD thuần (không động lượng), tốc độ học, số epoch và kích thước lô, chỉ khác framework. Kiến trúc và siêu tham số lấy từ cấu hình tốt nhất tìm được ở Assignment 03.

{{TABL t_c2_config}}

### 2.4.1. Scratch: NumPy thuần

Toàn bộ mạng nằm trong một lớp `MLPScratch` (khoảng 120 dòng cho khởi tạo, lan truyền tiến, lan truyền ngược, huấn luyện). Phần cốt lõi là hai hàm sau; lưu ý dòng `dZ = (cache["A"][-1] - y_true) / N` dùng chung cho cả ba bài toán.

{{CODE ch2/dl_scratch.py | dense_layer}}

{{CODEL ch2/dl_scratch.py | def forward(self, X): | return A, cache}}

{{CODEL ch2/dl_scratch.py | def backward(self, y_true, cache): | return grads}}

Trước khi huấn luyện, gradient giải tích được so với sai phân hữu hạn trung tâm: sai số tương đối lớn nhất là {{V '%.1e' % J('ch2/results/a03_system1.json')['gradcheck_max_err']}} (Diabetes), {{V '%.1e' % J('ch2/results/a03_system2.json')['gradcheck_max_err']}} (House Price) và {{V '%.1e' % J('ch2/results/a03_system3.json')['gradcheck_max_err']}} (Comments), nên backward tự viết là đúng.

### 2.4.2. PyTorch

Autograd thay thế toàn bộ hàm `backward`: ta chỉ khai báo kiến trúc, hàm mất mát và vòng lặp cập nhật.

{{CODEL ch2/ch2_run.py | def run_torch | return metrics(name, D, out), sec}}

### 2.4.3. Keras

Với Keras việc huấn luyện rút gọn còn `compile()` và `fit()`.

{{CODEL ch2/ch2_run.py | def run_keras | return metrics(name, D, out), sec}}

## 2.5. Đánh giá ba cài đặt trên ba bộ dữ liệu

Mỗi cài đặt chạy với ba hạt giống 42, 43, 44 (khác nhau ở khởi tạo trọng số và thứ tự trộn lô; tập huấn luyện và kiểm tra giữ nguyên). Bảng ghi trung bình ± độ lệch chuẩn trên tập kiểm tra; cuối mỗi bảng có thêm mô hình cổ điển tốt nhất ở mục 2.3 để đối chiếu.

{{TAB t_c2_diabetes}}

{{TAB t_c2_housing}}

{{TAB t_c2_comments}}

{{FIG c2_curves.png | Đường mất mát huấn luyện của ba cài đặt (hạt giống 42); trục tung của Diabetes theo thang log}}

Đường mất mát của NumPy và PyTorch gần như trùng nhau trên cả ba bài toán, và Keras trùng với chúng trên House Price và Comments. Ngoại lệ là Diabetes (full-batch, SGD thuần): Keras đi chậm hơn ở khoảng 100 đến 800 epoch đầu rồi tiến sát hai đường kia. Tôi chưa tách riêng nguyên nhân; ứng viên đầu tiên là khác biệt khởi tạo (`he_normal` của Keras lấy mẫu phân phối chuẩn cắt cụt, còn NumPy và PyTorch dùng phân phối chuẩn đầy đủ), vì với full-batch chỉ có 1.500 bước cập nhật nên sai khác ở điểm xuất phát còn thấy rõ ở giữa quá trình.

{{V c2_results_comment()}}

### Thời gian và độ dài code

{{TAB t_c2_time}}

{{V c2_time_comment()}}

## 2.6. Demo: sàng lọc tiểu đường bằng MLP chạy trong trình duyệt

Mô hình scratch của mục 2.4.1 (8 → 32 → 16 → 8 → 1, huấn luyện bằng NumPy thuần theo cấu hình ở Bảng cấu hình, hạt giống 42) được xuất thành tệp JSON gồm trọng số, độ lệch và thống kê chuẩn hóa. Trang `demo/ch2_diabetes.html` đọc các trọng số đó và tính lại toàn bộ lượt lan truyền tiến bằng JavaScript, nên không cần máy chủ hay thư viện học máy nào. Người dùng chỉnh tám chỉ số của một hồ sơ, xem xác suất mắc bệnh, và kéo thanh ngưỡng quyết định để thấy đánh đổi ở mục 2.5: ở ngưỡng 0,5 mô hình dè dặt (recall khoảng 0,65), hạ ngưỡng xuống thì phát hiện nhiều ca bệnh hơn nhưng báo nhầm nhiều hơn.

Để chắc rằng bản JavaScript tính đúng, tôi đối chiếu với đầu ra NumPy trên một hồ sơ kiểm tra: trang cho 5,1% và NumPy cho 5,09%. Trên toàn tập kiểm tra mô hình này đạt accuracy 0,9620 và F1 0,752 ở ngưỡng 0,5, trùng với kết quả hạt giống 42 ở mục 2.5. Trang ghi rõ đây là công cụ minh họa cho môn học, không phải công cụ chẩn đoán.

{{FIGW demo_c2.png | 80 | Demo Chương 2: nhập hồ sơ, xem xác suất và chỉnh ngưỡng quyết định}}

## 2.7. So sánh và nhận xét

**Ba cài đặt cho cùng một mô hình.** Với cùng kiến trúc, khởi tạo và thuật toán tối ưu, NumPy, PyTorch và Keras cho kết quả cùng bậc trên cả ba bài toán, và đường mất mát của chúng gần như trùng nhau (trừ giai đoạn đầu của Keras trên Diabetes). Khác biệt còn lại nhỏ (thường dưới một độ lệch chuẩn hạt giống đến vài lần) có hai nguồn đã biết: thư viện dùng thuật toán sinh số ngẫu nhiên và thứ tự trộn lô khác nhau, và PyTorch/Keras tính bằng số thực 32 bit trong khi NumPy dùng 64 bit.

**Mạng nơ-ron không tự động thắng mô hình cổ điển.** Đây là điểm quan trọng nhất của chương. {{V c2_summary_numbers()}} MLP chỉ thật sự có lợi thế khi bài toán có tương tác phi tuyến mà mô hình tuyến tính không bắt được và đủ dữ liệu để học chúng; trên dữ liệu bảng nhỏ, mô hình dựa trên cây thường là lựa chọn mạnh và rẻ hơn (vài giây hoặc ít hơn so với từ khoảng một đến bốn phút của MLP NumPy). Đây chính là quy luật thứ ba của Chương 1: kỹ thuật mới không xóa kỹ thuật cũ.

**Thời gian và kiểm soát.** Bản scratch dài hơn nhiều dòng code nhưng đổi lại cho kiểm soát tuyệt đối và kiểm chứng được gradient; PyTorch cho cân bằng linh hoạt và tốc độ; Keras ngắn nhất. Chênh lệch thời gian giữa các cài đặt phụ thuộc rất nhiều vào cỡ lô và độ rộng mạng (mục 2.5).

**Hạn chế.** Kết quả phụ thuộc vào một lần chia tập; mô hình cổ điển chưa được dò siêu tham số kỹ; mã hóa đặc trưng (đặc biệt vị trí của House Price) ảnh hưởng đến chất lượng nhiều hơn việc chọn mô hình. Chương 3 và 4 sẽ chuyển sang dữ liệu có cấu trúc không gian (CNN) và thứ tự (RNN), nơi mạng nơ-ron có lý do cấu trúc để vượt mô hình đơn giản.
