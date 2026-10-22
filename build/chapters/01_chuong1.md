# Chương 1. Lịch sử phát triển của trí tuệ nhân tạo

## 1.1. Trí tuệ nhân tạo là gì

Trí tuệ nhân tạo là lĩnh vực nghiên cứu cách làm cho máy tính thực hiện những việc mà con người cần đến trí thông minh: nhận thức, suy luận, học và ra quyết định. Thuật ngữ *artificial intelligence* do John McCarthy đề xuất trong đề cương hội thảo Dartmouth mùa hè 1956, cùng với Marvin Minsky, Nathaniel Rochester và Claude Shannon [1]. Trước đó, Alan Turing đã đặt câu hỏi "máy có thể nghĩ không" và đề xuất phép thử mang tên ông trong bài báo năm 1950 [2].

Ba khái niệm thường bị dùng lẫn lộn nằm lồng vào nhau: học sâu (deep learning) là một nhánh của học máy (machine learning), và học máy là một nhánh của AI. AI còn bao gồm cả những hệ thống không học gì, chẳng hạn hệ chuyên gia dựa trên luật do con người viết. Sự khác nhau này chính là sợi chỉ xuyên suốt lịch sử: AI đi từ việc **người viết luật cho máy** sang việc **máy tự rút luật từ dữ liệu**.

## 1.2. Các giai đoạn phát triển

### Giai đoạn nền móng (1943 - 1956)

McCulloch và Pitts (1943) mô tả nơ-ron nhân tạo đầu tiên như một ngưỡng logic, chứng minh rằng mạng các đơn vị như vậy có thể tính các hàm logic [3]. Hebb (1949) đề xuất quy tắc học "các nơ-ron cùng hoạt động thì liên kết mạnh lên". Turing (1950) đặt nền cho câu hỏi về trí thông minh máy. Hội thảo Dartmouth 1956 đặt tên và mở ra lĩnh vực, cùng chương trình Logic Theorist của Newell và Simon chứng minh định lý toán học.

### Thời kỳ lạc quan và mùa đông thứ nhất (1956 - 1980)

Rosenblatt (1958) công bố perceptron: một nơ-ron có trọng số học được bằng quy tắc sửa lỗi, và có định lý hội tụ khi dữ liệu tách được tuyến tính [4]. ELIZA (1966) mô phỏng đối thoại bằng luật khớp mẫu. Kỳ vọng rất lớn, nhưng năm 1969 Minsky và Papert chỉ ra perceptron một tầng không biểu diễn được hàm XOR và nhiều hàm khác [5]; cùng với báo cáo Lighthill (1973) phê phán kết quả của AI, nguồn tài trợ sụt giảm. Đây là "mùa đông AI" đầu tiên. Ở mục 1.5 chúng ta sẽ tự tái hiện đúng hạn chế này bằng code.

### Hệ chuyên gia và sự trở lại của mạng nơ-ron (1980 - 1993)

Thập niên 1980 thuộc về hệ chuyên gia: MYCIN chẩn đoán nhiễm khuẩn, XCON cấu hình máy tính cho hãng DEC, đều dựa trên hàng nghìn luật if-then do chuyên gia cung cấp. Điểm yếu là chi phí viết và bảo trì luật. Cùng thời gian, Rumelhart, Hinton và Williams (1986) phổ biến thuật toán lan truyền ngược, cho phép huấn luyện mạng nhiều tầng và vượt qua giới hạn XOR [6]. Khi thị trường máy Lisp sụp đổ cuối thập niên 1980, mùa đông thứ hai bắt đầu.

### Học máy thống kê (1990 - 2006)

Trọng tâm chuyển sang các phương pháp có nền tảng toán học vững: máy vector hỗ trợ SVM của Cortes và Vapnik (1995) [7], cây quyết định và rừng ngẫu nhiên, mô hình Markov ẩn. Hai cột mốc của mạng nơ-ron vẫn được đặt ra trong giai đoạn này mà ít người chú ý: LSTM của Hochreiter và Schmidhuber (1997) giải quyết vấn đề gradient tiêu biến cho dữ liệu chuỗi [8], và LeNet-5 của LeCun và cộng sự (1998) đọc chữ số viết tay bằng mạng tích chập [9]. Năm 1997, Deep Blue thắng nhà vô địch cờ vua Kasparov, một thành tựu của tìm kiếm và luật chứ không phải của học.

### Học sâu (2006 - nay)

Ba điều kiện hội tụ: dữ liệu lớn (ImageNet, 2009) [10], sức mạnh tính toán của GPU, và các cải tiến thuật toán (ReLU, dropout, khởi tạo trọng số tốt). Năm 2012, AlexNet đạt sai số top-5 khoảng 15% trên ImageNet, so với khoảng 26% của phương pháp tốt thứ hai [11] và mở ra làn sóng học sâu. Tiếp theo là ResNet với kết nối tắt cho phép huấn luyện mạng hàng trăm tầng (2015) [12], AlphaGo thắng kỳ thủ cờ vây chuyên nghiệp (2016), kiến trúc Transformer với cơ chế tự chú ý (2017) [13], và các mô hình ngôn ngữ lớn từ BERT (2018) đến GPT-3 (2020) và các trợ lý hội thoại. [Cùng năm 2020, Lewis và cộng sự đề xuất RAG: thay vì nhồi toàn bộ tri thức vào trọng số, mô hình truy hồi tài liệu liên quan rồi mới sinh câu trả lời [21], nhờ đó cập nhật được tri thức mà không phải huấn luyện lại và dẫn được nguồn. Hướng này tiếp tục được ghép với đồ thị tri thức để trả lời những câu hỏi mà không tài liệu đơn lẻ nào chứa đủ dữ kiện, tiêu biểu là GraphRAG của Edge và cộng sự (2024) [22]. Đây chính là mạch dẫn tới Chương 5, nơi ý tưởng đó được cài đặt và đo trên dữ liệu tài chính.]{custom-style="Moi"}

{{TABL t_timeline}}

## 1.3. Bài học xuyên suốt cho các chương sau

Nhìn lại, có ba quy luật lặp đi lặp lại. Thứ nhất, **mỗi bước tiến đi kèm một điều kiện kỹ thuật cụ thể**: lan truyền ngược cần thuật toán, học sâu cần dữ liệu và GPU. Thứ hai, **các ý tưởng thường có trước thời của chúng**: perceptron 1958, LSTM 1997, CNN 1998 đều phải chờ nhiều năm mới thành công. Thứ ba, **kỹ thuật mới không xóa kỹ thuật cũ**: trên dữ liệu bảng nhỏ, hồi quy logistic và rừng ngẫu nhiên vẫn thường ngang hoặc hơn mạng nơ-ron. Quy luật thứ ba sẽ được kiểm chứng bằng số liệu thật ở các Chương 2, 3 và 4[, và một lần nữa ở Chương 5, nơi việc thêm đồ thị tri thức vào truy hồi không cải thiện được miền dữ liệu vàng]{custom-style="Moi"}.

## 1.4. Dữ liệu minh họa của chương

Để phần lịch sử không chỉ là kể chuyện, chương này chọn ba bộ dữ liệu đại diện cho ba thời kỳ, và dùng chúng ở mục 1.5 để chạy lại các thế hệ phương pháp trên cùng một bài toán.

{{TABL t_c1_datasets}}

### Iris (1936): dữ liệu của thời thống kê

Bộ Iris do Fisher công bố năm 1936 để minh họa phân tích biệt thức tuyến tính [16]; ngày nay nó có mặt trong UCI Machine Learning Repository và trong scikit-learn. Mỗi hoa được đo bốn đặc trưng (chiều dài, chiều rộng của đài và cánh hoa), thuộc ba loài, mỗi loài đúng 50 mẫu.

{{TABL t_c1_iris_sample}}

{{FIGW c1_iris.png | 78 | Bộ Iris: hai cặp đặc trưng và phân bố lớp}}

**Nhận xét.** Bộ dữ liệu cân bằng hoàn hảo (50/50/50) nên accuracy là độ đo đủ tin cậy. Loài setosa tách hẳn khỏi hai loài còn lại ở cặp đặc trưng cánh hoa, còn versicolor và virginica chồng lấn một phần: đây là ví dụ kinh điển cho "tách được tuyến tính" và "không tách được hoàn toàn" mà mục 1.5 sẽ khai thác. Với chỉ 150 mẫu, bộ dữ liệu quá nhỏ để đánh giá mạng sâu, nên ở đây nó chỉ dùng để minh họa perceptron.

### MNIST (1998): dữ liệu của thời CNN đầu tiên

MNIST gồm 70.000 ảnh xám 28 × 28 chữ số viết tay, chia sẵn 60.000 ảnh huấn luyện và 10.000 ảnh kiểm tra, do LeCun và cộng sự tổng hợp từ dữ liệu của NIST. Chương này nạp bản sao trên OpenML (`mnist_784`), mỗi ảnh là một vectơ 784 điểm ảnh nhận giá trị 0 đến 255 và được chia cho 255.

{{FIGW c1_mnist_samples.png | 78 | Hai ảnh mẫu của mỗi chữ số trong MNIST}}

{{FIGW c1_mnist_dist.png | 45 | Phân bố 10 lớp của tập huấn luyện MNIST}}

**Nhận xét.** {{V mnist_comment()}}

### Diabetes (Kaggle): dữ liệu của thời học máy ứng dụng

Bộ Diabetes Prediction Dataset gồm 100.000 hồ sơ y tế với tám biến: giới tính, tuổi, tăng huyết áp, bệnh tim, tiền sử hút thuốc, BMI, nồng độ HbA1c và đường huyết, cùng nhãn có hay không bị tiểu đường. Đây cũng là bộ dữ liệu bảng đầu tiên của Chương 2.

{{TABL t_c1_diabetes_sample}}

{{FIGW c1_diabetes.png | 78 | Bộ Diabetes: nhãn mất cân bằng và phân bố HbA1c, tuổi theo nhãn}}

**Nhận xét.** {{V diabetes_comment()}}

## 1.5. Tái hiện lịch sử bằng thực nghiệm

### 1.5.1. Perceptron và bài toán XOR

Perceptron tính $\hat y = [\,\mathbf w^\top \mathbf x + b > 0\,]$ và cập nhật $\mathbf w \leftarrow \mathbf w + \eta\,(y - \hat y)\,\mathbf x$ mỗi khi dự đoán sai. Đoạn mã dưới đây là toàn bộ thuật toán 1958 viết bằng NumPy.

```python
class Perceptron:
    def __init__(s, n, lr=.1, seed=0):
        s.w = np.random.default_rng(seed).normal(0, .1, n); s.b = 0.; s.lr = lr
    def predict(s, X):
        return (X @ s.w + s.b > 0).astype(int)
    def fit(s, X, y, epochs=200):
        for _ in range(epochs):
            err = 0
            for xi, yi in zip(X, y):
                d = yi - (xi @ s.w + s.b > 0)       # 0 neu dung, +-1 neu sai
                s.w += s.lr * d * xi; s.b += s.lr * d; err += int(d != 0)
            if err == 0: break
```

{{TABL t_c1_xor}}

Với AND và OR perceptron hội tụ sau vài epoch và đúng 100%; với XOR nó không bao giờ hội tụ và chỉ đúng tối đa 2/4 điểm, vì không có đường thẳng nào tách hai điểm (0,0), (1,1) khỏi hai điểm (0,1), (1,0). Chỉ cần thêm một tầng ẩn bốn nơ-ron tanh và huấn luyện bằng lan truyền ngược, mạng 2-4-1 phân loại đúng cả bốn điểm. Thí nghiệm này là hình thu nhỏ của mùa đông thứ nhất và của lời giải năm 1986.

{{FIGW c1_xor.png | 78 | Perceptron một tầng không giải được XOR, mạng hai tầng giải được}}

Trên Iris, cùng thuật toán đó hội tụ chỉ sau {{V iris_eps()}} epoch cho bài setosa so với phần còn lại (đúng 100%), nhưng không hội tụ trong 200 epoch cho bài versicolor so với virginica và dừng ở độ chính xác {{V iris_acc2()}}, khớp với nhận xét ở mục 1.4 về vùng chồng lấn.

### 1.5.2. Các thế hệ phương pháp trên cùng bài toán MNIST

Mục này chạy năm phương pháp đại diện cho các thời kỳ, đều trên đúng tập huấn luyện 60.000 ảnh và đánh giá trên 10.000 ảnh kiểm tra của MNIST.

{{TAB t_c1_mnist}}

{{FIGW c1_mnist_eras.png | 78 | Sai số kiểm tra của các thế hệ phương pháp trên MNIST}}

{{V mnist_eras_comment()}}

### 1.5.3. Từ luật chuyên gia đến học từ dữ liệu trên bài chẩn đoán tiểu đường

Hệ chuyên gia thời 1980 sẽ mã hóa tri thức y khoa thành luật. Với tiểu đường, tiêu chuẩn chẩn đoán có sẵn: HbA1c từ 6,5% trở lên, hoặc đường huyết từ 200 mg/dL trở lên. Thí nghiệm dưới đây đặt luật đó cạnh các mô hình học từ dữ liệu, đánh giá trên cùng tập kiểm tra 19.031 hồ sơ (8,9% dương tính) mà Chương 2 sẽ dùng lại.

{{TAB t_c1_diab}}

{{V diab_eras_comment()}}

## 1.6. Demo: perceptron và XOR trong trình duyệt

Để người đọc tự thử thay vì chỉ xem hình, thí nghiệm ở mục 1.5.1 được đóng gói thành một trang web tĩnh (`demo/ch1_perceptron.html`, không cần máy chủ). Người dùng chọn AND, OR hoặc XOR rồi huấn luyện perceptron hay MLP 2-4-1; trang vẽ vùng quyết định ngay trên canvas. Thuật toán được viết lại bằng JavaScript, cùng thuật toán với mã Python ở trên, nên kết quả khớp: perceptron hội tụ với AND và OR, không hội tụ với XOR (đúng 2/4 điểm), MLP đúng 4/4.

{{FIGW demo_c1.png | 46 | Demo Chương 1: perceptron chỉ đạt 2/4 điểm trên XOR (trên), MLP 2-4-1 đạt 4/4 (dưới)}}

## 1.7. Tổng kết chương

Lịch sử AI là quá trình dịch chuyển tri thức từ đầu người viết luật sang tham số học từ dữ liệu, kèm theo ba bài học đã được kiểm chứng ở trên: giới hạn lý thuyết (XOR) có thể phá bằng cấu trúc (thêm tầng ẩn); cùng một bài toán, các thế hệ phương pháp cho kết quả khác nhau rõ rệt (MNIST); và một luật chuyên gia cứng, dù dựa trên tiêu chuẩn y khoa thật, vẫn thua mọi mô hình học từ dữ liệu khi nhãn có nhiều sắc thái hơn một ngưỡng duy nhất (Diabetes), nhưng khoảng cách giữa các mô hình học được thì nhỏ hơn nhiều. Các chương tiếp theo đi sâu vào ba dòng kỹ thuật học từ dữ liệu: mạng nhiều tầng (Chương 2), mạng tích chập (Chương 3) và mạng hồi quy (Chương 4)[, rồi Chương 5 quay lại đúng câu hỏi của mục 1.5.3 — tri thức nên nằm trong luật, trong trọng số hay trong một kho bên ngoài tra cứu được — lần này với đồ thị tri thức và truy hồi]{custom-style="Moi"}.
