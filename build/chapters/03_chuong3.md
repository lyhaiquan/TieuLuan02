# Chương 3. Mạng nơ-ron tích chập (CNN)

## 3.1. Cơ sở lý thuyết

### 3.1.1. Vì sao cần convolution

Một lớp kết nối đầy đủ (dense) nối mọi đầu vào với mọi nơ-ron nên số tham số tăng theo tích hai kích thước: ảnh 28 × 28 nối với 128 nơ-ron đã cần 100.480 tham số, và lớp đó không biết gì về việc hai điểm ảnh cạnh nhau có liên quan. Mạng tích chập dựa trên hai giả định về cấu trúc dữ liệu: **tính cục bộ** (đặc trưng hữu ích nằm trong một vùng nhỏ) và **tính bất biến dịch chuyển** (cùng một mẫu có thể xuất hiện ở vị trí bất kỳ). Hai giả định này dẫn đến một bộ lọc nhỏ được trượt trên toàn đầu vào và chia sẻ trọng số ở mọi vị trí [9].

### 3.1.2. Convolution một chiều

Với đầu vào $x$ có $C_{in}$ kênh và độ dài $L$, một lớp Conv1D với $C_{out}$ bộ lọc kích thước $k$ cho đầu ra

$$ y_{o}[i] \;=\; b_o + \sum_{c=1}^{C_{in}} \sum_{j=0}^{k-1} W_{o,c,j}\; x_{c}[\,s\,i + j\,] ,\qquad L_{out} = \left\lfloor \frac{L + 2p - k}{s} \right\rfloor + 1 $$

trong đó $s$ là bước nhảy (stride) và $p$ là độ đệm (padding). Số tham số là $C_{out}(C_{in}k + 1)$, không phụ thuộc $L$: đó là điều làm CNN gọn hơn mạng dense. Sau convolution thường là hàm kích hoạt ReLU, rồi **pooling** (lấy max trên cửa sổ) để giảm độ dài và tăng tính bất biến cục bộ. Với văn bản, một lớp embedding biến mỗi token thành vectơ, và Conv1D với $k = 3$ hoạt động như bộ dò cụm ba từ (n-gram) học được.

### 3.1.3. Từ LeNet đến các kiến trúc cải tiến

LeNet-5 (1998) xếp xen kẽ convolution và pooling rồi dense [9]. AlexNet (2012) dùng ReLU, dropout và GPU [11]. VGG (2014) chỉ dùng bộ lọc 3 × 3 chồng sâu. Khi mạng sâu hơn, độ chính xác lại giảm vì khó tối ưu (gradient tiêu biến, hiện tượng suy giảm). Ba họ kiến trúc giải quyết điều đó: **ResNet** cộng đầu vào vào đầu ra của khối, $y = x + F(x)$, nên gradient có đường đi thẳng [12]; **Inception** chạy nhiều bộ lọc kích thước khác nhau song song rồi ghép kênh, dùng bộ lọc 1 × 1 làm nút cổ chai [14]; **DenseNet** nối (concatenate) đầu ra mọi tầng trước vào đầu vào tầng sau [15]. Mục 3.5 đo thực nghiệm cả ba.

## 3.2. Dữ liệu

Chương này dùng ba bộ dữ liệu cho phần chính (hai bảng số và một văn bản, cả ba đều xử lý được bằng Conv1D) và hai bộ ảnh cho phần mở rộng ở mục 3.5. Quy trình tiền xử lý của chương này khác với Chương 2 (cỡ mẫu khác, đặc trưng khác), nên số liệu hai chương không so sánh trực tiếp với nhau.

{{TABL t_c3_datasets}}

{{FIG c3_datasets.png | Phân bố nhãn của ba bộ dữ liệu Chương 3 (tập huấn luyện)}}

### Diabetes

Cùng nguồn Kaggle với Chương 1, bỏ 18 hồ sơ giới tính "Other" rồi lấy mẫu 20.000 hồ sơ (16.000 huấn luyện, 4.000 kiểm tra, chia phân tầng, seed 42) để bản NumPy huấn luyện trong thời gian hợp lý. Tám đặc trưng được coi như một chuỗi độ dài 8 với một kênh, để Conv1D trượt trên chuỗi đó. Bản chất của dữ liệu là bảng chứ không phải chuỗi có thứ tự, nên đây là điểm cần cẩn trọng khi đọc kết quả ở mục 3.4.

{{TABL t_c1_diabetes_sample}}

**Nhận xét.** Nhãn dương chiếm {{V pc(J('ch3/ch3_stats.json')['diabetes']['pos_train'],1)}} ở cả tập huấn luyện lẫn kiểm tra, nên accuracy cao là chưa đủ để kết luận: luôn đoán âm đã đạt khoảng {{V pc(1-J('ch3/ch3_stats.json')['diabetes']['pos_train'],1)}}.

### House Price

Tệp `vietnam_housing_dataset.csv` gồm 30.229 tin rao bán nhà (12 cột), giá tính bằng tỷ đồng. Sau khi bỏ giá không dương và diện tích ngoài [10, 1.000] m² còn 30.223 tin (24.178 huấn luyện, 6.045 kiểm tra). Mục tiêu là $\ln(1 + \text{giá})$ rồi chuẩn hóa. Các cột số thiếu (mặt tiền thiếu 38%, đường vào thiếu 44%) được điền trung vị; hướng nhà thiếu 70% nên chỉ giữ cờ "có ghi hướng". Địa chỉ được tách tỉnh/thành, one-hot cho tám nơi nhiều tin nhất và gộp phần còn lại, cho tổng 19 đặc trưng.

{{TABL t_c3_housing_sample}}

**Nhận xét.** Giá trung vị {{V f(J('ch3/ch3_stats.json')['housing']['price_median'],1)}} tỷ, trung bình {{V f(J('ch3/ch3_stats.json')['housing']['price_mean'],1)}} tỷ, phân vị 95% là {{V f(J('ch3/ch3_stats.json')['housing']['price_p95'],1)}} tỷ: phân bố lệch phải, đó là lý do huấn luyện trên thang logarit. Dữ liệu không có tọa độ, chỉ có địa chỉ dạng chữ, nên thông tin vị trí (yếu tố quyết định giá nhà) chỉ vào mô hình qua tỉnh/thành. Giới hạn này sẽ giải thích mức $R^2$ khiêm tốn ở mục 3.4.

### Comments

Tệp `YoutubeCommentsDataSet.csv` có 18.408 bình luận YouTube gắn nhãn negative, neutral, positive. Sau khi bỏ dòng thiếu nội dung hoặc nhãn còn 18.364 bình luận (14.692 huấn luyện, 3.672 kiểm tra). Mỗi bình luận được tách thành token chữ thường, từ điển 15.000 từ chỉ xây trên tập huấn luyện, độ dài cắt hoặc đệm về 60 token.

{{TABL t_c3_comments_sample}}

**Nhận xét.** Ba lớp lệch mạnh: positive chiếm {{V pc(J('ch3/ch3_stats.json')['comments']['counts'][2]/sum(J('ch3/ch3_stats.json')['comments']['counts']),0)}}, neutral {{V pc(J('ch3/ch3_stats.json')['comments']['counts'][1]/sum(J('ch3/ch3_stats.json')['comments']['counts']),0)}}, negative chỉ {{V pc(J('ch3/ch3_stats.json')['comments']['counts'][0]/sum(J('ch3/ch3_stats.json')['comments']['counts']),0)}}. Do đó macro-F1 mới là độ đo công bằng giữa các lớp, và F1 của lớp negative là chỉ số khó nhất. Bình luận ngắn (trung vị {{V f(J('ch3/ch3_stats.json')['comments']['len_median'],0)}} từ) và nhiều bình luận không phải tiếng Anh, nên bộ lọc cục bộ 3 token phù hợp hơn mô hình cần ngữ cảnh dài.

## 3.3. Mô hình và ba cách cài đặt

Hai kiến trúc được dùng, giữ nguyên giữa ba cài đặt. **CNN bảng** (Diabetes, House Price): Conv1D(16, k=3) → ReLU → Conv1D(16, k=3) → ReLU → MaxPool(2) → Flatten → Dense (riêng House Price thêm một tầng Dense 32 nơ-ron + ReLU trước tầng ra; đầu ra 1: sigmoid + BCE cho phân loại, tuyến tính + MSE cho hồi quy). **CNN văn bản**: Embedding(15.000, 32) → Conv1D(64, k=3) → ReLU → MaxPool(2) → Conv1D(64, k=3) → ReLU → GlobalMaxPool → Dropout(0,3) → Dense(3) với softmax + cross-entropy. Cả ba cài đặt dùng cùng khởi tạo He, cùng tối ưu Adam và cùng số epoch (Diabetes 30, House Price 40, Comments 4), nên khác biệt còn lại là do framework.

### 3.3.1. Scratch: NumPy thuần

Lớp Conv1D tự viết dùng kỹ thuật im2col: gom mỗi cửa sổ trượt thành một cột rồi nhân ma trận bằng `einsum`. Lan truyền ngược tính $dW$ từ các cột đã lưu, và "rải" $dX$ về đúng vị trí đầu vào. Toàn bộ các lớp (Dense, Embedding, MaxPool, Dropout), hàm mất mát, tối ưu Adam cũng tự viết; gradient được kiểm tra bằng sai phân hữu hạn trước khi huấn luyện.

{{CODE ch3/src/nb01_scratch.py | Conv1D}}

### 3.3.2. PyTorch

Với `nn.Conv1d`, autograd tự tính gradient nên không còn hàm `backward`. Kiến trúc chỉ là khai báo; vòng lặp huấn luyện viết tay với `loss.backward()` và `optimizer.step()`.

{{CODE ch3/src/nb02_pytorch.py | TabularCNN,TextCNN}}

### 3.3.3. Keras

Keras dùng `layers.Conv1D` theo layout channels-last `(N, L, C)`, ngược với channels-first của PyTorch và NumPy. Việc huấn luyện rút gọn thành `compile()` và `fit()`.

{{CODE ch3/src/nb03_tensorflow.py | build_tabular_cnn,build_text_cnn}}

### 3.3.4. Kích thước tensor và số tham số

Bảng sau theo dõi độ dài chuỗi qua từng lớp và tính tay số tham số; kết quả khớp đúng với số mà ba cài đặt tự báo cáo, nên kiến trúc của ba bên thật sự giống nhau.

{{TABL t_c3_shapes}}

Điều đáng chú ý ở Comments: gần như toàn bộ tham số (480.000 trên 498.755, tức 96%) nằm ở lớp embedding, còn hai lớp Conv1D chỉ có chưa đầy 19.000. Với dữ liệu văn bản, khối lượng học nằm ở việc biểu diễn từ, không phải ở bộ lọc.

Để chắc rằng ba cài đặt thật sự tính cùng một phép toán, mỗi notebook đối chiếu `Conv1D` với hàm convolution viết bằng vòng lặp trên một ví dụ nhỏ; cả ba trùng khớp tới sai số làm tròn số thực. Số tham số của ba cài đặt giống hệt nhau ({{V n(J('ch3/results/numpy_diabetes.json')['n_params'])}} cho Diabetes, {{V n(J('ch3/results/numpy_housing.json')['n_params'])}} cho House Price, {{V n(J('ch3/results/numpy_comments.json')['n_params'])}} cho Comments), điều kiện cần để so sánh công bằng.

## 3.4. Đánh giá ba cài đặt trên ba bộ dữ liệu

Mỗi cấu hình chạy **một lần** với seed 42 (ghi chú này quan trọng: khác biệt dưới một điểm phần trăm nằm trong dao động của hạt giống và không nên đọc như thứ hạng).

{{TAB t_c3_diabetes}}

{{TAB t_c3_housing}}

{{TAB t_c3_comments}}

{{FIG fig_losscurves.png | Đường loss huấn luyện của ba cài đặt trên ba bộ dữ liệu}}

{{FIG fig_quality.png | So sánh chất lượng của ba cài đặt}}

{{V c3_results_comment()}}

### Thời gian và độ dài code

{{TAB t_c3_time}}

{{V c3_time_comment()}}

## 3.5. Mở rộng: kiến trúc cải tiến trên dữ liệu ảnh

Để kiểm chứng các ý tưởng ResNet, Inception và DenseNet ở mục 3.1.3, một bài thực hành riêng (Lab 05) đã cài đặt cả ba bằng PyTorch với số tham số xấp xỉ nhau (khoảng 0,28 triệu cho ảnh) so với một CNN cơ bản, trên hai bộ ảnh Fashion-MNIST (ảnh xám 28 × 28, 10 loại quần áo) và CIFAR-10 (ảnh màu 32 × 32, 10 loại vật thể) cùng bộ Diabetes. Mỗi bộ dùng 10.000 ảnh huấn luyện và 3.000 ảnh kiểm tra, 10 epoch, một hạt giống.

{{TABL t_c3_arch}}

Kết quả trái với kỳ vọng đơn giản "kiến trúc mới hơn thì tốt hơn": ở quy mô nhỏ này, mạng CNN cơ bản đạt cao nhất hoặc ngang tốt nhất trên cả ba bộ. Trên CIFAR-10, ResNet thấp hơn mạng cơ bản khoảng 4,5 điểm, Inception và DenseNet thấp hơn khoảng 2 điểm; trên Fashion-MNIST cả bốn ngang nhau trong phạm vi sai số. Lý do không phải kiến trúc kém mà là bài toán chưa đủ khó: các cải tiến ra đời để xử lý mạng rất sâu và dữ liệu rất lớn, còn mạng chỉ khoảng 13 tầng có BatchNorm thì vấn đề suy giảm chưa xuất hiện. Cái giá phải trả thì rõ ràng: với cùng số tham số, thời gian huấn luyện trên CIFAR-10 là 132,6 giây cho mạng cơ bản, 419,0 giây cho ResNet, 766,3 giây cho Inception và 1.322,3 giây cho DenseNet. Kết quả này cũng chỉ là một hạt giống và 10 epoch, nên đọc như xu hướng chứ không như xếp hạng.

## 3.6. Demo: phân loại cảm xúc bằng Conv1D chạy trong trình duyệt

Mạng văn bản của mục 3.3 (Embedding 15.000 × 32 → Conv1D 64 → MaxPool → Conv1D 64 → GlobalMaxPool → Dense 3, tức 498.755 tham số) được huấn luyện lại bằng PyTorch với hạt giống 42 và 4 epoch, rồi xuất sang JSON (khoảng 3 MB: từ điển, ma trận embedding, các bộ lọc). Trang `demo/ch3_sentiment.html` tách token bằng cùng biểu thức chính quy `[a-z]+`, ánh xạ sang chỉ số từ điển, đệm về 60 token, rồi tự tính embedding, hai phép convolution, pooling và softmax bằng JavaScript. Người dùng gõ một bình luận và thấy xác suất ba lớp, kèm danh sách token để biết từ nào ngoài từ điển (bị thay bằng `<unk>`).

Kiểm chứng: xác suất do JavaScript tính trùng với PyTorch trên câu mẫu. Mô hình dùng cho demo đạt accuracy {{V f(J('demo/data/ch3_acc.json')['acc'], 3)}} trên 3.672 bình luận kiểm tra, thấp hơn một chút so với bảng ở mục 3.4 vì vòng lặp huấn luyện của demo được viết gọn hơn (kích thước lô 64, tốc độ học 1,5e-3, không dừng sớm). Điều này không ảnh hưởng đến mục đích minh họa, nhưng nghĩa là con số của demo và con số ở mục 3.4 là hai lần huấn luyện khác nhau. Giới hạn đáng chú ý: mô hình chỉ biết tiếng Anh và tập huấn luyện lệch về lớp tích cực, nên câu ngắn hoặc ngoại ngữ thường bị đẩy về trung tính hoặc tích cực.

{{FIGW demo_c3.png | 80 | Demo Chương 3: nhập bình luận, xem xác suất ba lớp và các token}}

## 3.7. So sánh và nhận xét

**Chất lượng gần như đồng nhất.** Trên cả ba bộ dữ liệu, chênh lệch giữa NumPy, PyTorch và Keras nhỏ hơn dao động thông thường giữa hai hạt giống: ví dụ trên Comments accuracy chỉ trong khoảng {{V f(min(c3('accuracy')),3)}} đến {{V f(max(c3('accuracy')),3)}}. Điều này khẳng định framework không quyết định chất lượng; nó chỉ là cách viết cùng một phép toán. Những chênh lệch nhìn thấy được (ví dụ F1 lớp negative) đến từ thứ tự xáo trộn mẫu và khởi tạo ngẫu nhiên khác nhau giữa các thư viện, không phải từ thuật toán.

**Tốc độ là điểm khác biệt thật sự.** NumPy thuần chậm hơn rõ rệt ở dữ liệu lớn (Comments, 4 epoch), vì không có kernel tối ưu và đa luồng như PyTorch hay TensorFlow. Ở bài rất nhỏ (Diabetes, 881 tham số) khoảng cách gần như biến mất, vì chi phí cố định của framework lấn át phép tính.

**Giá trị của bản scratch.** Viết `Conv1D` bằng tay buộc phải hiểu im2col, cách gradient "rải" ngược về đầu vào và vì sao pooling chỉ truyền gradient đến vị trí cực đại. Kiểu hiểu này không cần khi dùng thư viện, nhưng cần khi gỡ lỗi hoặc thiết kế lớp mới.

**CNN trên dữ liệu bảng chưa chắc là lựa chọn tốt.** Các hàng "(đối chiếu)" ở mục 3.4 cho bằng chứng trực tiếp trên đúng các phép chia của chương này. {{V c3_baseline_comment()}} Nói cách khác, Conv1D có cơ sở cấu trúc rõ nhất ở văn bản, vì cụm từ liền kề thật sự mang nghĩa, nhưng ở đây ngay cả chỗ đó nó cũng chưa vượt được đường cơ sở đơn giản; ở dữ liệu bảng, tám hay mười chín đặc trưng không có thứ tự không gian để bộ lọc trượt khai thác, và kết quả CNN chủ yếu phản ánh khả năng phi tuyến chung của mạng nhiều tầng. Bài học chọn kiến trúc theo cấu trúc dữ liệu cũng là kết luận của mục 3.5 và là cầu nối sang Chương 4: khi dữ liệu có thứ tự thời gian, ta cần một kiến trúc nhớ được quá khứ.
