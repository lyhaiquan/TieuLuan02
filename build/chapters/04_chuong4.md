# Chương 4. Mạng nơ-ron hồi quy (RNN)

## 4.1. Cơ sở lý thuyết

### 4.1.1. Dữ liệu chuỗi và trạng thái ẩn

Với dữ liệu có thứ tự (chuỗi giá, văn bản, tín hiệu), mô hình phải dùng thông tin ở các bước trước để hiểu bước hiện tại. Mạng hồi quy giải quyết bằng cách mang theo một **trạng thái ẩn** $h_t$, cập nhật ở mỗi bước bằng cùng một bộ trọng số:

$$ h_t = \tanh\!\big(W_{ih}\,x_t + W_{hh}\,h_{t-1} + b\big), \qquad \hat y = W_{hy}\,h_T + b_y $$

Số tham số không đổi dù chuỗi dài bao nhiêu, và mô hình xử lý được chuỗi độ dài tùy ý. Trong tiểu luận này mọi mạng đều theo kiểu "nhiều vào, một ra": đọc cả chuỗi rồi dự đoán từ trạng thái ẩn cuối $h_T$.

### 4.1.2. Lan truyền ngược theo thời gian và gradient tiêu biến

Huấn luyện RNN dùng lan truyền ngược theo thời gian (BPTT): trải mạng theo $T$ bước thành một mạng sâu $T$ tầng chia sẻ trọng số rồi lan truyền ngược như bình thường. Gradient của loss đối với trạng thái ở bước $t$ là tích của nhiều ma trận Jacobian:

$$ \frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \operatorname{diag}\big(1 - h_k^2\big)\,W_{hh} $$

Nếu bán kính phổ của $W_{hh}$ nhỏ hơn 1 tích này co lại theo cấp số nhân (gradient tiêu biến: mạng không học được phụ thuộc xa); nếu lớn hơn 1 nó bùng nổ, và được xử lý bằng cách cắt chuẩn gradient. Gradient tiêu biến thì không cắt được.

### 4.1.3. LSTM và GRU

LSTM [8] thêm một ô nhớ $c_t$ cập nhật cộng tính dưới sự điều khiển của ba cổng (vào, quên, ra):

$$ c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t) $$

Đạo hàm dọc theo đường ô nhớ chỉ là $\operatorname{diag}(f_t)$, không có $W_{hh}$ và không có đạo hàm tanh, nên khi cổng quên mở gần hết gradient đi qua được rất xa. GRU gộp hai cổng (đặt lại, cập nhật) và bỏ ô nhớ riêng. Với trạng thái ẩn $H$ và đầu vào $F$ chiều, một tầng có $H(F+H+2)$ tham số cho RNN, gấp 3 lần cho GRU và gấp 4 lần cho LSTM (PyTorch giữ hai vectơ bias). Bản NumPy ở mục 4.3 dùng cùng quy ước trọng số để đối chiếu từng con số.

**Một chi tiết quyết định trong thực hành.** Khả năng nhớ lâu của LSTM chỉ có khi cổng quên mở lúc khởi tạo. PyTorch khởi tạo mọi bias gần 0 nên $f \approx 0{,}5$; Keras đặt bias cổng quên bằng 1 nên $f \approx 0{,}73$. Thí nghiệm trí nhớ ở mục 4.5 đo hệ quả của khác biệt nhỏ này.

## 4.2. Dữ liệu

Chương này dùng hai bộ dữ liệu, một bảng khách hàng và một chuỗi thời gian tài chính, cho ba bài toán.

{{TABL t_c4_datasets}}

### Churn Modelling

Mỗi dòng là một khách hàng của ngân hàng với 14 cột; ba cột định danh bị bỏ, hai cột chữ (quốc gia, giới tính) được mã hóa, còn 11 đặc trưng. Nhãn `Exited` cho biết khách đã rời ngân hàng. Dữ liệu chia 64/16/20 phân tầng theo nhãn (6.400 huấn luyện, 1.600 validation, 2.000 kiểm tra), chuẩn hóa bằng thống kê của tập huấn luyện. Vì dữ liệu là một bảng không có trục thời gian, để đưa vào RNN ta coi 11 đặc trưng là một chuỗi 11 bước, mỗi bước một đặc trưng, theo thứ tự cố định: nhân khẩu học, tài chính, rồi quan hệ với ngân hàng.

{{TABL t_c4_churn_sample}}

{{FIG fig_churn_corr.png | Ma trận tương quan của 11 đặc trưng và nhãn Churn}}

**Nhận xét.** {{V c4_churn_comment()}}

### Amazon (AMZN)

Chuỗi giá hằng ngày của cổ phiếu Amazon từ {{V J('ch4/ch4_stats.json')['amzn']['start']}} đến {{V J('ch4/ch4_stats.json')['amzn']['end']}}: {{V n(J('ch4/ch4_stats.json')['amzn']['shape'][0])}} phiên, bảy cột (giá mở, cao, thấp, đóng, đóng điều chỉnh, khối lượng). Từ đó dựng ba đặc trưng không phụ thuộc mức giá (lợi suất log, biên độ trong ngày, khối lượng tương đối) để dùng được cho cả năm 1997 lẫn 2023, và cửa sổ đầu vào 60 phiên. Hai bài toán: **lợi suất** của phiên kế tiếp (bài toán 1) và **biến động thực** 5 phiên tới (bài toán 2, căn bậc hai trung bình bình phương lợi suất). Chia theo thời gian: huấn luyện đến 2018, validation 2019 - 2020, kiểm tra 2021 - 2023, không xáo trộn, và mỗi cửa sổ được xếp vào tập theo ngày cuối cùng mà mục tiêu chạm tới nên không có rò rỉ thông tin tương lai.

{{TABL t_c4_amzn_sample}}

{{FIG fig_amzn_price.png | Giá đóng cửa điều chỉnh của AMZN (trên: thang tuyến tính, dưới: thang log) và ba giai đoạn huấn luyện, validation, kiểm tra}}

{{FIG fig_amzn_returns.png | Lợi suất log hằng ngày: phân phối đuôi dày so với phân phối chuẩn}}

**Nhận xét.** {{V c4_amzn_comment()}}

## 4.3. Ba cách cài đặt

Mọi cài đặt dùng cùng dữ liệu, cùng chia tập, cùng siêu tham số (Churn: 32 đơn vị ẩn, Adam 3e-3, batch 64, tối đa 60 epoch; AMZN: Adam 1e-3, batch 64, tối đa 40 epoch, dropout 0,2), cùng cắt chuẩn gradient 1,0, cùng dừng sớm theo loss validation, và ba hạt giống 42, 43, 44.

### 4.3.1. Scratch: NumPy thuần

Phần mới của tiểu luận so với các bài thực hành trước: một thư viện RNN, GRU, LSTM tự viết hoàn chỉnh, gồm lan truyền tiến, BPTT, Adam, cắt gradient và dừng sớm. Trọng số bố trí giống hệt `torch.nn` (`W_ih`, `W_hh`, hai vectơ bias, các cổng xếp theo thứ tự r-z-n cho GRU và i-f-g-o cho LSTM). Nhờ vậy có thể nạp cùng trọng số vào PyTorch và so sánh từng số. Đoạn trích dưới đây là phần tính ô LSTM: lan truyền tiến lưu bộ nhớ đệm, lan truyền ngược chạy từ bước cuối về bước đầu.

{{CODEL ch4/rnn_numpy.py | g_all = XW[:, t] + gh | cache.append((h, c, i, f, g, o, tc))}}

Phần lan truyền ngược của nhánh LSTM:

{{CODEL ch4/rnn_numpy.py | h_prev, c_prev, i, f, g, o, tc = cache[t] | dh = dgh @ p["W_hh"]}}

**Kiểm chứng.** Trước khi huấn luyện, bản NumPy được đối chiếu với `torch.nn` trên cùng trọng số: chênh lệch tối đa của đầu ra tiến là {{V '%.0e' % max(J('ch4/results/numpy_verify.json')[k]['fwd_diff'] for k in ('RNN','GRU','LSTM'))}}, của gradient là {{V '%.0e' % max(J('ch4/results/numpy_verify.json')[k]['grad_diff'] for k in ('RNN','GRU','LSTM'))}}, và sai số tương đối so với sai phân hữu hạn tối đa {{V '%.0e' % max(J('ch4/results/numpy_verify.json')[k]['fd_rel_err'] for k in ('RNN','GRU','LSTM'))}}, cả ba cho RNN, GRU và LSTM. Nói cách khác BPTT tự viết là đúng, và mọi khác biệt kết quả ở các bảng sau không đến từ lỗi tính gradient.

### 4.3.2. PyTorch

`nn.RNN`, `nn.LSTM`, `nn.GRU` thay toàn bộ phần tính ô và BPTT bằng autograd; vòng lặp huấn luyện vẫn viết tay (Adam, `clip_grad_norm_`, dừng sớm).

{{CODE ch4/src/nb03_rnn_pytorch.py | SeqNet}}

### 4.3.3. Keras

Keras dùng `layers.SimpleRNN`, `layers.LSTM`, `layers.GRU` và `fit()` với `global_clipnorm=1.0` (để có cùng phép cắt chuẩn toàn cục như PyTorch) và callback dừng sớm.

{{CODE ch4/src/nb04_rnn_keras.py | seq_model}}

## 4.4. Đánh giá ba cài đặt trên ba bài toán

Mỗi ô là trung bình và độ lệch chuẩn trên ba hạt giống. Churn dùng ROC-AUC (không phụ thuộc ngưỡng) và F1 tại ngưỡng chọn trên tập validation; dự báo lợi suất dùng R² so với dự báo "lợi suất bằng 0" (naive, tức giá mai bằng giá hôm nay), nên R² âm nghĩa là tệ hơn naive; dự báo biến động dùng R² thông thường.

### 4.4.1. Churn: phân lớp khách hàng rời bỏ

{{TAB t_c4_churn}}

{{V c4_churn_result()}}

### 4.4.2. AMZN bài toán 1: dự báo lợi suất

{{TAB t_c4_ret}}

{{V c4_ret_result()}}

### 4.4.3. AMZN bài toán 2: dự báo biến động 5 phiên tới

{{TAB t_c4_vol}}

{{FIG fig_torch_vol_test.png | Biến động 5 phiên tới trên tập kiểm tra: thực tế, đường cơ sở 20 phiên và LSTM (PyTorch)}}

{{V c4_vol_result()}}

### 4.4.4. Thời gian và độ dài code

{{TAB t_c4_time}}

{{V c4_time_comment()}}

## 4.5. Phân tích thêm

### Thí nghiệm trí nhớ: vì sao khởi tạo quyết định LSTM

Dữ liệu đồ chơi: một dấu hiệu ±1 xuất hiện ở bước 0, theo sau là $T-1$ bước nhiễu, và mạng phải nhớ dấu hiệu đó để phân loại. Năm cấu hình (RNN, LSTM với bias cổng quên khoảng 0 như PyTorch, bằng 1 như Keras, bằng 3, và GRU), ba độ dài $T$, hai hạt giống.

{{TAB t_c4_memory}}

{{FIG fig_memory.png | Độ chính xác khi phải nhớ dấu hiệu ở bước 0 qua T - 1 bước nhiễu}}

Kết quả đảo ngược trực giác "LSTM luôn nhớ lâu hơn RNN": với khởi tạo mặc định của PyTorch, LSTM thất bại ngay từ $T = 25$ (độ chính xác 0,500, tức đoán ngẫu nhiên), còn kém cả RNN thuần; chỉ cần đặt bias cổng quên bằng 1 (mặc định của Keras) là giải được $T = 25$, và bằng 3 thì giải được cả $T = 50$. Thứ tự này trùng với lượng gradient về tới bước đầu đo lúc khởi tạo trong bài thực hành RNN (Lab 06). Đây là một khác biệt thật giữa hai framework mà người dùng không thấy nếu không đọc tài liệu: cùng một dòng `LSTM(32)` cho hai mô hình có trí nhớ ban đầu khác nhau.

### Cái bẫy "dự báo giá trông rất đẹp"

Cách làm hay gặp trong các hướng dẫn trên mạng là đưa thẳng giá đóng cửa vào LSTM và dự báo giá ngày mai. Đồ thị dự báo trông gần như trùng với giá thật. Nhưng khi đo, sai số RMSE của LSTM trên mức giá là {{V f(J('ch4/results/torch__extra.json')['price_trap']['lstm_price_rmse'],2)}} USD, gấp {{V f(J('ch4/results/torch__extra.json')['price_trap']['lstm_price_rmse']/J('ch4/results/torch__extra.json')['price_trap']['naive_price_rmse'],1)}} lần dự báo naive "giá mai bằng giá hôm nay" ({{V f(J('ch4/results/torch__extra.json')['price_trap']['naive_price_rmse'],2)}} USD). Đường dự báo trông đẹp vì mô hình chỉ chép lại giá hôm qua, và còn bị chặn trên (mức dự báo cao nhất {{V f(J('ch4/results/torch__extra.json')['price_trap']['pred_max'],0)}} USD khi giá thật lên tới {{V f(J('ch4/results/torch__extra.json')['price_trap']['test_max'],0)}} USD) do chuẩn hóa theo tập huấn luyện. Bài học: không bao giờ tin đồ thị dự báo giá nếu chưa so với naive.

{{FIGW fig_torch_price_trap.png | 85 | LSTM học trên mức giá: dự báo trên tập kiểm tra và mức độ giống giá của k phiên trước}}

## 4.6. Demo: dự báo biến động AMZN bằng GRU chạy trong trình duyệt

Mô hình GRU 32 đơn vị ẩn của mục 4.3.1 (NumPy thuần, lan truyền ngược theo thời gian tự viết, hạt giống 42) được xuất cùng 736 cửa sổ kiểm tra. Trang `demo/ch4_volatility.html` cài đặt lại phép tính GRU bằng JavaScript theo đúng công thức cổng đặt lại, cổng cập nhật và ứng viên của PyTorch, đọc cửa sổ 60 phiên đã chọn và trả về biến động 5 phiên tới. Người dùng kéo thanh trượt để chọn phiên (2021 - 2023) và so ba con số: dự báo của GRU, giá trị thực và đường cơ sở "biến động 20 phiên qua". Thanh trượt thứ hai là một kịch bản giả định: nhân độ lớn lợi suất và biên độ của 5 phiên cuối với một hệ số để xem dự báo phản ứng thế nào khi thị trường đột ngột biến động mạnh hay yên ắng.

Kiểm chứng: ở phiên 21/05/2021 trang cho dự báo 1,38%, trùng với đầu ra NumPy (1,38%). Trên cả 736 phiên, mô hình này có R² = 0,288, ngang hồi quy Ridge ở mục 4.4.3. Cần nhớ kết luận của chương: đây là dự báo độ lớn của biến động, bài toán dự báo được, còn chiều của giá thì không; trang chỉ phục vụ minh họa học thuật.

{{FIGW demo_c4.png | 80 | Demo Chương 4: chọn phiên kiểm tra, xem dự báo của GRU, giá trị thực và đường cơ sở}}

## 4.7. So sánh và nhận xét

**Ba cài đặt cho cùng kết luận định tính, nhưng không cho cùng con số.** Trên mọi bài toán, ba cài đặt dẫn đến cùng các kết luận: mạng vượt hồi quy logistic trên Churn, không hơn dự báo naive trên lợi suất, ngang hồi quy Ridge trên biến động. Về con số, NumPy và PyTorch (cùng quy ước khởi tạo) sát nhau, thường trong khoảng một lần độ lệch chuẩn giữa hạt giống. Keras thì lệch xa hơn ở một số ô (LSTM trên Churn khoảng ba lần độ lệch chuẩn, GRU và LSTM trên biến động ba đến năm lần, theo hai chiều ngược nhau), nghĩa là khác biệt giữa framework ở đó là thật, không phải nhiễu hạt giống. Bài toán lợi suất thì mọi khác biệt đều nằm trong nhiễu. Nguyên nhân khả dĩ là mặc định khởi tạo khác nhau của Keras và PyTorch (mục 4.5); tôi chưa kiểm chứng riêng. Việc bản scratch tự viết, với gradient đã kiểm chứng với `torch.nn`, cho kết quả sát PyTorch là bằng chứng mạnh rằng các kết luận không do lỗi cài đặt.

**RNN không tự động thắng đối chứng đơn giản.** Trên Churn, MLP đọc cả 11 đặc trưng cùng lúc ngang hoặc nhỉnh hơn mọi ô hồi quy, vì dữ liệu không có trục thời gian. Trên lợi suất AMZN, không mô hình nào hơn dự báo naive (R² so với naive đều âm hoặc xấp xỉ 0), điều khớp với giả thuyết thị trường hiệu quả ở dạng yếu. Trên biến động, cả ba ô đạt R² khoảng 0,24 đến 0,28, ngang hồi quy Ridge trên cùng cửa sổ 60 ngày. RNN có lợi thế thật khi phụ thuộc theo thời gian vừa dài vừa phi tuyến và có nhiều chuỗi để học; hai bộ dữ liệu này không thuộc loại đó. Nhờ có đối chứng, các con số như ROC-AUC 0,85 hay R² 0,28 không bị đọc nhầm thành thành công của RNN.

**Mức kiểm soát và chi phí.** NumPy cho kiểm soát tuyệt đối (thấy từng Jacobian) nhưng chậm hơn rõ và dài hơn nhiều dòng; PyTorch cân bằng giữa linh hoạt và tốc độ; Keras ngắn nhất nhưng ẩn nhiều mặc định (bias cổng quên, cách cắt gradient) có thể đổi hành vi mô hình. Cả ba phục vụ những mục đích khác nhau: hiểu (scratch), nghiên cứu (PyTorch), triển khai nhanh (Keras).

**Giới hạn.** Churn không phải chuỗi thời gian thật; tập kiểm tra AMZN là một giai đoạn ba năm duy nhất (cần kiểm định trượt để kết luận chặt hơn); mô hình nhỏ (một tầng, 32 đơn vị ẩn) và chỉ ba hạt giống; một mã cổ phiếu. Tiếp theo, Chương 5 trở đi chuyển sang tri thức có cấu trúc, nơi các mô hình chuỗi và ngôn ngữ được kết hợp với đồ thị tri thức.
