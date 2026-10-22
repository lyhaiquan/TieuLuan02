---
title: "TIỂU LUẬN MÔN HỌC: PHÁT TRIỂN CÁC HỆ THỐNG THÔNG MINH"
subtitle: "Từ lịch sử trí tuệ nhân tạo đến học máy, mạng tích chập và mạng hồi quy: so sánh ba cách cài đặt NumPy, PyTorch và Keras"
author: "AUTHOR_LINE"
date: "Học viện Công nghệ Bưu chính Viễn thông - Giảng viên: PGS.TS Trần Đình Quế"
lang: vi
toc-title: "Mục lục"
---

# Mở đầu

## Lý do chọn đề tài

Trí tuệ nhân tạo (AI) hiện diện trong gần như mọi hệ thống phần mềm: từ chẩn đoán y khoa, định giá bất động sản đến phân tích cảm xúc của người dùng mạng xã hội. Tuy vậy, người học thường tiếp cận từng kỹ thuật như một "hộp đen" qua một thư viện có sẵn, nên khó trả lời ba câu hỏi nền tảng: kỹ thuật này sinh ra để giải quyết vấn đề gì, bên trong nó tính toán những gì, và khi đổi từ thư viện này sang thư viện khác thì kết quả có thay đổi không.

Tiểu luận này trả lời ba câu hỏi đó cho bốn mảng kiến thức đầu tiên của môn học: lịch sử phát triển của AI (Chương 1), các kỹ thuật học máy cơ bản và mạng nơ-ron nhiều tầng (Chương 2), mạng tích chập CNN (Chương 3), mạng hồi quy RNN (Chương 4) [và Knowledge Graph kết hợp RAG cho miền tài chính (Chương 5)]{custom-style="Moi"}. Ba chương kỹ thuật cùng theo một khuôn: trình bày cơ sở lý thuyết ngắn gọn, mô tả dữ liệu, cài đặt cùng một mô hình theo ba cách rồi đánh giá và so sánh.

## Mục tiêu

1. Hệ thống hóa các mốc lịch sử chính của AI và chỉ ra mỗi giai đoạn để lại kỹ thuật nào cho các chương sau (Chương 1).
2. Với mỗi chương, giới thiệu từ hai đến ba bộ dữ liệu: nguồn, kích thước, mẫu dữ liệu, phân bố và nhận xét.
3. Với Chương 2, 3 và 4, cài đặt mỗi mô hình theo ba cách: **tự viết bằng NumPy (scratch)**, **PyTorch** và **Keras**, dùng cùng dữ liệu, cùng chia tập, cùng siêu tham số.
4. Đánh giá ba cài đặt trên hai đến ba bộ dữ liệu của từng chương, rồi so sánh về chất lượng, thời gian, số dòng code và mức độ kiểm soát.

## Đối tượng nghiên cứu

Đối tượng là các kỹ thuật học có giám sát, tức là học một ánh xạ từ đầu vào sang nhãn hoặc giá trị từ các ví dụ đã gán nhãn. Trên trục thời gian, chúng nằm trên một dòng phát triển liền mạch: mô hình thống kê và học máy cổ điển (đến thập niên 2000), mạng nơ-ron nhiều tầng huấn luyện bằng lan truyền ngược, rồi các kiến trúc khai thác cấu trúc của dữ liệu: cấu trúc không gian bằng CNN và cấu trúc thứ tự bằng RNN. Mỗi chương kỹ thuật chọn dữ liệu sao cho kiến trúc có "đất diễn": văn bản và bảng số cho Conv1D, chuỗi giá cho mạng hồi quy, và đặt kết quả cạnh một đường cơ sở đơn giản để biết mô hình học được gì thật sự.

## Phương pháp so sánh công bằng

Điểm khác biệt của tiểu luận so với việc chạy ba thư viện rồi so số là cách giữ cho phép so sánh sạch. Cụ thể:

- **Cùng dữ liệu, cùng chia tập.** Tiền xử lý nằm trong một mô-đun dùng chung, và ba cài đặt đọc cùng các mảng $X$, $y$ đã chia và chuẩn hóa (chỉ `fit` trên tập huấn luyện).
- **Cùng mô hình.** Cùng kiến trúc, cùng cách khởi tạo (He), cùng tối ưu, cùng tốc độ học, cùng số epoch và kích thước lô. Số tham số được in ra để kiểm tra ba bên giống nhau.
- **Cài đặt scratch được kiểm chứng.** Trước khi huấn luyện, gradient viết tay được so với sai phân hữu hạn và, ở Chương 4, với `torch.nn` trên cùng trọng số. Nhờ vậy khác biệt kết quả không thể đổ cho lỗi tính gradient.
- **Lặp nhiều hạt giống.** Ở Chương 2 và 4 mỗi cấu hình chạy ba hạt giống và báo cáo trung bình ± độ lệch chuẩn; chênh lệch nhỏ hơn độ lệch chuẩn không được diễn giải thành thứ hạng.
- **Đường cơ sở đơn giản.** Mỗi bài toán có ít nhất một đối chứng không dùng mạng sâu (hồi quy, rừng ngẫu nhiên, dự báo naive, hồi quy Ridge).
- **Đo thời gian tuần tự.** Thời gian chỉ được so sánh khi các cài đặt chạy lần lượt trên máy rảnh, và được ghi rõ khi không đảm bảo điều đó.

## Cấu trúc tiểu luận

**Chương 1** kể lại lịch sử AI theo sáu giai đoạn rồi tái hiện ba điểm ngoặt bằng thực nghiệm: perceptron và bài toán XOR, các thế hệ phương pháp trên MNIST, và luật chuyên gia đối chọi với học máy trên bài chẩn đoán tiểu đường. **Chương 2** trình bày các mô hình học máy cơ bản và mạng nhiều tầng, cài MLP bằng NumPy, PyTorch và Keras trên ba bài toán nhị phân, hồi quy và đa lớp. **Chương 3** giới thiệu CNN, cài Conv1D theo ba cách trên dữ liệu bảng và văn bản, và mở rộng sang ResNet, Inception, DenseNet trên ảnh. **Chương 4** trình bày RNN, LSTM, GRU, cài cả ba bằng NumPy (có BPTT tự viết), PyTorch và Keras trên dữ liệu khách hàng và giá cổ phiếu, kèm thí nghiệm giải thích vì sao khởi tạo quyết định khả năng nhớ của LSTM. [**Chương 5** chuyển sang một loại bài toán khác: thay vì dự đoán một con số, hệ phải trả lời câu hỏi bằng ngôn ngữ tự nhiên dựa trên tri thức nằm trong tài liệu và bảng số. Chương xây một hệ hỏi đáp cho miền vàng và chứng khoán với bốn thay đổi so với RAG thông thường: định tuyến câu hỏi số sang bộ tính thay vì truy hồi, cạnh đồ thị phân loại theo nguồn gốc, lọc thời gian trước khi truy hồi, và dùng đồ thị để kiểm chứng khẳng định nhân quả của báo chí thay vì chỉ để mở rộng truy hồi.]{custom-style="Moi"} Mỗi chương đều có một mục demo: trang web tĩnh chạy hoàn toàn trong trình duyệt (thư mục `demo/`), dùng đúng trọng số của mô hình đã huấn luyện, để người đọc tự thử mô hình thay vì chỉ đọc bảng số.

## Phạm vi và cách làm

Tiểu luận gồm [năm]{custom-style="Moi"} chương và phần mở đầu này; [hai]{custom-style="Moi"} chương còn lại của đề cương ([Deep Graph, hệ đa tác tử]{custom-style="Moi"}) thuộc đợt sau. [Phần in màu đỏ trong tiểu luận là nội dung mới bổ sung so với Tiểu luận 01.]{custom-style="Moi"} Toàn bộ thực nghiệm chạy trên một máy tính xách tay chỉ có CPU (Intel Core i5-12500H, 16 GB RAM), Python 3.14 cho NumPy và PyTorch 2.10, Python 3.13 với TensorFlow 2.21 và Keras 3 cho phần Keras. Mọi con số trong văn bản đều được sinh ra từ các tệp kết quả JSON của lần chạy cuối, không nhập tay.

Bảng dưới đây tóm tắt dữ liệu và mô hình của từng chương. Một chi tiết cần nói trước: các chương dùng chung ba tên dữ liệu (Diabetes, House Price, Comments) ở Chương 2 và 3, nhưng mỗi chương có quy trình làm sạch và cỡ mẫu riêng vì được thực hiện ở các bài tập khác nhau. Vì vậy số liệu giữa Chương 2 và Chương 3 không so sánh trực tiếp với nhau, chỉ so sánh trong nội bộ từng chương.

{{TABL t_overview}}

Quy ước trình bày: số thập phân dùng dấu phẩy; độ lệch chuẩn ghi sau dấu ±, tính trên ba hạt giống ngẫu nhiên 42, 43, 44 ở những chỗ có chạy lặp, và ghi rõ là "một hạt giống" ở những chỗ chỉ chạy một lần. Mã nguồn đầy đủ nằm trong các thư mục `ch1` đến `ch4` đi kèm; trong văn bản chỉ trích những đoạn then chốt.
