# Tiểu luận 02 — Phát triển các hệ thống thông minh

Lý Hải Quân. Nhóm lớp 05, nhóm tiểu luận 02.

Bản nộp: `tieuluan02_05_02_quanlh.docx` và `.pdf` (65 trang). **Phần in màu đỏ trong báo cáo
là nội dung mới bổ sung so với Tiểu luận 01.**

Repo: <https://github.com/lyhaiquan/TieuLuan02>

## Nội dung

Tiểu luận 01 gồm bốn chương: lịch sử AI, học máy cơ bản và MLP, CNN, RNN. Tiểu luận 02 giữ
nguyên bốn chương đó và thêm **Chương 5: Knowledge Graph và RAG cho miền tài chính**.

| Thư mục | Nội dung |
|---|---|
| `ch1/` | Thực nghiệm lịch sử AI: XOR, Iris, MNIST, Diabetes |
| `ch2/` | MLP ba cài đặt (NumPy, PyTorch, Keras) trên ba bài toán |
| `ch3/` | CNN 1 chiều và các kiến trúc cải tiến |
| `ch4/` | RNN, LSTM, GRU (có BPTT tự viết bằng NumPy) |
| `ch5/` | **Mới.** Hệ hỏi đáp tài chính có kiểm chứng |
| `demo/` | 5 demo chạy thẳng trong trình duyệt, mở `index.html` |
| `build/` | `chapters/*.md`, `tables*.py`, `build.py` (md → docx → pdf) |
| `figs/` | Hình dùng trong báo cáo |

## Chương 5 có gì mới

Bốn thay đổi so với một hệ RAG thông thường, mỗi cái đều được đo:

1. **Bảng số thì truy vấn, văn bản mới truy hồi.** Câu hỏi số được định tuyến sang bộ tính
   chạy thẳng trên chuỗi giá. Trên 360 câu hỏi sinh tự động từ bảng giá, cách cũ (biến bảng
   thành câu thống kê rồi truy hồi) đạt 0,283 còn cách này đạt 1,000. Chênh lệch nằm trọn ở
   những câu hỏi mà cách cũ không có gì để tìm: hỏi theo quý và hỏi tương quan.
2. **Cạnh đồ thị có kiểu, tách theo nguồn gốc.** `belongs_to` tra cứu được, `correlates_with`
   tính từ giá, `claims_cause` là lời nhà báo. Ba loại không được trộn.
3. **Thời gian là chiều bắt buộc.** Không lọc thời gian thì 66,7% tài liệu truy hồi được đến
   từ sau cửa sổ câu hỏi, tức rò rỉ nhìn trước. Cạnh đồ thị cũng cắt theo quý: cặp GOLD–SP500
   có hệ số toàn giai đoạn +0,122 nhưng biên độ theo quý là 0,950 và đổi dấu 5 lần.
4. **Đồ thị dùng để kiểm chứng, không chỉ để mở rộng.** Trích khẳng định nhân quả từ tiêu đề
   tin rồi đối chiếu với tương quan tính từ giá. Kết quả đáng chú ý nhất: 65,4% khẳng định
   viện tới nguyên nhân không có chuỗi giá (Fed, lạm phát, địa chính trị) nên không kiểm
   chứng được bằng dữ liệu thị trường.

Mục 5.6 của báo cáo ghi lại năm lỗi đã mắc trong quá trình làm và cách phát hiện chúng; cả
năm đều làm sai lệch kết quả theo hướng có lợi cho phương pháp mới.

## Chạy lại Chương 5

```bash
cd ch5
python run_claims.py          # trích và kiểm chứng khẳng định nhân quả
python run_kg.py              # đồ thị: cạnh tương quan theo quý, cạnh claims_cause
python run_numeric_eval.py    # so sánh hai cách trả lời câu hỏi số
python run_narrative_eval.py  # rò rỉ nhìn trước và độ liên quan
python run_stocks.py          # miền chứng khoán
python make_figs.py           # bốn hình của chương
python export_demo.py         # dữ liệu cho demo trong trình duyệt
```

Kết quả ghi ra `ch5/results/*.json`; báo cáo đọc trực tiếp các tệp đó nên không có con số nào
nhập tay.

## Dựng lại báo cáo

```bash
cd build
python build.py
```

Cần `pandoc` và Microsoft Word (Word dùng để cập nhật mục lục rồi xuất PDF).

## Dữ liệu

Hai tệp dữ liệu lớn không đưa lên repo (`ch1/mnist.npz`, `ch2/data/hanoi_housing.csv`); xem
hướng dẫn tải trong từng chương. Dữ liệu Chương 5 là bản chụp ngày 15/10/2026 lấy qua
`yfinance` và Google News RSS, đã kèm sẵn trong `ch5/data/`.

## Môi trường

Python 3.14 (NumPy, PyTorch 2.10, scikit-learn, sentence-transformers) và Python 3.13 với
TensorFlow 2.21, Keras 3 cho phần Keras của Chương 2–4. Toàn bộ chạy trên CPU.
