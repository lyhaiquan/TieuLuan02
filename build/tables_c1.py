# -*- coding: utf-8 -*-
"""Mo dau va Chuong 1."""
from tables_base import J, f, pm, pc, n, loc


# ======================================================================= Mo dau
def t_overview():
    return ("Dữ liệu, mô hình và ba cài đặt của từng chương",
            ["Chương", "Chủ đề", "Dữ liệu (kích thước)", "Mô hình", "Ba cài đặt"],
            [["1", "Lịch sử AI", "Iris (150), MNIST (70.000), Diabetes (100.000)", "Perceptron, MLP, SVM, CNN LeNet, luật chuyên gia", "NumPy, scikit-learn, PyTorch"],
             ["2", "Học máy cơ bản và MLP", "Diabetes (95.155), House Price Hà Nội (64.663), Comments (17.582)", "Hồi quy tuyến tính/logistic, KNN, SVM, cây, rừng; MLP", "NumPy, PyTorch, Keras"],
             ["3", "CNN", "Diabetes (20.000), House Price (30.229), Comments (18.408); mở rộng Fashion-MNIST, CIFAR-10", "CNN 1 chiều; ResNet, Inception, DenseNet", "NumPy, PyTorch, Keras"],
             ["4", "RNN", "Churn (10.000), AMZN lợi suất và biến động (6.684 phiên)", "RNN, LSTM, GRU", "NumPy, PyTorch, Keras"]])


# ======================================================================= Chuong 1
def t_timeline():
    rows = [["1943", "Nơ-ron nhân tạo McCulloch - Pitts", "Nền móng"],
            ["1950", "Phép thử Turing", "Nền móng"],
            ["1956", "Hội thảo Dartmouth, tên gọi \"AI\"", "Khai sinh"],
            ["1958", "Perceptron (Rosenblatt)", "Mạng nơ-ron đầu tiên"],
            ["1969", "Minsky - Papert: giới hạn perceptron", "Mùa đông thứ nhất (1974 - 1980)"],
            ["1980", "Hệ chuyên gia XCON, MYCIN", "Hệ dựa trên luật"],
            ["1986", "Lan truyền ngược (Rumelhart, Hinton, Williams)", "Mạng nhiều tầng"],
            ["1995", "SVM (Cortes, Vapnik)", "Học máy thống kê"],
            ["1997", "LSTM; Deep Blue thắng Kasparov", "Chuỗi và tìm kiếm"],
            ["1998", "LeNet-5, MNIST", "CNN đầu tiên"],
            ["2009", "ImageNet", "Dữ liệu lớn"],
            ["2012", "AlexNet thắng ImageNet", "Học sâu bùng nổ"],
            ["2015", "ResNet", "Mạng rất sâu"],
            ["2017", "Transformer", "Tự chú ý"],
            ["2018 - 2022", "BERT, GPT-3, ChatGPT", "Mô hình ngôn ngữ lớn"],
            ['[2020 - 2024]{custom-style="Moi"}',
             '[RAG (Lewis và cộng sự); GraphRAG (Edge và cộng sự)]{custom-style="Moi"}',
             '[Tri thức ra ngoài trọng số]{custom-style="Moi"}']]
    return ("Các cột mốc chính của lịch sử AI", ["Năm", "Sự kiện", "Giai đoạn"], rows)


def t_c1_datasets():
    d, m = J("ch1/results_ch1_diabetes.json"), J("ch1/results_ch1_mnist.json")
    return ("Ba bộ dữ liệu của Chương 1",
            ["", "Iris", "MNIST", "Diabetes"],
            [["Liên kết", "[archive.ics.uci.edu/dataset/53/iris](https://archive.ics.uci.edu/dataset/53/iris)",
              "[yann.lecun.com/exdb/mnist](https://yann.lecun.com/exdb/mnist/); bản OpenML `mnist_784`",
              "[kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)"],
             ["Kích thước", "150 × 4", "70.000 × 784 (28 × 28)", "%s × %d" % (n(d["raw_shape"][0]), d["raw_shape"][1])],
             ["Bài toán", "phân loại 3 loài hoa", "phân loại 10 chữ số", "phân loại nhị phân"],
             ["Phân bố lớp", "50 / 50 / 50", "gần cân bằng (%s - %s ảnh mỗi chữ số)" % (n(min(m["train_counts"])), n(max(m["train_counts"]))), "%s%% / %s%%" % (f(100 * (1 - d["raw_pos"] / d["raw_shape"][0]), 1), f(100 * d["raw_pos"] / d["raw_shape"][0], 1))],
             ["Thời kỳ đại diện", "thống kê, perceptron", "mạng nhiều tầng, SVM, CNN", "học máy ứng dụng"]])


def t_c1_iris_sample():
    r = J("ch1/results_ch1_iris.json")["head"]
    return ("Sáu mẫu ngẫu nhiên của bộ Iris (đơn vị cm)",
            ["Dài đài", "Rộng đài", "Dài cánh", "Rộng cánh", "Loài"],
            [[f(x["sepal length (cm)"], 1), f(x["sepal width (cm)"], 1), f(x["petal length (cm)"], 1), f(x["petal width (cm)"], 1), x["species"]] for x in r])


def t_c1_diabetes_sample():
    r = J("ch1/results_ch1_diabetes.json")["head"]
    return ("Năm hồ sơ đầu của bộ Diabetes",
            ["Giới tính", "Tuổi", "Tăng HA", "Bệnh tim", "Hút thuốc", "BMI", "HbA1c", "Đường huyết", "Nhãn"],
            [[x["gender"], f(x["age"], 0), x["hypertension"], x["heart_disease"], x["smoking_history"], f(x["bmi"], 2), f(x["HbA1c_level"], 1), x["blood_glucose_level"], x["diabetes"]] for x in r])


def mnist_comment():
    m = J("ch1/results_ch1_mnist.json")
    return ("Mười lớp gần cân bằng (từ %s đến %s ảnh mỗi chữ số), nên accuracy là độ đo hợp lý và không cần xử lý mất cân bằng. Chỉ %s điểm ảnh khác 0, nghĩa là dữ liệu thưa và nằm gọn ở giữa khung. "
            "Hai tính chất này giải thích vì sao MNIST từng là \"bài thi chuẩn\" cho mọi thế hệ thuật toán: bài toán đủ dễ để thuật toán nào cũng chạy được, nhưng vẫn đủ phân hóa để thấy khác biệt giữa chúng. Ngày nay nó quá dễ để phân biệt các mô hình hiện đại, nên Chương 3 chuyển sang Fashion-MNIST và CIFAR-10."
            % (n(min(m["train_counts"])), n(max(m["train_counts"])), pc(m["pixel_nonzero"], 0)))


def diabetes_comment():
    d = J("ch1/results_ch1_diabetes.json")
    n1 = d["raw_pos"]
    N = d["raw_shape"][0]
    return ("Chỉ %s hồ sơ (%s) dương tính, nên một mô hình luôn đoán \"không bệnh\" đã đạt accuracy %s mà không phát hiện được ca nào: với dữ liệu mất cân bằng phải xem thêm precision, recall và F1. "
            "HbA1c chỉ nhận %d giá trị rời rạc; nhóm bệnh dồn về phía giá trị cao, nhưng hai phân bố chồng lấn trong khoảng 5,7 đến 6,6, quanh ngưỡng chẩn đoán 6,5%%; nhóm bệnh cũng lệch về tuổi cao. Mục 1.5.3 sẽ đo xem một luật ngưỡng dựa trên tiêu chuẩn y khoa mô tả nhãn của bộ dữ liệu này tốt đến đâu."
            % (n(n1), pc(n1 / N, 1), pc(1 - n1 / N, 1), 18))


def t_c1_xor():
    r = J("ch1/results_ch1_xor.json")
    rows = [[k, "perceptron (1958)", f(100 * r[k]["acc"], 0) + "%", ("hội tụ sau %d epoch" % r[k]["epochs"]) if r[k]["converged"] else "không hội tụ (200 epoch)"] for k in ("AND", "OR", "XOR")]
    rows.append(["XOR", "MLP 2-4-1 (1986)", f(100 * r["XOR_MLP"]["acc"], 0) + "%", "5.000 bước lan truyền ngược"])
    return ("Perceptron và MLP trên các hàm logic", ["Hàm", "Mô hình", "Độ chính xác", "Ghi chú"], rows)


def iris_eps():
    return str(J("ch1/results_ch1_iris.json")["perceptron"]["setosa_vs_rest"]["epochs"])


def iris_acc2():
    return pc(J("ch1/results_ch1_iris.json")["perceptron"]["versicolor_vs_virginica"]["acc"], 0)


def t_c1_mnist():
    m = J("ch1/results_ch1_mnist.json")["rows"]
    return ("Sai số kiểm tra trên 10.000 ảnh MNIST theo từng thế hệ phương pháp (một lần chạy; thời gian đo khi máy đang chạy các tác vụ khác nên chỉ mang tính tham khảo)",
            ["Phương pháp", "Thời kỳ", "Sai số", "Độ chính xác", "Thời gian huấn luyện (s)"],
            [[r["model"], r["era"], f(r["err"], 2) + "%", f(100 - r["err"], 2) + "%", f(r["sec"], 0)] for r in m])


def mnist_eras_comment():
    m = {r["model"].split(" (")[0]: r for r in J("ch1/results_ch1_mnist.json")["rows"]}
    ks = list(m)
    lin, knn, svm, mlp, cnn = [m[k]["err"] for k in ks]
    return ("Kết quả có đủ hai quy luật. Thứ nhất, bước từ mô hình tuyến tính sang mô hình phi tuyến giảm sai số mạnh nhất: từ %s%% xuống %s%% (KNN), %s%% (MLP) và %s%% (SVM nhân RBF), tức giảm %s đến %s lần. "
            "Thứ hai, CNN kiểu LeNet-5, vốn khai thác cấu trúc không gian của ảnh bằng bộ lọc dùng chung trọng số, đạt sai số thấp nhất: %s%%, thấp hơn cả SVM (%s%%) và MLP (%s%%). "
            "Có một chi tiết thú vị: MLP nhỏ huấn luyện 10 epoch (%s%%) kém SVM (%s%%), gợi lại giai đoạn cuối thập niên 1990 khi SVM cạnh tranh sòng phẳng với các mạng nhiều tầng cỡ nhỏ trước khi học sâu bùng nổ. "
            "Báo cáo gốc của LeCun cũng cho cùng một thứ tự định tính (mô hình tuyến tính tệ nhất, SVM và LeNet-5 tốt nhất) [9]. "
            "Cần nhớ đây là một lần chạy, mô hình nhỏ và chỉ 10 epoch; chênh lệch cỡ vài phần mười điểm phần trăm (chẳng hạn giữa CNN và SVM) chưa đủ để xếp hạng chắc chắn."
            % (f(lin, 2), f(knn, 2), f(mlp, 2), f(svm, 2), f(lin / knn, 1), f(lin / svm, 1), f(cnn, 2), f(svm, 2), f(mlp, 2), f(mlp, 2), f(svm, 2)))


def t_c1_diab():
    r = J("ch1/results_ch1_diabetes.json")["rows"]
    return ("Các thế hệ phương pháp trên tập kiểm tra Diabetes (19.031 hồ sơ)",
            ["Phương pháp", "Thời kỳ", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            [[x["model"], x["era"], f(x["accuracy"]), f(x["precision"]), f(x["recall"]), f(x["f1"]), f(x["auc"]) if x["auc"] else "-"] for x in r])


def diab_eras_comment():
    r = J("ch1/results_ch1_diabetes.json")["rows"]
    rule, lr, dt, mlp = r[0], r[2], r[3], r[4]
    return ("Luật chuyên gia bắt được phần lớn ca bệnh (recall %s) nhưng báo động nhầm rất nhiều: precision chỉ %s, nên F1 = %s, thấp hơn hẳn mọi mô hình học từ dữ liệu (F1 từ %s đến %s). Một lời giải thích hợp lý là nhãn của bộ dữ liệu này không khớp với một ngưỡng chẩn đoán duy nhất (hai phân bố HbA1c chồng lấn ở hình trên), nên một luật cứng áp cùng một ngưỡng cho mọi người mất nhiều thông tin (tuổi, BMI, bệnh nền) mà mô hình học được tận dụng. "
            "Bước nhảy lớn nhất của bảng là từ luật cứng sang mô hình học được; giữa hồi quy logistic, cây quyết định và MLP nhỏ khác biệt F1 nhỏ hơn nhiều, trong khi recall của mọi mô hình học được đều quanh %s đến %s: một phần ba ca bệnh vẫn bị bỏ sót, tức giới hạn nằm ở dữ liệu chứ không ở độ phức tạp mô hình. "
            "Chương 2 sẽ đào sâu đúng điểm này, cùng với hiệu chỉnh ngưỡng quyết định."
            % (f(rule["recall"], 2), f(rule["precision"], 2), f(rule["f1"], 2), f(lr["f1"], 2), f(max(lr["f1"], dt["f1"], mlp["f1"]), 2), f(min(lr["recall"], dt["recall"], mlp["recall"]), 2), f(max(lr["recall"], dt["recall"], mlp["recall"]), 2)))
