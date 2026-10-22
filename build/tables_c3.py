# -*- coding: utf-8 -*-
"""Bang va nhan xet cho Chuong 3 (CNN)."""
from tables_base import J, f, pc, n, loc

FW3 = [("numpy", "NumPy from scratch"), ("pytorch", "PyTorch"), ("tensorflow", "Keras (TensorFlow)")]


def r3(fw, ds):
    return J("ch3/results/%s_%s.json" % (fw, ds))


def BL():
    return J("ch3/results/baselines.json")


def c3(key, ds="comments"):
    return [r3(fw, ds)["metrics"][key] for fw, _ in FW3]


def t_c3_datasets():
    s = J("ch3/ch3_stats.json")
    h, c, d = s["housing"], s["comments"], s["diabetes"]
    return ("Ba bộ dữ liệu chính của Chương 3",
            ["", "Diabetes", "House Price", "Comments"],
            [["Liên kết", "[Kaggle: iammustafatz/diabetes-prediction-dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)",
              "Kaggle, tệp `vietnam_housing_dataset.csv` (chưa xác minh được URL, xem Tài liệu tham khảo)",
              "Kaggle, tệp `YoutubeCommentsDataSet.csv` (chưa xác minh được URL)"],
             ["Kích thước gốc", "%s × %d" % (n(d["raw_shape"][0]), d["raw_shape"][1]), "%s × %d" % (n(h["raw_shape"][0]), h["raw_shape"][1]), "18.408 × 2"],
             ["Dùng trong bài", "%s huấn luyện / %s kiểm tra (mẫu 20.000)" % (n(d["n_train"]), n(d["n_test"])), "%s / %s" % (n(h["n_train"]), n(h["n_test"])), "%s / %s" % (n(c["n_train"]), n(c["n_test"]))],
             ["Đầu vào", "%d đặc trưng số (chuỗi dài %d)" % (d["n_feat"], d["n_feat"]), "%d đặc trưng số" % h["n_feat"], "chuỗi %d token, từ điển %s" % (c["max_len"], n(c["vocab"]))],
             ["Bài toán", "nhị phân, sigmoid + BCE", "hồi quy, tuyến tính + MSE", "3 lớp, softmax + cross-entropy"],
             ["Phân bố nhãn", "%s dương" % pc(d["pos_train"]), "giá trung vị %s tỷ, lệch phải" % f(h["price_median"], 1), "%s / %s / %s (neg/neu/pos)" % tuple(pc(x / sum(c["counts"]), 0) for x in c["counts"])]])


def _cell(v):
    v = str(v)
    if v in ("nan", "None", ""):
        return "(thiếu)"
    try:
        x = float(v)
        return f(x, 0) if x == int(x) else f(x, 1)
    except ValueError:
        return v


def t_c3_housing_sample():
    r = J("ch3/ch3_stats.json")["housing"]["head"]
    return ("Bốn tin đầu của bộ House Price (một số cột; \"(thiếu)\" là ô trống trong dữ liệu gốc)", ["Địa chỉ (phần cuối)", "Diện tích (m²)", "Số tầng", "Phòng ngủ", "Pháp lý", "Giá (tỷ)"],
            [[x["Address"][-36:], _cell(x["Area"]), _cell(x["Floors"]), _cell(x["Bedrooms"]), _cell(x["Legal status"]), _cell(x["Price"])] for x in r])


def t_c3_comments_sample():
    r = J("ch3/ch3_stats.json")["comments"]["samples"]
    return ("Sáu bình luận mẫu của bộ Comments (cắt 90 ký tự)", ["Bình luận", "Nhãn"], [[x["Comment"].replace("|", "/").replace("\n", " "), x["Sentiment"]] for x in r])


def t_c3_diabetes():
    rows = []
    for fw, nm in FW3:
        r = r3(fw, "diabetes")
        m = r["metrics"]
        rows.append([nm, f(m["accuracy"]), f(m["precision"]), f(m["recall"]), f(m["f1"]), n(r["n_params"]), f(r["seconds"], 1)])
    for k, m in BL()["diabetes"].items():
        rows.append(["(đối chiếu) " + k, f(m["accuracy"]), f(m["precision"]), f(m["recall"]), f(m["f1"]), "-", f(m["seconds"], 1)])
    return ("Diabetes (4.000 mẫu kiểm tra, 30 epoch, một hạt giống; hai hàng cuối là mô hình không dùng mạng nơ-ron trên cùng phép chia)", ["Cài đặt", "Accuracy", "Precision", "Recall", "F1", "Tham số", "Thời gian (s)"], rows)


def t_c3_housing():
    rows = []
    for fw, nm in FW3:
        r = r3(fw, "housing")
        m = r["metrics"]
        rows.append([nm, f(m["mae"], 3), f(m["rmse"], 3), f(m["r2"]), f(m["r2_log"]), n(r["n_params"]), f(r["seconds"], 1)])
    for k, m in BL()["housing"].items():
        rows.append(["(đối chiếu) " + k, f(m["mae"], 3), f(m["rmse"], 3), f(m["r2"]), f(m["r2_log"]), "-", f(m["seconds"], 1)])
    return ("House Price (6.045 mẫu kiểm tra, 40 epoch, một hạt giống; MAE và RMSE tính bằng tỷ đồng; hai hàng cuối là mô hình không dùng mạng nơ-ron)", ["Cài đặt", "MAE", "RMSE", "R²", "R² (log)", "Tham số", "Thời gian (s)"], rows)


def t_c3_comments():
    rows = []
    for fw, nm in FW3:
        r = r3(fw, "comments")
        m = r["metrics"]
        pcs = [x["f1"] for x in m["per_class"]]
        rows.append([nm, f(m["accuracy"]), f(m["macro_f1"]), f(pcs[0], 3), f(pcs[1], 3), f(pcs[2], 3), f(r["seconds"], 1)])
    for k, m in BL()["comments"].items():
        pcs = [x["f1"] for x in m["per_class"]]
        rows.append(["(đối chiếu) " + k, f(m["accuracy"]), f(m["macro_f1"]), f(pcs[0], 3), f(pcs[1], 3), f(pcs[2], 3), f(m["seconds"], 1)])
    return ("Comments (3.672 mẫu kiểm tra, 4 epoch, một hạt giống; hàng cuối là hồi quy logistic trên túi từ, không dùng mạng nơ-ron)", ["Cài đặt", "Accuracy", "Macro-F1", "F1 negative", "F1 neutral", "F1 positive", "Thời gian (s)"], rows)


def c3_results_comment():
    d = [r3(fw, "diabetes")["metrics"] for fw, _ in FW3]
    h = [r3(fw, "housing")["metrics"] for fw, _ in FW3]
    c = [r3(fw, "comments")["metrics"] for fw, _ in FW3]
    return ("**Diabetes.** Accuracy của cả ba cài đặt nằm trong khoảng %s đến %s, nhưng chỉ 8,5%% mẫu là dương: đoán toàn âm đã cho accuracy khoảng 0,915. Precision cao (%s đến %s) đi cùng recall trung bình (%s đến %s), tức mô hình dè dặt: ít báo nhầm nhưng bỏ sót khoảng một phần ba ca bệnh. Với bài sàng lọc, đó là đánh đổi ngược với mong muốn và cần hạ ngưỡng quyết định hoặc thêm trọng số lớp. "
            "**House Price.** MAE khoảng %s đến %s tỷ trên mức giá trung bình 5,9 tỷ, R² khoảng %s đến %s. Mức này phản ánh giới hạn của dữ liệu (không có tọa độ) hơn là giới hạn của CNN. "
            "**Comments.** Accuracy từ %s đến %s, macro-F1 từ %s đến %s; lớp positive chiếm đa số nên F1 lớp này (khoảng 0,85) vượt xa lớp negative (%s đến %s)."
            % (f(min(x["accuracy"] for x in d), 3), f(max(x["accuracy"] for x in d), 3), f(min(x["precision"] for x in d), 2), f(max(x["precision"] for x in d), 2), f(min(x["recall"] for x in d), 2), f(max(x["recall"] for x in d), 2),
               f(min(x["mae"] for x in h), 2), f(max(x["mae"] for x in h), 2), f(min(x["r2"] for x in h), 3), f(max(x["r2"] for x in h), 3),
               f(min(x["accuracy"] for x in c), 3), f(max(x["accuracy"] for x in c), 3), f(min(x["macro_f1"] for x in c), 3), f(max(x["macro_f1"] for x in c), 3),
               f(min(x["per_class"][0]["f1"] for x in c), 2), f(max(x["per_class"][0]["f1"] for x in c), 2)))


def t_c3_time():
    rows = []
    tot0 = None
    for fw, nm in FW3:
        t = [r3(fw, ds)["seconds"] for ds in ("diabetes", "housing", "comments")]
        tot0 = tot0 or sum(t)
        rows.append([nm] + [f(x, 1) for x in t] + [f(sum(t), 1), f(tot0 / sum(t), 2) + "×"])
    return ("Thời gian huấn luyện (giây), đo trên CPU", ["Cài đặt", "Diabetes (30 ep)", "House Price (40 ep)", "Comments (4 ep)", "Tổng", "Nhanh hơn NumPy"], rows)


def c3_time_comment():
    tn = [sum(r3(fw, ds)["seconds"] for ds in ("diabetes", "housing", "comments")) for fw, _ in FW3]
    dn = [r3(fw, "diabetes")["seconds"] for fw, _ in FW3]
    cn = [r3(fw, "comments")["seconds"] for fw, _ in FW3]
    l_np = loc("ch3/src/nb01_scratch.py", ["Conv1D", "ReLU", "MaxPool1D", "GlobalMaxPool1D", "Flatten", "Dense", "Embedding", "Dropout", "Sequential", "Adam", "train"])
    l_pt = loc("ch3/src/nb02_pytorch.py", ["TabularCNN", "TextCNN", "train_torch"])
    l_tf = loc("ch3/src/nb03_tensorflow.py", ["build_tabular_cnn", "build_text_cnn"])
    return ("Tính tổng ba bộ dữ liệu, PyTorch nhanh hơn NumPy %s lần và Keras nhanh hơn %s lần. Khoảng cách tập trung ở Comments (%s giây của NumPy so với %s và %s giây), nơi lớp embedding và convolution trên 14.692 chuỗi dài 60 có lẽ hưởng lợi nhiều từ kernel tối ưu và đa luồng của PyTorch và TensorFlow (tôi chưa tách riêng từng yếu tố). Ở Diabetes, bài nhỏ nhất, NumPy (%s giây) nhanh hơn cả PyTorch (%s giây), vì chi phí cố định của framework (khởi tạo, chuyển dữ liệu qua tensor) có thể lấn át phép tính khi mạng chỉ có 881 tham số. "
            "Về độ dài code, phần mô hình và huấn luyện cần khoảng %d dòng (không tính dòng trống và chú thích) cho NumPy, %d cho PyTorch (gồm cả vòng lặp huấn luyện) và %d cho Keras (chỉ phần xây mô hình, vì `fit()` thay cho vòng lặp). "
            "Bản scratch dài gấp khoảng %s lần bản PyTorch: đó là cái giá của việc tự viết lan truyền ngược."
            % (f(tn[0] / tn[1], 1), f(tn[0] / tn[2], 1), f(cn[0], 1), f(cn[1], 1), f(cn[2], 1), f(dn[0], 1), f(dn[1], 1), l_np, l_pt, l_tf, f(l_np / l_pt, 1)))


def t_c3_arch():
    rows = [["Fashion-MNIST", "CNN cơ bản", 0.8820, 0.8827, 288170, 114.9], ["Fashion-MNIST", "ResNet", 0.8713, 0.8697, 272790, 164.2],
            ["Fashion-MNIST", "Inception", 0.8857, 0.8862, 272522, 511.8], ["Fashion-MNIST", "DenseNet", 0.8830, 0.8815, 279690, 849.0],
            ["CIFAR-10", "CNN cơ bản", 0.7110, 0.7084, 288746, 132.6], ["CIFAR-10", "ResNet", 0.6663, 0.6621, 273150, 419.0],
            ["CIFAR-10", "Inception", 0.6933, 0.6894, 273098, 766.3], ["CIFAR-10", "DenseNet", 0.6917, 0.6863, 280410, 1322.3],
            ["Diabetes", "CNN cơ bản", 0.9708, 0.8935, 96482, 10.4], ["Diabetes", "ResNet", 0.9695, 0.8894, 94422, 63.6],
            ["Diabetes", "Inception", 0.9692, 0.8890, 130146, 174.7], ["Diabetes", "DenseNet", 0.9712, 0.8940, 163042, 160.3]]
    return ("Bốn kiến trúc tích chập trên ba bộ dữ liệu (số liệu từ báo cáo Lab 05 của chính tác giả, PyTorch, một hạt giống)",
            ["Bộ dữ liệu", "Kiến trúc", "Accuracy", "Macro-F1", "Tham số", "Thời gian (s)"],
            [[a, b, f(c), f(d), n(e), f(g, 1)] for a, b, c, d, e, g in rows])


def t_c3_shapes():
    d = J("ch3/results/numpy_diabetes.json")["n_params"]
    h = J("ch3/results/numpy_housing.json")["n_params"]
    c = J("ch3/results/numpy_comments.json")["n_params"]
    rows = [["Diabetes", "Conv1D(1→16, k=3)", "8 → 6", "1·16·3 + 16 = 64"],
            ["", "Conv1D(16→16, k=3)", "6 → 4", "16·16·3 + 16 = 784"],
            ["", "MaxPool(2), Flatten", "4 → 2 (16 × 2 = 32)", "0"],
            ["", "Dense(32→1)", "32 → 1", "32 + 1 = 33"],
            ["", "Tổng", "", "%s (đo được: %s)" % (n(64 + 784 + 33), n(d))],
            ["House Price", "Conv1D(1→16), Conv1D(16→16), MaxPool, Flatten", "19 → 17 → 15 → 7 (16 × 7 = 112)", "64 + 784 = 848"],
            ["", "Dense(112→32) + ReLU, Dense(32→1)", "112 → 32 → 1", "(112·32 + 32) + (32 + 1) = 3.649"],
            ["", "Tổng", "", "%s (đo được: %s)" % (n(848 + 3649), n(h))],
            ["Comments", "Embedding(15.000 × 32)", "60 → 60 × 32", "480.000"],
            ["", "Conv1D(32→64, k=3)", "60 → 58", "32·64·3 + 64 = 6.208"],
            ["", "MaxPool(2)", "58 → 29", "0"],
            ["", "Conv1D(64→64, k=3)", "29 → 27", "64·64·3 + 64 = 12.352"],
            ["", "GlobalMaxPool, Dense(64→3)", "27 → 1 (64)", "64·3 + 3 = 195"],
            ["", "Tổng", "", "%s (đo được: %s)" % (n(480000 + 6208 + 12352 + 195), n(c))]]
    return ("Kích thước đầu ra theo công thức $L_{out} = L - k + 1$ và số tham số của từng lớp (tính tay, đối chiếu với số do ba cài đặt báo cáo)", ["Bài toán", "Lớp", "Độ dài chuỗi", "Số tham số"], rows)


def c3_baseline_comment():
    b = BL()
    dn = [r3(fw, "diabetes")["metrics"]["f1"] for fw, _ in FW3]
    hn = [r3(fw, "housing")["metrics"]["r2"] for fw, _ in FW3]
    cn = [r3(fw, "comments")["metrics"] for fw, _ in FW3]
    bd, bh, bc = b["diabetes"], b["housing"], b["comments"]["Hồi quy logistic trên túi từ"]
    return ("Trên Diabetes, CNN (F1 %s đến %s) vượt hồi quy logistic (%s) nhưng thấp hơn rừng ngẫu nhiên (%s). Trên House Price, CNN (R² %s đến %s) nằm giữa hồi quy tuyến tính (%s) và rừng ngẫu nhiên (%s): bước nhảy lớn nhất là từ mô hình tuyến tính lên mô hình phi tuyến, và CNN không đi xa hơn rừng ngẫu nhiên. "
            "Trên Comments, hồi quy logistic trên túi từ (accuracy %s, macro-F1 %s) ngang, thậm chí nhỉnh hơn CNN (accuracy %s đến %s, macro-F1 %s đến %s): ở độ dài chuỗi 60 và 14.692 mẫu, bộ lọc 3 token chưa thêm được gì so với việc đếm từ."
            % (f(min(dn), 3), f(max(dn), 3), f(bd["Hồi quy logistic"]["f1"], 3), f(bd["Rừng ngẫu nhiên (200 cây)"]["f1"], 3), f(min(hn), 3), f(max(hn), 3), f(bh["Hồi quy tuyến tính"]["r2"], 3), f(bh["Rừng ngẫu nhiên (200 cây)"]["r2"], 3),
               f(bc["accuracy"], 3), f(bc["macro_f1"], 3), f(min(x["accuracy"] for x in cn), 3), f(max(x["accuracy"] for x in cn), 3), f(min(x["macro_f1"] for x in cn), 3), f(max(x["macro_f1"] for x in cn), 3)))
