# -*- coding: utf-8 -*-
"""Bang va nhan xet cho Chuong 2 (ML co ban + MLP ba cai dat)."""
import os
from tables_base import J, f, pm, pc, n, loc, ROOT

FW2 = [("numpy", "NumPy (scratch)"), ("torch", "PyTorch"), ("keras", "Keras")]


def R2(fw):
    return J("ch2/results/ch2_%s.json" % fw)


def SK():
    return J("ch2/results/ch2_sklearn.json")


def a03(i):
    return J("ch2/results/a03_system%d.json" % i)


def t_c2_datasets():
    s1, s2, s3 = a03(1), a03(2), a03(3)
    return ("Ba bộ dữ liệu chính của Chương 2",
            ["", "Diabetes", "House Price (Hà Nội)", "Comments (YouTube)"],
            [["Liên kết", "[Kaggle: iammustafatz/diabetes-prediction-dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)",
              "Kaggle, tệp `VN_housing_dataset.csv` (chưa xác minh được URL, xem Tài liệu tham khảo)", "Kaggle, tệp `YoutubeCommentsDataSet.csv` (chưa xác minh được URL)"],
             ["Kích thước gốc", "100.000 × 9", "82.497 × 13", "18.408 × 2"],
             ["Sau làm sạch", n(s1["n_clean"]), n(s2["n_clean"]), n(s3["n_clean"])],
             ["Huấn luyện / kiểm tra", "%s / %s" % (n(s1["n_train"]), n(s1["n_test"])), "%s / %s" % (n(s2["n_train"]), n(s2["n_test"])), "%s / %s" % (n(s3["n_train"]), n(s3["n_test"]))],
             ["Đầu vào", "8 biến, chuẩn hóa z-score", "15 đặc trưng, chuẩn hóa z-score", "TF-IDF 1.000 chiều, thưa %s" % pc(s3["sparsity"], 1)],
             ["Bài toán và mất mát", "nhị phân, sigmoid + BCE", "hồi quy trên ln(giá), tuyến tính + MSE", "3 lớp, softmax + cross-entropy"],
             ["Phân bố nhãn", "%s dương tính" % pc(s1["pos_rate"], 1), "giá lệch phải (skew %s, sau ln: %s)" % (f(s2["skew_raw"], 1), f(s2["skew_log"], 1)),
              "%s / %s / %s (neg/neu/pos)" % tuple(pc(s3["class_counts"][k] / s3["n_clean"], 0) for k in ("negative", "neutral", "positive"))]])


def t_c2_housing_sample():
    import pandas as pd
    r = pd.read_csv(ROOT / "ch2/data/hanoi_housing.csv").head(4)
    return ("Bốn tin đầu của bộ House Price (một số cột)", ["Quận", "Loại hình", "Pháp lý", "Số tầng", "Phòng ngủ", "Diện tích", "Giá/m²"],
            [[x["Quận"], x["Loại hình nhà ở"], x["Giấy tờ pháp lý"] if isinstance(x["Giấy tờ pháp lý"], str) else "(thiếu)", x["Số tầng"] if isinstance(x["Số tầng"], str) or x["Số tầng"] == x["Số tầng"] else "(thiếu)", x["Số phòng ngủ"], x["Diện tích"], x["Giá/m2"]] for _, x in r.iterrows()])


def t_c2_comments_sample():
    import pandas as pd
    c = pd.read_csv(ROOT / "ch2/data/youtube_comments.csv").dropna()
    rows = []
    for lab in ("negative", "neutral", "positive"):
        for t in c[c.Sentiment == lab].Comment.head(2):
            rows.append([str(t)[:90].replace("|", "/").replace("\n", " "), lab])
    return ("Sáu bình luận mẫu của bộ Comments (cắt 90 ký tự)", ["Bình luận", "Nhãn"], rows)


def t_c2_config():
    cfg = R2("torch")
    rows = []
    for k, nm in (("diabetes", "Diabetes"), ("housing", "House Price"), ("comments", "Comments")):
        c = cfg[k]["config"]
        rows.append([nm, " → ".join(map(str, c["sizes"])), n(cfg[k]["runs"][0]["params"]), {"binary": "nhị phân", "regression": "hồi quy", "multiclass": "3 lớp"}[c["task"]], str(c["lr"]).replace(".", ","),
                     str(c["epochs"]), "toàn bộ (full-batch)" if not c["batch"] else str(c["batch"])])
    return ("Kiến trúc và siêu tham số dùng chung cho cả ba cài đặt (ReLU ở các tầng ẩn, SGD thuần)", ["Bộ dữ liệu", "Kiến trúc", "Tham số", "Bài toán", "Tốc độ học", "Epoch", "Kích thước lô"], rows)


def _sk_table(ds, cap, cols, hdr):
    rows = [[k] + [f(v[c], 3 if c != "mae" and c != "rmse" else 3) for c in cols] + [f(v["seconds"], 1)] for k, v in SK()[ds].items()]
    return (cap, hdr + ["Thời gian (s)"], rows)


def t_c2_sk_diabetes():
    return _sk_table("diabetes", "Các mô hình cổ điển trên Diabetes (một lần chạy)", ["accuracy", "precision", "recall", "f1", "roc_auc"], ["Mô hình", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])


def t_c2_sk_housing():
    return _sk_table("housing", "Các mô hình cổ điển trên House Price (MAE, RMSE tính bằng tỷ đồng)", ["r2_log", "r2_price", "mae", "rmse"], ["Mô hình", "R² (log)", "R² (tỷ đồng)", "MAE", "RMSE"])


def t_c2_sk_comments():
    return _sk_table("comments", "Các mô hình cổ điển trên Comments (một lần chạy)", ["accuracy", "macro_f1", "f1_neg", "f1_neu", "f1_pos"], ["Mô hình", "Accuracy", "Macro-F1", "F1 neg", "F1 neu", "F1 pos"])


def _best(ds, key):
    d = SK()[ds]
    k = max(d, key=lambda x: d[x][key])
    return k, d[k]


def c2_sk_comment():
    d, h, c = SK()["diabetes"], SK()["housing"], SK()["comments"]
    kd, vd = _best("diabetes", "f1")
    kh, vh = _best("housing", "r2_log")
    kc, vc = _best("comments", "macro_f1")
    return ("Trên Diabetes mô hình cổ điển tốt nhất theo F1 là %s (F1 = %s, ROC-AUC = %s). Trên House Price, %s đạt R² (log) = %s và MAE = %s tỷ, trong khi hồi quy tuyến tính chỉ đạt %s: quan hệ giữa vị trí, diện tích và giá có tính phi tuyến rõ. "
            "Trên Comments, %s đạt macro-F1 cao nhất (%s); Naive Bayes thấp rõ rệt (%s) vì lệch về lớp đa số. Nhìn chung, mô hình tuyến tính đã là đường cơ sở mạnh, và các mô hình dựa trên cây phi tuyến cải thiện thêm trên hai bộ bảng."
            % (kd, f(vd["f1"], 3), f(vd["roc_auc"], 3), kh, f(vh["r2_log"], 3), f(vh["mae"], 2), f(h["Hồi quy tuyến tính"]["r2_log"], 3), kc, f(vc["macro_f1"], 3), f(c["Naive Bayes đa thức"]["macro_f1"], 3)))


def _row(fw, nm, ds, cols, nd):
    m, s = R2(fw)[ds]["mean"], R2(fw)[ds]["std"]
    return [nm] + ["%s ± %s" % (f(m[c], nd), f(s[c], nd)) for c in cols]


def t_c2_diabetes():
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = [_row(fw, nm, "diabetes", cols, 4) for fw, nm in FW2]
    k, v = _best("diabetes", "f1")
    rows.append(["(đối chiếu) " + k] + [f(v[c], 4) for c in cols])
    return ("Diabetes, MLP 8 → 32 → 16 → 8 → 1 (19.031 mẫu kiểm tra; ba hạt giống; ngưỡng 0,5)", ["Cài đặt", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], rows)


def t_c2_housing():
    cols = ["r2_log", "r2_price", "mae", "rmse"]
    rows = [_row(fw, nm, "housing", cols, 3) for fw, nm in FW2]
    k, v = _best("housing", "r2_log")
    rows.append(["(đối chiếu) " + k] + [f(v[c], 3) for c in cols])
    return ("House Price, MLP 15 → 128 → 64 → 1 (12.934 mẫu kiểm tra; ba hạt giống; MAE, RMSE tính bằng tỷ đồng)", ["Cài đặt", "R² (log)", "R² (tỷ đồng)", "MAE", "RMSE"], rows)


def t_c2_comments():
    cols = ["accuracy", "macro_f1", "f1_neg", "f1_neu", "f1_pos"]
    rows = [_row(fw, nm, "comments", cols, 3) for fw, nm in FW2]
    k, v = _best("comments", "macro_f1")
    rows.append(["(đối chiếu) " + k] + [f(v[c], 3) for c in cols])
    return ("Comments, MLP 1000 → 32 → 3 (3.516 mẫu kiểm tra; ba hạt giống)", ["Cài đặt", "Accuracy", "Macro-F1", "F1 neg", "F1 neu", "F1 pos"], rows)


def c2_results_comment():
    out = []
    d = {fw: R2(fw)["diabetes"] for fw, _ in FW2}
    h = {fw: R2(fw)["housing"] for fw, _ in FW2}
    c = {fw: R2(fw)["comments"] for fw, _ in FW2}
    kd, vd = _best("diabetes", "f1")
    kh, vh = _best("housing", "r2_log")
    kc, vc = _best("comments", "macro_f1")
    f1s = [d[fw]["mean"]["f1"] for fw in d]
    out.append("**Diabetes.** F1 của ba cài đặt nằm trong khoảng %s đến %s (độ lệch chuẩn giữa hạt giống từ %s đến %s) và accuracy %s đến %s: ba cài đặt không phân biệt được với nhau. Cả ba đều thấp hơn %s (F1 = %s, precision = 1,000, recall %s): một lý do hợp lý là cây quyết định chia trực tiếp theo các ngưỡng của HbA1c và đường huyết, điều mà MLP với SGD thuần phải học dần. Recall chỉ khoảng %s ở mọi mô hình nghĩa là khoảng một phần ba ca bệnh bị bỏ sót tại ngưỡng 0,5." % (
        f(min(f1s), 3), f(max(f1s), 3), f(min(d[fw]["std"]["f1"] for fw in d), 3), f(max(d[fw]["std"]["f1"] for fw in d), 3), f(min(d[fw]["mean"]["accuracy"] for fw in d), 4), f(max(d[fw]["mean"]["accuracy"] for fw in d), 4), kd, f(vd["f1"], 3), f(vd["recall"], 3), f(sum(d[fw]["mean"]["recall"] for fw in d) / 3, 2)))
    r2s = [h[fw]["mean"]["r2_log"] for fw in h]
    lin = SK()["housing"]["Hồi quy tuyến tính"]
    out.append("**House Price.** R² (log) của ba cài đặt từ %s đến %s và MAE từ %s đến %s tỷ đồng; trên thang tiền thật (R² theo tỷ đồng) độ lệch chuẩn giữa hạt giống lớn hơn (đến %s), do vài căn rất đắt chi phối sai số. MLP tốt hơn hồi quy tuyến tính (R² log %s, MAE %s tỷ) nhưng thấp hơn %s (R² log %s, MAE %s tỷ): ở đây mạng nơ-ron bắt được tính phi tuyến mà mô hình tuyến tính bỏ lỡ, song rừng ngẫu nhiên bắt được nhiều hơn." % (
        f(min(r2s), 3), f(max(r2s), 3), f(min(h[fw]["mean"]["mae"] for fw in h), 3), f(max(h[fw]["mean"]["mae"] for fw in h), 3), f(max(h[fw]["std"]["r2_price"] for fw in h), 3),
        f(lin["r2_log"], 3), f(lin["mae"], 2), kh, f(vh["r2_log"], 3), f(vh["mae"], 2)))
    mf = [c[fw]["mean"]["macro_f1"] for fw in c]
    lr = SK()["comments"]["Hồi quy logistic đa lớp"]
    out.append("**Comments.** Accuracy %s đến %s, macro-F1 %s đến %s. F1 lớp negative (lớp thiểu số, khó nhất) chỉ %s đến %s so với lớp positive %s đến %s. Mô hình cổ điển %s đạt macro-F1 %s và hồi quy logistic %s, tức MLP một tầng ẩn chỉ ngang các mô hình tuyến tính trên đặc trưng TF-IDF: với 1.000 đặc trưng thưa và 14.066 mẫu, phần tương tác phi tuyến mà MLP có thể thêm vào rất ít." % (
        f(min(c[fw]["mean"]["accuracy"] for fw in c), 3), f(max(c[fw]["mean"]["accuracy"] for fw in c), 3), f(min(mf), 3), f(max(mf), 3),
        f(min(c[fw]["mean"]["f1_neg"] for fw in c), 2), f(max(c[fw]["mean"]["f1_neg"] for fw in c), 2), f(min(c[fw]["mean"]["f1_pos"] for fw in c), 3), f(max(c[fw]["mean"]["f1_pos"] for fw in c), 3), kc, f(vc["macro_f1"], 3), f(lr["macro_f1"], 3)))
    return "\n\n".join(out)


def c2_summary_numbers():
    d = sum(R2(fw)["diabetes"]["mean"]["f1"] for fw, _ in FW2) / 3
    h = sum(R2(fw)["housing"]["mean"]["r2_log"] for fw, _ in FW2) / 3
    c = sum(R2(fw)["comments"]["mean"]["macro_f1"] for fw, _ in FW2) / 3
    kd, vd = _best("diabetes", "f1")
    kh, vh = _best("housing", "r2_log")
    kc, vc = _best("comments", "macro_f1")
    return ("Trên Diabetes, %s đạt F1 = %s so với trung bình %s của ba cài đặt MLP (hơn khoảng %s điểm); trên House Price, %s đạt R² (log) %s so với %s của MLP, tuy MLP vượt hồi quy tuyến tính, KNN, SVR và cây đơn; trên Comments, MLP (macro-F1 %s) ngang %s (%s) và hồi quy logistic." % (
        kd, f(vd["f1"], 3), f(d, 3), f(100 * (vd["f1"] - d), 1), kh, f(vh["r2_log"], 3), f(h, 3), f(c, 3), kc, f(vc["macro_f1"], 3)))


def _tim(fw, ds):
    p = ROOT / ("ch2/results/ch2_%s_timing.json" % fw)
    if p.exists():
        return J("ch2/results/ch2_%s_timing.json" % fw)[ds]["seconds"], True
    return R2(fw)[ds]["seconds"], False


def t_c2_time():
    rows = []
    for fw, nm in FW2:
        t = [_tim(fw, ds)[0] for ds in ("diabetes", "housing", "comments")]
        rows.append([nm] + [f(x, 1) for x in t] + [f(sum(t), 1)])
    t0 = [sum(_tim("numpy", ds)[0] for ds in ("diabetes", "housing", "comments"))]
    for r, (fw, _) in zip(rows, FW2):
        r.append(f(t0[0] / sum(_tim(fw, ds)[0] for ds in ("diabetes", "housing", "comments")), 2) + "×")
    return ("Thời gian huấn luyện một mô hình (giây, hạt giống 42, CPU, đo tuần tự khi máy không chạy việc khác; NumPy một luồng, PyTorch 4 luồng, Keras dùng mặc định)",
            ["Cài đặt", "Diabetes", "House Price", "Comments", "Tổng", "Nhanh hơn NumPy"], rows)


def c2_time_comment():
    tn = {fw: [_tim(fw, ds)[0] for ds in ("diabetes", "housing", "comments")] for fw, _ in FW2}
    l_np = loc("ch2/dl_scratch.py", ["relu", "relu_derivative", "sigmoid", "softmax", "binary_cross_entropy", "mse_loss", "categorical_cross_entropy", "dense_layer", "MLPScratch"])
    l_pt = loc("ch2/ch2_run.py", ["run_torch"])
    l_k = loc("ch2/ch2_run.py", ["run_keras"])
    return ("Trên Diabetes (huấn luyện toàn bộ dữ liệu 1.500 lần, mỗi lần một ma trận 76.124 × 32) NumPy mất %s giây so với %s (PyTorch) và %s (Keras); trên House Price (lô 1.024) lần lượt %s, %s và %s giây; trên Comments (ma trận đầu vào 1.000 chiều) %s, %s và %s giây. "
            "Cần đọc các con số này cùng với điều kiện đo: bản NumPy chạy một luồng còn PyTorch dùng bốn luồng, nên một phần khoảng cách là do số luồng chứ không do framework; Keras (TensorFlow) chậm hơn PyTorch ở bài full-batch Diabetes, nơi chi phí mỗi bước `fit` là chi phí cố định. Khoảng cách cũng thay đổi theo cỡ bài toán (từ %s lần ở Comments đến %s lần ở Diabetes giữa NumPy và PyTorch). Bản scratch cần khoảng %d dòng code so với %d dòng (PyTorch) và %d dòng (Keras) cho phần mô hình và huấn luyện; đó là cái giá của việc tự viết lan truyền ngược và là lý do các framework tồn tại."
            % (f(tn["numpy"][0], 1), f(tn["torch"][0], 1), f(tn["keras"][0], 1), f(tn["numpy"][1], 1), f(tn["torch"][1], 1), f(tn["keras"][1], 1), f(tn["numpy"][2], 1), f(tn["torch"][2], 1), f(tn["keras"][2], 1), f(tn["numpy"][2] / tn["torch"][2], 1), f(tn["numpy"][0] / tn["torch"][0], 1), l_np, l_pt, l_k))
