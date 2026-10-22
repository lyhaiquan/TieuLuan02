# -*- coding: utf-8 -*-
"""Bang va nhan xet cho Chuong 4 (RNN)."""
from tables_base import J, f, pm, pc, n, loc

FW4 = [("numpy", "NumPy"), ("torch", "PyTorch"), ("keras", "Keras")]
CELLS = ["RNN", "LSTM", "GRU"]


def R4(fw, task, model):
    return J("ch4/results/%s__%s__%s.json" % (fw, task, model))


def S4():
    return J("ch4/ch4_stats.json")


def sec(fw, task, model):
    r = R4(fw, task, model)["runs"]
    return sum(x["seconds"] for x in r) / len(r)


def t_c4_datasets():
    s = S4()
    c, a = s["churn"], s["amzn"]
    return ("Hai bộ dữ liệu của Chương 4",
            ["", "Churn Modelling", "Amazon (AMZN)"],
            [["Liên kết", "[github.com/HarshitWaldia/Churn_Modelling](https://github.com/HarshitWaldia/Churn_Modelling)",
              "[kaggle.com/datasets/henryshan/amazon-com-inc-amzn](https://www.kaggle.com/datasets/henryshan/amazon-com-inc-amzn)"],
             ["Kích thước gốc", "%s × %d" % (n(c["shape"][0]), c["shape"][1]), "%s × %d (ngày %s đến %s)" % (n(a["shape"][0]), a["shape"][1], a["start"], a["end"])],
             ["Mẫu dùng", "%s huấn luyện / %s validation / %s kiểm tra" % (n(c["n_train"]), n(c["n_val"]), n(c["n_test"])),
              "lợi suất: %s / %s / %s; biến động: %s / %s / %s (cửa sổ 60 phiên)" % (n(a["ret"]["n_train"]), n(a["ret"]["n_val"]), n(a["ret"]["n_test"]), n(a["vol"]["n_train"]), n(a["vol"]["n_val"]), n(a["vol"]["n_test"]))],
             ["Bài toán", "phân lớp nhị phân (khách rời ngân hàng)", "hồi quy: lợi suất phiên sau; biến động thực 5 phiên tới"],
             ["Đầu vào RNN", "chuỗi 11 bước × 1 đặc trưng", "chuỗi 60 bước × 3 đặc trưng"],
             ["Phân bố", "%s rời đi (mất cân bằng)" % pc(c["rate"]), "%s số phiên tăng giá; độ lệch chuẩn lợi suất ngày %s" % (pc(a["up_share"]), pc(a["ret_std"], 2))]])


def t_c4_churn_sample():
    r = S4()["churn"]["head"]
    return ("Năm khách hàng đầu của bộ Churn (một số cột)", ["Điểm tín dụng", "Quốc gia", "Giới tính", "Tuổi", "Thâm niên", "Số dư", "Số sản phẩm", "Hoạt động", "Rời đi"],
            [[x["CreditScore"], x["Geography"], x["Gender"], x["Age"], x["Tenure"], n(float(x["Balance"])), x["NumOfProducts"], x["IsActiveMember"], x["Exited"]] for x in r])


def t_c4_amzn_sample():
    s = S4()["amzn"]
    rows = [[x["Date"][:10], x["Open"], x["High"], x["Low"], x["Close"], n(float(x["Volume"]))] for x in s["head"][:3]]
    rows.append(["...", "...", "...", "...", "...", "..."])
    rows += [[x["Date"][:10], x["Open"], x["High"], x["Low"], x["Close"], n(float(x["Volume"]))] for x in s["tail"][-2:]]
    return ("Một số dòng của bộ AMZN (giá USD, đã điều chỉnh chia cổ phiếu)", ["Ngày", "Mở", "Cao", "Thấp", "Đóng", "Khối lượng"], rows)


def c4_churn_comment():
    s = S4()["churn"]
    return ("Chỉ %s khách rời đi, nên một mô hình luôn đoán \"ở lại\" đạt accuracy %s mà vô dụng: ta dùng ROC-AUC và F1. Quốc gia phân bố 50/25/25 giữa Pháp, Đức, Tây Ban Nha; %s khách có số dư bằng 0. "
            "Trong ma trận tương quan, các hệ số tuyến tính với nhãn lớn nhất là tuổi (0,29), quốc gia Đức (0,17), tình trạng hoạt động (−0,16) và số dư (0,12), còn số sản phẩm (−0,05) và thâm niên (−0,01) gần như không tương quan tuyến tính. Điều đó không có nghĩa chúng vô dụng: phân tích ở bài thực hành RNN cho thấy tuổi và số sản phẩm có quan hệ phi tuyến với việc rời bỏ, đó là lý do mô hình phi tuyến vượt hồi quy logistic (ROC-AUC 0,775 trên cùng chia tập). "
            "Quan trọng hơn cho việc đọc kết quả: đây là ảnh chụp một thời điểm, không có thứ tự thời gian thật giữa các cột. Thứ tự chuỗi do chúng ta đặt ra, nên RNN không có lý do lý thuyết để hơn một mạng đọc cả hàng cùng lúc."
            % (pc(s["rate"]), pc(1 - s["rate"]), pc(s["bal0"], 0)))


def c4_amzn_comment():
    a = S4()["amzn"]
    return ("Giá cổ phiếu tăng từ %s USD (1997) lên %s USD (2023), nghĩa là hơn một nghìn lần, nên giá thô không có phân phối ổn định và không dùng thẳng làm đầu vào được. Lợi suất log hằng ngày thì dao động quanh 0 với độ lệch chuẩn %s, đuôi dày hơn phân phối chuẩn (nhiều ngày biến động cực đoan), và chỉ %s số phiên tăng giá, gần như tung đồng xu. "
            "Đây là tính chất quen thuộc: chiều của lợi suất gần như không dự báo được, nhưng độ lớn của nó thì có trí nhớ dài (biến động cao kéo theo biến động cao), nên bài toán biến động là chỗ một mô hình chuỗi có cơ hội lập được kết quả."
            % (f(a["close_first"], 2), f(a["close_last"], 0), pc(a["ret_std"], 1), pc(a["up_share"], 1)))


def _fmt(task, key, ms, scale=1.0, nd=3):
    return pm(ms, nd, scale)


def t_c4_churn():
    base = J("ch4/results/baselines.json")["churn_logreg"]
    rows = [["Hồi quy logistic (sklearn)", "-", f(base["roc_auc"], 4), "-", f(base["f1"], 4), "-"]]
    for model in ["MLP"] + CELLS:
        for fw, nm in FW4:
            try:
                r = R4(fw, "churn", model)
            except FileNotFoundError:
                continue
            s = r["summary"]
            rows.append([model, nm, pm(s["roc_auc"]), pm(s["pr_auc"]), pm(s["f1"]), n(r["params"])])
    return ("Churn (ba hạt giống; ROC-AUC, PR-AUC, F1 trên 2.000 khách kiểm tra)", ["Mô hình", "Cài đặt", "ROC-AUC", "PR-AUC", "F1", "Tham số"], rows)


def _mean(fw, task, model, key):
    return R4(fw, task, model)["summary"][key][0]


def _gap(task, key, a, b):
    """Chenh lech lon nhat (qua cac o RNN/LSTM/GRU) giua hai cai dat, theo don vi do lech chuan cua cai dat b."""
    best = (-1, None, 0, 0)
    for m in CELLS:
        x, y = R4(a, task, m)["summary"][key], R4(b, task, m)["summary"][key]
        z = abs(x[0] - y[0]) / max(y[1], 1e-9)
        if z > best[0]:
            best = (z, m, x[0], y[0])
    return best


def c4_churn_result():
    out = {}
    for model in CELLS:
        v = [_mean(fw, "churn", model, "roc_auc") for fw, _ in FW4]
        out[model] = (min(v), max(v))
    mlp = [_mean(fw, "churn", "MLP", "roc_auc") for fw in ("numpy", "torch", "keras")]
    lr = J("ch4/results/baselines.json")["churn_logreg"]["roc_auc"]
    g_np = _gap("churn", "roc_auc", "numpy", "torch")
    g_k = _gap("churn", "roc_auc", "keras", "torch")
    return ("Mọi mạng nơ-ron đều vượt xa hồi quy logistic (ROC-AUC %s): RNN từ %s đến %s, LSTM từ %s đến %s, GRU từ %s đến %s giữa ba cài đặt. "
            "NumPy và PyTorch, vốn dùng cùng quy ước khởi tạo, sát nhau: chênh lệch lớn nhất ở ô %s là %s điểm ROC-AUC, tức khoảng %s lần độ lệch chuẩn giữa hạt giống. Keras khác rõ hơn: ở ô %s, Keras đạt %s so với %s của PyTorch, khoảng %s lần độ lệch chuẩn, và ba cài đặt không xếp cùng một thứ tự ở mọi ô. "
            "Đây là khác biệt thật chứ không chỉ là nhiễu hạt giống. Ứng viên giải thích tự nhiên là các mặc định của Keras (khởi tạo Glorot cho trọng số đầu vào, trực giao cho trọng số hồi quy, bias cổng quên bằng 1) khác PyTorch (khởi tạo đều trong khoảng ±1/√H cho mọi trọng số và bias); điều này nhất quán với thí nghiệm trí nhớ ở mục 4.5, nhưng tôi chưa tách riêng nguyên nhân bằng thí nghiệm đổi khởi tạo. "
            "MLP, đối chứng không hồi quy nhận cả 11 đặc trưng cùng lúc, đạt %s, %s và %s (NumPy, PyTorch, Keras), ngang hoặc nhỉnh hơn mọi ô hồi quy: cấu trúc hồi quy không mang lại gì khi dữ liệu là một bảng không có trục thời gian."
            % (f(lr, 3), f(out["RNN"][0], 3), f(out["RNN"][1], 3), f(out["LSTM"][0], 3), f(out["LSTM"][1], 3), f(out["GRU"][0], 3), f(out["GRU"][1], 3),
               g_np[1], f(100 * abs(g_np[2] - g_np[3]), 1), f(g_np[0], 1), g_k[1], f(g_k[2], 3), f(g_k[3], 3), f(g_k[0], 1), f(mlp[0], 3), f(mlp[1], 3), f(mlp[2], 3)))


def t_c4_ret():
    b = J("ch4/results/baselines.json")["ret"]
    rows = [["Naive (lợi suất = 0)", "-", "0,00", "-", f(b["Naive (loi suat 0)"]["price_rmse"], 3)],
            ["Ridge (cửa sổ 60 ngày)", "-", f(100 * b["Ridge (cua so 60 ngay)"]["r2_vs_zero"], 2), f(b["Ridge (cua so 60 ngay)"]["dir_acc"], 3), f(b["Ridge (cua so 60 ngay)"]["price_rmse"], 3)]]
    for model in CELLS:
        for fw, nm in FW4:
            s = R4(fw, "ret", model)["summary"]
            rows.append([model, nm, pm(s["r2_vs_zero"], 2, 100), pm(s["dir_acc"], 3), pm(s["price_rmse"], 3)])
    return ("AMZN, dự báo lợi suất phiên kế tiếp (736 phiên kiểm tra 2021 - 2023). R² so với naive tính bằng %; đúng chiều là tỷ lệ đoán đúng tăng/giảm; RMSE quy về giá USD",
            ["Mô hình", "Cài đặt", "R² so với naive (%)", "Đúng chiều", "RMSE giá (USD)"], rows)


def c4_ret_result():
    vals = [(fw, m, _mean(fw, "ret", m, "r2_vs_zero") * 100) for fw, _ in FW4 for m in CELLS]
    best = max(vals, key=lambda x: x[2])
    worst = min(vals, key=lambda x: x[2])
    da = [_mean(fw, "ret", m, "dir_acc") for fw, _ in FW4 for m in CELLS]
    up = J("ch4/results/baselines.json")["up_share_test"]
    return ("Không có cài đặt nào hơn dự báo naive: R² so với naive nằm trong khoảng từ %s%% đến %s%%, nghĩa là mọi ô đều kém hơn hoặc xấp xỉ bằng việc đoán lợi suất bằng 0. Tỷ lệ đoán đúng chiều từ %s đến %s, sát mức %s của việc luôn đoán \"tăng\" (tập kiểm tra có %s số phiên tăng). "
            "Đây là kết quả âm nhưng có giá trị: nó khớp với tính chất đã biết của thị trường cổ phiếu lớn (lợi suất ngày gần như không dự báo được từ quá khứ của chính nó), và việc ba cài đặt độc lập cùng cho kết quả này loại trừ khả năng nguyên nhân là lỗi mã."
            % (f(worst[2], 2), f(best[2], 2), f(min(da), 3), f(max(da), 3), f(up, 3), pc(up, 1)))


def t_c4_vol():
    b = J("ch4/results/baselines.json")["vol"]
    rows = [["Biến động 20 phiên qua", "-", f(b["Bien dong 20 ngay qua"]["r2"], 3), f(b["Bien dong 20 ngay qua"]["rmse"], 5)],
            ["Ridge (cửa sổ 60 ngày)", "-", f(b["Ridge (cua so 60 ngay)"]["r2"], 3), f(b["Ridge (cua so 60 ngay)"]["rmse"], 5)]]
    for model in CELLS:
        for fw, nm in FW4:
            r = R4(fw, "vol", model)
            s = r["summary"]
            rows.append([model, nm, pm(s["r2"], 3), pm(s["rmse"], 5)])
    return ("AMZN, dự báo biến động thực 5 phiên tới (736 phiên kiểm tra)", ["Mô hình", "Cài đặt", "R²", "RMSE"], rows)


def c4_vol_result():
    vals = [_mean(fw, "vol", m, "r2") for fw, _ in FW4 for m in CELLS]
    rid = J("ch4/results/baselines.json")["vol"]["Ridge (cua so 60 ngay)"]["r2"]
    g_np = _gap("vol", "r2", "numpy", "torch")
    g_k = _gap("vol", "r2", "keras", "torch")
    return ("Ngược với lợi suất, biến động dự báo được: R² trung bình của chín tổ hợp ô và cài đặt nằm trong khoảng %s đến %s, so với %s của hồi quy Ridge trên cùng cửa sổ 60 ngày và %s của đường cơ sở \"biến động 20 phiên qua\". Nói cách khác ô hồi quy học được thật, nhưng chỉ ngang mô hình tuyến tính: phần lớn trí nhớ của biến động là tuyến tính và Ridge đã bắt được. "
            "Giữa các cài đặt, NumPy và PyTorch sát nhau (chênh lệch lớn nhất %s ở ô %s, khoảng %s lần độ lệch chuẩn); Keras lệch xa hơn và không theo một chiều nhất quán: ở ô LSTM Keras cao hơn PyTorch (%s so với %s), còn ở ô GRU thấp hơn (%s so với %s), với khoảng cách ba đến năm lần độ lệch chuẩn giữa hạt giống. Không cài đặt nào thắng ở mọi ô, nên khác biệt giữa framework ở đây là có thật nhưng nhỏ so với khoảng cách giữa mô hình học được và đường cơ sở yếu, và không đủ để đổi kết luận. "
            "Ghép hai mô hình PyTorch (lợi suất kỳ vọng và biến động dự báo) cho khoảng dự báo 95%% có độ phủ thực tế %s trên tập kiểm tra, gần mức danh định."
            % (f(min(vals), 3), f(max(vals), 3), f(rid, 3), f(J("ch4/results/baselines.json")["vol"]["Bien dong 20 ngay qua"]["r2"], 3),
               f(abs(g_np[2] - g_np[3]), 3), g_np[1], f(g_np[0], 1), f(_mean("keras", "vol", "LSTM", "r2"), 3), f(_mean("torch", "vol", "LSTM", "r2"), 3), f(_mean("keras", "vol", "GRU", "r2"), 3), f(_mean("torch", "vol", "GRU", "r2"), 3), pc(J("ch4/results/torch__extra.json")["forecast"]["coverage_test"], 1)))


def BN(fw):
    return J("ch4/results/bench_%s.json" % fw)


def _locs():
    l_np = loc("ch4/rnn_numpy.py", ["Recurrent", "SeqNetNP", "Adam", "clip_global", "fit", "bce_grad", "mse_grad", "sigm"])
    l_pt = loc("ch4/src/nb03_rnn_pytorch.py", ["SeqNet", "fit", "predict"])
    l_k = loc("ch4/src/nb04_rnn_keras.py", ["seq_model", "fit"])
    return l_np, l_pt, l_k


def t_c4_time():
    rows = []
    B = {fw: BN(fw) for fw, _ in FW4}
    for task, lab in (("churn", "Churn (6.400 mẫu, chuỗi 11 bước)"), ("vol", "AMZN (5.359 mẫu, chuỗi 60 bước)")):
        for model in CELLS:
            t = [B[fw]["%s__%s" % (task, model)] for fw, _ in FW4]
            rows.append([lab, model] + [f(x, 2) for x in t] + [f(t[0] / t[1], 1) + "×"])
    l = _locs()
    rows.append(["Số dòng code", "mô hình + huấn luyện"] + [str(x) for x in l] + ["-"])
    return ("Thời gian một epoch (giây, trung bình ba epoch sau một epoch khởi động, đo tuần tự khi máy không chạy việc khác, batch 64, 32 đơn vị ẩn) và độ dài code",
            ["Bài toán", "Ô", "NumPy", "PyTorch", "Keras", "NumPy / PyTorch"], rows)


def c4_time_comment():
    B = {fw: BN(fw) for fw, _ in FW4}
    keys = list(B["numpy"].keys())
    r_np = {k: B["numpy"][k] / B["torch"][k] for k in keys}
    r_k = {k: B["keras"][k] / B["torch"][k] for k in keys}
    l_np, l_pt, l_k = _locs()
    return ("Thời gian một epoch của NumPy bằng %s đến %s lần PyTorch tùy ô và độ dài chuỗi; ô LSTM (nhiều phép nhân ma trận nhất mỗi bước) là chỗ khoảng cách lớn nhất (%s lần ở AMZN, %s lần ở Churn). Keras ở mức %s đến %s lần PyTorch. "
            "Mạng ở đây rất nhỏ (32 đơn vị ẩn, batch 64), mỗi phép nhân ma trận chỉ vài nghìn phần tử nên cả ba đều bị chi phí điều phối của vòng lặp theo thời gian chi phối hơn là phép tính thuần; vì vậy chênh lệch nhỏ hơn nhiều so với kỳ vọng thông thường rằng bản NumPy chậm hơn hàng chục lần. Với mạng lớn hơn hoặc batch lớn hơn khoảng cách sẽ rộng ra. "
            "Về độ dài code, bản NumPy dài %d dòng so với %d dòng của PyTorch và %d dòng của Keras (không tính dòng trống và chú thích): ta trả thêm khoảng %d dòng để tự viết ba ô tính và BPTT, đổi lại thấy rõ từng bước và kiểm chứng được gradient."
            % (f(min(r_np.values()), 1), f(max(r_np.values()), 1), f(r_np["vol__LSTM"], 1), f(r_np["churn__LSTM"], 1), f(min(r_k.values()), 1), f(max(r_k.values()), 1), l_np, l_pt, l_k, l_np - l_pt))


def t_c4_memory():
    rows = J("ch4/results/concept__memory.json")["rows"]
    cfgs = []
    for r in rows:
        if r["cau_hinh"] not in cfgs:
            cfgs.append(r["cau_hinh"])
    Ts = sorted({r["T"] for r in rows})
    out = []
    for c in cfgs:
        line = [c]
        for T in Ts:
            a = [r["acc"] for r in rows if r["cau_hinh"] == c and r["T"] == T]
            line.append(f(sum(a) / len(a), 3))
        out.append(line)
    return ("Độ chính xác khi phải nhớ dấu hiệu ở bước 0 qua T - 1 bước nhiễu (trung bình hai hạt giống; đoán ngẫu nhiên = 0,5)", ["Cấu hình"] + ["T = %d" % T for T in Ts], out)
