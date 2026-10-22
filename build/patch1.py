# -*- coding: utf-8 -*-
import re
# ---------------- tables_c3: baseline rows + comment
c = open('tables_c3.py', encoding='utf8').read()
c = c.replace('''        rows.append([nm, f(m["accuracy"]), f(m["precision"]), f(m["recall"]), f(m["f1"]), n(r["n_params"]), f(r["seconds"], 1)])
    return ("Diabetes (4.000 mẫu kiểm tra, 30 epoch, một hạt giống)"''', '''        rows.append([nm, f(m["accuracy"]), f(m["precision"]), f(m["recall"]), f(m["f1"]), n(r["n_params"]), f(r["seconds"], 1)])
    for k, m in BL()["diabetes"].items():
        rows.append(["(đối chiếu) " + k, f(m["accuracy"]), f(m["precision"]), f(m["recall"]), f(m["f1"]), "-", f(m["seconds"], 1)])
    return ("Diabetes (4.000 mẫu kiểm tra, 30 epoch, một hạt giống; hai hàng cuối là mô hình không dùng mạng nơ-ron trên cùng phép chia)"''')
c = c.replace('''        rows.append([nm, f(m["mae"], 3), f(m["rmse"], 3), f(m["r2"]), f(m["r2_log"]), n(r["n_params"]), f(r["seconds"], 1)])
    return ("House Price (6.045 mẫu kiểm tra, 40 epoch, một hạt giống; MAE và RMSE tính bằng tỷ đồng)"''', '''        rows.append([nm, f(m["mae"], 3), f(m["rmse"], 3), f(m["r2"]), f(m["r2_log"]), n(r["n_params"]), f(r["seconds"], 1)])
    for k, m in BL()["housing"].items():
        rows.append(["(đối chiếu) " + k, f(m["mae"], 3), f(m["rmse"], 3), f(m["r2"]), f(m["r2_log"]), "-", f(m["seconds"], 1)])
    return ("House Price (6.045 mẫu kiểm tra, 40 epoch, một hạt giống; MAE và RMSE tính bằng tỷ đồng; hai hàng cuối là mô hình không dùng mạng nơ-ron)"''')
c = c.replace('''        rows.append([nm, f(m["accuracy"]), f(m["macro_f1"]), f(pcs[0], 3), f(pcs[1], 3), f(pcs[2], 3), f(r["seconds"], 1)])
    return ("Comments (3.672 mẫu kiểm tra, 4 epoch, một hạt giống)"''', '''        rows.append([nm, f(m["accuracy"]), f(m["macro_f1"]), f(pcs[0], 3), f(pcs[1], 3), f(pcs[2], 3), f(r["seconds"], 1)])
    for k, m in BL()["comments"].items():
        pcs = [x["f1"] for x in m["per_class"]]
        rows.append(["(đối chiếu) " + k, f(m["accuracy"]), f(m["macro_f1"]), f(pcs[0], 3), f(pcs[1], 3), f(pcs[2], 3), f(m["seconds"], 1)])
    return ("Comments (3.672 mẫu kiểm tra, 4 epoch, một hạt giống; hàng cuối là hồi quy logistic trên túi từ, không dùng mạng nơ-ron)"''')
c = c.replace('def c3(key, ds="comments"):', '''def BL():
    return J("ch3/results/baselines.json")


def c3(key, ds="comments"):''')
c += '''

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
'''
open('tables_c3.py', 'w', encoding='utf8').write(c)

# ---------------- chapter 3 md
p = 'chapters/03_chuong3.md'
t = open(p, encoding='utf8').read()
a = t.index("**CNN trên dữ liệu bảng chưa chắc là lựa chọn tốt.**")
t = t[:a] + """**CNN trên dữ liệu bảng chưa chắc là lựa chọn tốt.** Các hàng "(đối chiếu)" ở mục 3.4 cho bằng chứng trực tiếp trên đúng các phép chia của chương này. {{V c3_baseline_comment()}} Nói cách khác, Conv1D có cơ sở cấu trúc rõ nhất ở văn bản, vì cụm từ liền kề thật sự mang nghĩa, nhưng ở đây ngay cả chỗ đó nó cũng chưa vượt được đường cơ sở đơn giản; ở dữ liệu bảng, tám hay mười chín đặc trưng không có thứ tự không gian để bộ lọc trượt khai thác, và kết quả CNN chủ yếu phản ánh khả năng phi tuyến chung của mạng nhiều tầng. Bài học chọn kiến trúc theo cấu trúc dữ liệu cũng là kết luận của mục 3.5 và là cầu nối sang Chương 4: khi dữ liệu có thứ tự thời gian, ta cần một kiến trúc nhớ được quá khứ.
"""
open(p, 'w', encoding='utf8').write(t)

# ---------------- tables_c4
d = open('tables_c4.py', encoding='utf8').read()
d = d.replace('rows.append([model, nm, pm(s["roc_auc"]), pm(s["pr_auc"]), pm(s["f1"]), n(r["params"]), f(sec(fw, "churn", model), 1)])', 'rows.append([model, nm, pm(s["roc_auc"]), pm(s["pr_auc"]), pm(s["f1"]), n(r["params"])])')
d = d.replace('rows = [["Hồi quy logistic (sklearn)", "-", f(base["roc_auc"], 4), "-", f(base["f1"], 4), "-", "-"]]', 'rows = [["Hồi quy logistic (sklearn)", "-", f(base["roc_auc"], 4), "-", f(base["f1"], 4), "-"]]')
d = d.replace('["Mô hình", "Cài đặt", "ROC-AUC", "PR-AUC", "F1", "Tham số", "Giây/lần"], rows)', '["Mô hình", "Cài đặt", "ROC-AUC", "PR-AUC", "F1", "Tham số"], rows)')
d = d.replace('rows.append([model, nm, pm(s["r2"], 3), pm(s["rmse"], 5), f(sec(fw, "vol", model), 1)])', 'rows.append([model, nm, pm(s["r2"], 3), pm(s["rmse"], 5)])')
d = d.replace('''rows = [["Biến động 20 phiên qua", "-", f(b["Bien dong 20 ngay qua"]["r2"], 3), f(b["Bien dong 20 ngay qua"]["rmse"], 5), "-"],
            ["Ridge (cửa sổ 60 ngày)", "-", f(b["Ridge (cua so 60 ngay)"]["r2"], 3), f(b["Ridge (cua so 60 ngay)"]["rmse"], 5), "-"]]''', '''rows = [["Biến động 20 phiên qua", "-", f(b["Bien dong 20 ngay qua"]["r2"], 3), f(b["Bien dong 20 ngay qua"]["rmse"], 5)],
            ["Ridge (cửa sổ 60 ngày)", "-", f(b["Ridge (cua so 60 ngay)"]["r2"], 3), f(b["Ridge (cua so 60 ngay)"]["rmse"], 5)]]''')
d = d.replace('["Mô hình", "Cài đặt", "R²", "RMSE", "Giây/lần"], rows)', '["Mô hình", "Cài đặt", "R²", "RMSE"], rows)')

a = d.index("def c4_churn_result")
b = d.index("def t_c4_ret")
d = d[:a] + '''def _gap(task, key, a, b):
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
    mlp = [_mean(fw, "churn", "MLP", "roc_auc") for fw in ("torch", "keras")]
    lr = J("ch4/results/baselines.json")["churn_logreg"]["roc_auc"]
    g_np = _gap("churn", "roc_auc", "numpy", "torch")
    g_k = _gap("churn", "roc_auc", "keras", "torch")
    return ("Mọi mạng nơ-ron đều vượt xa hồi quy logistic (ROC-AUC %s): RNN từ %s đến %s, LSTM từ %s đến %s, GRU từ %s đến %s giữa ba cài đặt. "
            "NumPy và PyTorch, vốn dùng cùng quy ước khởi tạo, sát nhau: chênh lệch lớn nhất ở ô %s là %s điểm ROC-AUC, tức khoảng %s lần độ lệch chuẩn giữa hạt giống. Keras khác rõ hơn: ở ô %s, Keras đạt %s so với %s của PyTorch, khoảng %s lần độ lệch chuẩn, và ba cài đặt không xếp cùng một thứ tự ở mọi ô. "
            "Đây là khác biệt thật chứ không chỉ là nhiễu hạt giống. Ứng viên giải thích tự nhiên là các mặc định của Keras (khởi tạo Glorot cho trọng số đầu vào, trực giao cho trọng số hồi quy, bias cổng quên bằng 1) khác PyTorch (khởi tạo đều trong khoảng ±1/√H cho mọi trọng số và bias); điều này nhất quán với thí nghiệm trí nhớ ở mục 4.5, nhưng tôi chưa tách riêng nguyên nhân bằng thí nghiệm đổi khởi tạo. "
            "MLP, đối chứng không hồi quy nhận cả 11 đặc trưng cùng lúc, đạt %s và %s (PyTorch, Keras), ngang hoặc nhỉnh hơn mọi ô hồi quy: cấu trúc hồi quy không mang lại gì khi dữ liệu là một bảng không có trục thời gian."
            % (f(lr, 3), f(out["RNN"][0], 3), f(out["RNN"][1], 3), f(out["LSTM"][0], 3), f(out["LSTM"][1], 3), f(out["GRU"][0], 3), f(out["GRU"][1], 3),
               g_np[1], f(100 * abs(g_np[2] - g_np[3]), 1), f(g_np[0], 1), g_k[1], f(g_k[2], 3), f(g_k[3], 3), f(g_k[0], 1), f(mlp[0], 3), f(mlp[1], 3)))


''' + d[b:]
a = d.index("def c4_vol_result")
b = d.index("def BN")
d = d[:a] + '''def c4_vol_result():
    vals = [_mean(fw, "vol", m, "r2") for fw, _ in FW4 for m in CELLS]
    rid = J("ch4/results/baselines.json")["vol"]["Ridge (cua so 60 ngay)"]["r2"]
    g_np = _gap("vol", "r2", "numpy", "torch")
    g_k = _gap("vol", "r2", "keras", "torch")
    return ("Ngược với lợi suất, biến động dự báo được: R² trung bình của chín tổ hợp ô và cài đặt nằm trong khoảng %s đến %s, so với %s của hồi quy Ridge trên cùng cửa sổ 60 ngày và %s của đường cơ sở \\"biến động 20 phiên qua\\". Nói cách khác ô hồi quy học được thật, nhưng chỉ ngang mô hình tuyến tính: phần lớn trí nhớ của biến động là tuyến tính và Ridge đã bắt được. "
            "Giữa các cài đặt, NumPy và PyTorch sát nhau (chênh lệch lớn nhất %s ở ô %s, khoảng %s lần độ lệch chuẩn); Keras lệch xa hơn và không theo một chiều nhất quán: ở ô %s Keras đạt %s so với %s của PyTorch (khoảng %s lần độ lệch chuẩn), nhưng ở ô GRU lại thấp hơn PyTorch. Không cài đặt nào thắng ở mọi ô, nên khác biệt giữa framework ở đây là có thật nhưng nhỏ so với khoảng cách giữa mô hình học được và đường cơ sở yếu, và không đủ để đổi kết luận. "
            "Ghép hai mô hình PyTorch (lợi suất kỳ vọng và biến động dự báo) cho khoảng dự báo 95%% có độ phủ thực tế %s trên tập kiểm tra, gần mức danh định."
            % (f(min(vals), 3), f(max(vals), 3), f(rid, 3), f(J("ch4/results/baselines.json")["vol"]["Bien dong 20 ngay qua"]["r2"], 3),
               f(abs(g_np[2] - g_np[3]), 3), g_np[1], f(g_np[0], 1), g_k[1], f(g_k[2], 3), f(g_k[3], 3), f(g_k[0], 1), pc(J("ch4/results/torch__extra.json")["forecast"]["coverage_test"], 1)))


''' + d[b:]
open('tables_c4.py', 'w', encoding='utf8').write(d)
print("patched")
