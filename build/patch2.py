# -*- coding: utf-8 -*-
p = 'chapters/04_chuong4.md'
t = open(p, encoding='utf8').read()
a = t.index("**Ba cài đặt cho cùng kết luận.**")
b = t.index("**RNN không tự động thắng đối chứng đơn giản.**")
t = t[:a] + """**Ba cài đặt cho cùng kết luận định tính, nhưng không cho cùng con số.** Trên mọi bài toán, ba cài đặt dẫn đến cùng các kết luận: mạng vượt hồi quy logistic trên Churn, không hơn dự báo naive trên lợi suất, ngang hồi quy Ridge trên biến động. Về con số, NumPy và PyTorch (cùng quy ước khởi tạo) sát nhau, thường trong khoảng một lần độ lệch chuẩn giữa hạt giống. Keras thì lệch xa hơn ở một số ô (LSTM trên Churn khoảng ba lần độ lệch chuẩn, GRU và LSTM trên biến động ba đến năm lần, theo hai chiều ngược nhau), nghĩa là khác biệt giữa framework ở đó là thật, không phải nhiễu hạt giống. Bài toán lợi suất thì mọi khác biệt đều nằm trong nhiễu. Nguyên nhân khả dĩ là mặc định khởi tạo khác nhau của Keras và PyTorch (mục 4.5); tôi chưa kiểm chứng riêng. Việc bản scratch tự viết, với gradient đã kiểm chứng với `torch.nn`, cho kết quả sát PyTorch là bằng chứng mạnh rằng các kết luận không do lỗi cài đặt.

""" + t[b:]
open(p, 'w', encoding='utf8').write(t)

# Chuong 2: them mau Diabetes trong chuong, dich nhan bai toan
p = 'chapters/02_chuong2.md'
t = open(p, encoding='utf8').read()
t = t.replace("Năm hồ sơ đầu đã in ở mục 1.4 của Chương 1.", "Năm hồ sơ đầu của bộ dữ liệu:\n\n{{TABL t_c1_diabetes_sample}}")
open(p, 'w', encoding='utf8').write(t)

d = open('tables_c2.py', encoding='utf8').read()
d = d.replace('rows.append([nm, " → ".join(map(str, c["sizes"])), n(cfg[k]["runs"][0]["params"]), c["task"],', 'rows.append([nm, " → ".join(map(str, c["sizes"])), n(cfg[k]["runs"][0]["params"]), {"binary": "nhị phân", "regression": "hồi quy", "multiclass": "3 lớp"}[c["task"]],')
open('tables_c2.py', 'w', encoding='utf8').write(d)

# bieu dong: tach ro hai o LSTM va GRU
d = open('tables_c4.py', encoding='utf8').read()
old = d[d.index('"Giữa các cài đặt, NumPy và PyTorch sát nhau'):d.index('"Ghép hai mô hình PyTorch')]
new = '''"Giữa các cài đặt, NumPy và PyTorch sát nhau (chênh lệch lớn nhất %s ở ô %s, khoảng %s lần độ lệch chuẩn); Keras lệch xa hơn và không theo một chiều nhất quán: ở ô LSTM Keras cao hơn PyTorch (%s so với %s), còn ở ô GRU thấp hơn (%s so với %s), với khoảng cách ba đến năm lần độ lệch chuẩn giữa hạt giống. Không cài đặt nào thắng ở mọi ô, nên khác biệt giữa framework ở đây là có thật nhưng nhỏ so với khoảng cách giữa mô hình học được và đường cơ sở yếu, và không đủ để đổi kết luận. "
            '''
d = d.replace(old, new)
d = d.replace('f(abs(g_np[2] - g_np[3]), 3), g_np[1], f(g_np[0], 1), g_k[1], f(g_k[2], 3), f(g_k[3], 3), f(g_k[0], 1), pc(J("ch4/results/torch__extra.json")["forecast"]["coverage_test"], 1)))',
              'f(abs(g_np[2] - g_np[3]), 3), g_np[1], f(g_np[0], 1), f(_mean("keras", "vol", "LSTM", "r2"), 3), f(_mean("torch", "vol", "LSTM", "r2"), 3), f(_mean("keras", "vol", "GRU", "r2"), 3), f(_mean("torch", "vol", "GRU", "r2"), 3), pc(J("ch4/results/torch__extra.json")["forecast"]["coverage_test"], 1)))')
open('tables_c4.py', 'w', encoding='utf8').write(d)
print("ok")
