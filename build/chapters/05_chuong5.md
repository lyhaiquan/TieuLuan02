# Chương 5. Knowledge Graph và RAG cho miền tài chính

## 5.1. Đặt vấn đề

Bốn chương trước đi theo một mạch: mạng nơ-ron học từ dữ liệu có sẵn nhãn. Chương này đổi loại bài toán. Hệ thống không còn dự đoán một con số mà phải **trả lời một câu hỏi bằng ngôn ngữ tự nhiên**, dựa trên tri thức nằm rải rác trong tài liệu và bảng số. Đó là bài toán của RAG (Retrieval-Augmented Generation) kết hợp với đồ thị tri thức (Knowledge Graph).

Điểm xuất phát của chương là một nhận xét về cách làm thường gặp. Hệ RAG tiêu chuẩn chỉ biết một thao tác: tìm những đoạn văn bản giống câu hỏi nhất rồi trả lời dựa trên chúng. Với dữ liệu tài chính, cách đó vấp phải ba vấn đề mà không kỹ thuật truy hồi nào giải quyết được, vì chúng nằm ở chỗ khác.

Thứ nhất, **phần lớn tri thức tài chính nằm trong bảng số**, mà bảng số thì không truy hồi theo độ tương đồng văn bản được. Một bảng 2.480 dòng giá vàng không có "đoạn nào giống câu hỏi".

Thứ hai, **quan hệ giữa các tài sản thay đổi theo thời gian**. Vàng và đồng đô thường đi ngược nhau, nhưng có giai đoạn cùng tăng. Một con số tương quan duy nhất cho cả giai đoạn là trung bình hoá mất hiện tượng.

Thứ ba, **tin tức tài chính đầy khẳng định nhân quả chưa được kiểm chứng**. Tiêu đề "vàng giảm vì lợi suất tăng" là lời của người viết, không phải dữ kiện đo được. Hệ RAG thường lấy nguyên câu đó làm câu trả lời và trình bày nó như kết luận.

Chương này xây một hệ hỏi đáp xử lý cả ba vấn đề, trên hai miền dữ liệu là vàng và chứng khoán, rồi đo xem mỗi thay đổi đem lại gì so với cách làm thông thường.

## 5.2. Cơ sở lý thuyết

### 5.2.1. Đồ thị tri thức

Một **đồ thị tri thức** (Knowledge Graph) là cách biểu diễn tri thức dưới dạng đồ thị có hướng, có nhãn trên cạnh. Đơn vị cơ bản là một **bộ ba** (triple) gồm chủ thể, vị từ và đối tượng:

$$ G = \{(h, r, t) \mid h, t \in \mathcal{E},\; r \in \mathcal{R}\} $$

trong đó $\mathcal{E}$ là tập thực thể (nút) và $\mathcal{R}$ là tập loại quan hệ (nhãn cạnh). Ví dụ `(AAPL, belongs_to, Technology)` là một bộ ba: nút AAPL, nhãn `belongs_to`, nút Technology. Giữa hai nút có thể có nhiều cạnh khác nhãn, nên về hình thức đây là một **đa đồ thị có hướng gắn nhãn**, không phải đồ thị đơn.

Có hai quy ước cài đặt phổ biến. Mô hình **RDF** coi mọi thứ là bộ ba và truy vấn bằng SPARQL; mô hình **property graph** (Neo4j, NetworkX) cho phép gắn thuộc tính trực tiếp lên nút và cạnh. Chương này dùng mô hình thứ hai, vì mỗi cạnh tương quan cần mang theo ít nhất ba thuộc tính là hệ số, khoảng thời gian và số phiên dùng để tính.

Điều khiến đồ thị tri thức khác một bảng quan hệ là **cạnh gián tiếp**: từ $h$ tới $t$ có thể không có cạnh trực tiếp nhưng tồn tại đường đi qua các nút trung gian. Nhờ đó trả lời được những câu hỏi mà không tài liệu đơn lẻ nào chứa đủ dữ kiện — gọi là bài toán **multi-hop**. Bậc của một nút và độ dài đường đi vì thế là hai đại lượng quyết định đồ thị có hữu ích hay không; một đồ thị 18 nút với bậc trung bình nhỏ gần như không có đường đi dài hơn một bước để khai thác, và mục 5.5.6 đo đúng hệ quả đó.

Về cách xây, có hai nguồn cạnh khác nhau về bản chất. Cạnh **tra cứu** lấy từ dữ liệu tham chiếu đã có cấu trúc, độ tin cậy coi như chắc chắn. Cạnh **trích xuất** rút từ văn bản tự do bằng nhận dạng thực thể (NER), liên kết thực thể (entity linking) rồi trích quan hệ; cạnh loại này mang sai số của bộ trích, và trong tài chính còn mang thêm một vấn đề nữa là nội dung câu được trích chỉ là **lời khẳng định của người viết** chứ không phải sự kiện đo được. Phần lớn công việc của chương này nằm ở chỗ đó.

### 5.2.2. RAG

Mô hình ngôn ngữ lưu tri thức trong trọng số, nên không cập nhật được sau khi huấn luyện và không chỉ ra được nguồn của câu trả lời. **RAG** (Retrieval-Augmented Generation) [21] tách tri thức ra khỏi trọng số: giữ một kho tài liệu bên ngoài, với mỗi câu hỏi thì truy hồi vài tài liệu liên quan rồi mới sinh câu trả lời dựa trên chúng. Quy trình gồm ba khâu.

**Lập chỉ mục.** Kho tài liệu được cắt thành đoạn, mỗi đoạn $d$ được một bộ mã hoá chuyển thành vectơ $\mathbf{v}_d \in \mathbb{R}^n$.

**Truy hồi.** Câu hỏi $q$ cũng được mã hoá, rồi xếp hạng tài liệu theo độ tương đồng. Với vectơ dày (embedding) thì dùng cosine:

$$ \operatorname{sim}(q, d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\lVert \mathbf{v}_q \rVert \, \lVert \mathbf{v}_d \rVert} $$

Với vectơ thưa thì dùng trọng số từ khoá TF-IDF, tức tần suất từ trong tài liệu nhân với độ hiếm của từ trong cả kho, $\mathrm{tfidf}(w, d) = \mathrm{tf}(w, d) \cdot \log \frac{N}{\mathrm{df}(w)}$. Hai cách bắt hai loại tín hiệu khác nhau: embedding bắt được câu hỏi diễn đạt khác từ ngữ tài liệu, TF-IDF bắt được tên riêng và con số mà embedding thường làm nhoè. Vì vậy thực hành phổ biến là **truy hồi lai**, cộng hai điểm sau khi chuẩn hoá — đây chính là cách chấm điểm trong `TimeScopedRetriever` ở mục 5.4.3.

**Sinh câu trả lời.** $k$ tài liệu điểm cao nhất được ghép vào lời nhắc để mô hình sinh câu trả lời kèm dẫn nguồn. Hệ trong chương này dừng ở mức **trích rút**: chọn và ghép câu có sẵn, không có mô hình ngôn ngữ viết lại, nên không thể bịa ra con số — đánh đổi được nêu ở mục 5.7.

Đánh giá RAG tách làm hai phần. Phần truy hồi đo bằng `context_recall`, tỉ lệ tài liệu đúng nằm trong top-$k$; phần sinh đo bằng độ khớp với đáp án. Tách như vậy là cần thiết vì một hệ có thể trả lời sai do truy hồi trượt, hoặc truy hồi đúng mà diễn giải sai, và hai lỗi này sửa bằng hai cách khác nhau.

### 5.2.3. Ghép đồ thị tri thức vào RAG

Hạn chế của RAG thuần là nó chỉ biết độ tương đồng văn bản. Câu hỏi cần nối hai dữ kiện ở hai tài liệu khác nhau thì không tài liệu nào "giống câu hỏi" đủ để được chọn. **GraphRAG** [22] khắc phục bằng cách dùng đồ thị làm cầu, theo hai hướng chính.

Hướng thứ nhất là **mở rộng truy hồi**: nhận ra thực thể trong câu hỏi, lấy các nút lân cận trong đồ thị, rồi cộng điểm cho những tài liệu nhắc tới các nút đó. Công thức điểm trở thành

$$ \operatorname{score}(q, d) = \operatorname{sim}(q, d) + \lambda \cdot \lvert \mathcal{N}(q) \cap \mathcal{E}(d) \rvert $$

với $\mathcal{N}(q)$ là lân cận của các thực thể trong câu hỏi và $\mathcal{E}(d)$ là thực thể xuất hiện trong tài liệu. Hướng thứ hai là **tóm tắt theo cộng đồng**: phân cụm đồ thị, tóm tắt từng cụm, rồi trả lời câu hỏi tổng quát bằng các bản tóm tắt thay vì bằng tài liệu gốc.

Cả hai hướng đều coi đồ thị là công cụ **tìm thêm** tài liệu. Chương này dùng hướng thứ nhất ở miền y văn (mục 5.5.6) nhưng đóng góp chính lại nằm ở một hướng thứ ba: dùng đồ thị để **kiểm chứng** câu trả lời. Nếu một cạnh `claims_cause` trích từ tin tức hàm ý hai chuỗi giá đi ngược nhau, thì cạnh `correlates_with` tính từ chính dữ liệu giá sẽ nói lời khẳng định đó có được dữ liệu chống đỡ hay không. Điều này chỉ làm được khi cạnh đồ thị **có kiểu và có nguồn gốc rõ ràng**, nên hai mục 5.4.2 và 5.4.4 gắn với nhau chặt chẽ.

## 5.3. Dữ liệu

{{TABL t_c5_data}}

Có một giới hạn phải nói ngay vì nó định hình toàn bộ thiết kế thí nghiệm. Tin tức lấy qua Google News RSS, mà nguồn này chỉ trả về tin gần đây. Kết quả là kho tin lệch nặng về hiện tại: với miền vàng, 238 trên 632 tiêu đề thuộc riêng tháng 10/2026, còn trước năm 2026 chỉ có 16 tiêu đề. Vì vậy hệ được chia đôi theo thời gian: **phần số chạy trên toàn bộ 2020–2026**, còn **phần diễn giải và kiểm chứng chỉ chạy trên năm 2026**, nơi tin đủ dày để cắt theo tháng. Mọi câu hỏi đánh giá về diễn giải đều nằm trong khoảng có tin; hỏi ra ngoài khoảng đó là đo lỗ hổng dữ liệu chứ không đo hệ thống.

## 5.4. Cơ chế hoạt động

Hệ gồm bốn thay đổi so với một hệ RAG thông thường. Mỗi mục dưới đây nêu nguyên lý, trích code thật, rồi giải thích code đó làm gì.

### 5.4.1. Bảng số thì truy vấn, văn bản mới truy hồi

**Nguyên lý.** Cách thường gặp khi muốn cho RAG đọc bảng số là biến mỗi dòng hoặc mỗi nhóm dòng thành một câu tiếng Anh rồi ném vào kho truy hồi, kiểu `"GOLD in 2020: price return +24.3%, volatility 21.7%"`. Cách này chạy được nhưng chỉ trả lời được đúng những phép tính đã viết sẵn thành câu. Hỏi theo quý, hỏi tương quan, hỏi một khoảng bất kỳ là chịu, vì không có câu nào tương ứng để tìm.

Thay vào đó, hệ phân loại câu hỏi trước rồi mới chọn đường xử lý: câu hỏi số đi sang bộ tính chạy thẳng trên bảng giá, câu hỏi diễn giải và nhân quả mới đi sang truy hồi văn bản.

{{CODE ch5/fin_core.py | parse_question}}

**Giải thích.** Hàm nhận câu hỏi và trả về bốn thứ. `entities` là danh sách thực thể nhận ra được, lọc lấy những thực thể có chuỗi giá. `metric` dò từ khoá độ đo; thứ tự duyệt trong `METRIC_WORDS` là có chủ ý vì câu "tương quan giữa vàng và lợi suất trái phiếu" chứa từ khoá của cả hai độ đo, nên độ đo cụ thể hơn phải đứng trước. `period` gọi sang `parse_period` để rút khoảng thời gian, hỗ trợ cả "năm 2022", "quý 2/2023" lẫn "tháng 9/2026"; thiếu bước này thì mọi câu hỏi đều chạy trên toàn giai đoạn và câu trả lời cho hai năm khác nhau sẽ giống hệt nhau. `kind` quyết định nhánh xử lý: có độ đo thì là câu hỏi số, có từ để hỏi nguyên nhân thì là câu hỏi nhân quả, còn lại là diễn giải.

{{CODE ch5/fin_core.py | answer_numeric}}

**Giải thích.** Nhánh số không truy hồi gì cả. Nó lấy khoảng thời gian đã phân tích được (mặc định là toàn giai đoạn nếu câu hỏi không nêu), lấy tài sản đầu tiên nhận ra được, rồi gọi `compute` để tính thẳng trên bảng. Trường hợp câu hỏi tương quan cần hai tài sản nên có kiểm tra riêng. Kết quả là con số chính xác tuyệt đối chứ không phải con số rút từ một câu văn.

Có một chi tiết tính toán nhỏ nhưng quan trọng nằm ở hàm biến đổi chuỗi:

{{CODE ch5/fin_core.py | returns}}

**Giải thích.** Tương quan phải tính trên biến động ngày chứ không trên mức giá, vì hai chuỗi cùng có xu hướng tăng sẽ cho tương quan gần 1 dù không liên quan gì nhau. Nhưng không phải chuỗi nào cũng lấy lợi suất log. US10Y là **lãi suất** chứ không phải giá: khi lãi suất ở mức 0,5% mà tăng lên 0,6% thì theo log là +18%, trong khi thực chất chỉ là 10 điểm cơ bản, nên tương quan tính theo log sẽ méo nặng ở giai đoạn lãi suất thấp. Quy ước trong tài chính là lấy chênh lệch, và dòng `if col in RATE_SERIES` làm đúng việc đó. Dòng `s = s[s > 0]` xử lý một ca có thật: ngày 20/4/2020 hợp đồng dầu WTI xuống giá âm (−37,63), và `log` của số âm trả về `NaN` làm hỏng toàn bộ cột.

### 5.4.2. Cạnh đồ thị có kiểu, tách theo nguồn gốc

**Nguyên lý.** Đồ thị tri thức thường chỉ có một loại cạnh "có liên quan". Với dữ liệu tài chính, gộp như vậy là trộn lẫn ba thứ có độ tin cậy hoàn toàn khác nhau: một quan hệ tra cứu được trong dữ liệu tham chiếu, một hệ số tính ra từ giá, và một câu nhà báo viết. Hệ này tách hẳn ba loại.

{{TABL t_c5_edges}}

{{FIG ch5_graph.png | Ba loại cạnh của đồ thị. Trái: miền vàng ở lát quý 2/2026, cạnh liền là tương quan tính từ giá (xanh dương là dương, đỏ là âm, độ dày theo trị tuyệt đối), cạnh nét đứt là khẳng định trích từ tin. Phải: cạnh belongs_to của miền chứng khoán}}

**Đọc hình.** Hình trên là đồ thị thật, dựng từ `kg_corr_edges.csv` và `kg_claim_edges.csv` chứ không phải sơ đồ minh hoạ. Ba loại cạnh được vẽ bằng ba kiểu nét khác nhau, và đó chính là điểm của mục này.

Cạnh liền ở panel trái là `correlates_with` trong một lát thời gian duy nhất là quý 2/2026. Màu mã hoá dấu, độ dày mã hoá độ mạnh: vàng–bạc +0,82 là cạnh dày nhất, vàng–USD −0,61 và vàng–VIX −0,45 là hai cạnh âm đậm. Hình chỉ vẽ bảy cạnh chạm tới nút GOLD cho dễ nhìn, còn lát quý này thật ra có 28 cạnh vì mọi cặp trong tám chuỗi giá đều được tính. Trên toàn bộ {{V n(C5('kg_summary.json')['so_lat_thoi_gian'])}} lát, đồ thị có {{V n(C5('kg_summary.json')['canh_correlates_with'])}} cạnh loại này — tức cùng một cặp tài sản xuất hiện nhiều lần với hệ số khác nhau, và sang quý khác hình vẽ sẽ đổi màu. Đây là hệ quả trực tiếp của việc cắt cạnh theo thời gian ở mục 5.4.3; vẽ một cạnh duy nhất cho cả sáu năm thì cặp vàng–S&P 500 sẽ hiện ra mảnh và gần như trung tính, che mất biên độ 0,950 mà mục 5.5.4 đo được.

Cạnh nét đứt là `claims_cause`, trích từ tiêu đề tin trong tháng 6/2026. Chúng không cùng loại với cạnh liền dù cũng nối hai nút: cạnh liền là đại lượng tính được từ dữ liệu, cạnh nét đứt là lời của người viết. Bốn mũi tên nét đứt chĩa vào GOLD cho thấy hiện tượng mà mục 5.5.3 đo bằng số: ba nguồn nguyên nhân được báo chí viện tới nhiều nhất là Fed, địa chính trị và lạm phát đều **không có chuỗi giá** nên vẽ bằng hình chữ nhật xám và không kiểm chứng được, trong khi chỉ một tin quy cho đồng đô là đối chiếu được với cạnh tương quan tương ứng, và mũi tên xanh lá là của tin ấy.

Chỗ này làm rõ vì sao ba loại cạnh không được trộn. Nút USD nối với GOLD bằng **hai cạnh cùng lúc**: một cạnh `correlates_with` mang giá trị −0,61 tính từ giá, và một cạnh `claims_cause` nói nhà báo cho rằng đồng đô là nguyên nhân. Nếu đồ thị chỉ có một loại cạnh "có liên quan" thì hai thông tin này nhập làm một và không còn cách nào hỏi câu hỏi trung tâm của chương: lời khẳng định kia có được dữ liệu chống đỡ không. Tách ra thì câu hỏi ấy trở thành một phép so sánh giữa hai cạnh, và đó đúng là việc `verify_claim` làm ở mục 5.4.4.

Panel phải là loại cạnh thứ ba, `belongs_to`, lấy từ dữ liệu tham chiếu nên không cần kiểm chứng và không gắn với thời gian. Hình vẽ hai mã trong năm mã, mỗi mã ba cạnh. Nút `Vietnam` được hai mã dùng chung, và đó là ví dụ nhỏ nhất của đường đi gián tiếp đã nói ở mục 5.2.1: từ FPT tới VCB không có cạnh trực tiếp nhưng có đường đi hai bước qua nút quốc gia. Với đồ thị vàng chỉ {{V n(C5('kg_summary.json')['so_nut'])}} nút thì những đường đi như vậy quá ít và quá ngắn để khai thác, và mục 5.5.6 cho thấy hệ quả đo được: mở rộng truy hồi bằng đồ thị giúp ở miền y văn nhưng không giúp ở miền vàng.

{{CODE ch5/fin_core.py | belongs_to_edges}}

**Giải thích.** Loại cạnh đơn giản nhất, lấy từ `company_info.json`: mỗi mã cổ phiếu sinh ra ba cạnh tới ngành, nhóm ngành và quốc gia. Đây là sự thật tra cứu được, không cần kiểm chứng.

{{CODE ch5/fin_core.py | correlation_edges}}

**Giải thích.** Loại cạnh thứ hai được tính từ giá, và điểm mới nằm ở `pd.Grouper(freq=freq)`: thay vì một hệ số cho cả giai đoạn, mỗi quý có cạnh riêng. Vòng lặp hai lớp duyệt mọi cặp cột, gọi `metric_correlation` trên từng lát thời gian, và chỉ giữ lát nào đủ `min_days` phiên. Mỗi cạnh mang theo `n_days` để người đọc biết con số dựa trên bao nhiêu quan sát; một hệ số tính trên 9 phiên và một hệ số tính trên 63 phiên không đáng tin như nhau.

### 5.4.3. Thời gian là công dân hạng nhất

**Nguyên lý.** Câu hỏi diễn giải luôn gắn với một khoảng thời gian. Nếu truy hồi không lọc theo khoảng đó, hệ có thể trả lời câu hỏi về tháng 3 bằng một bài phân tích viết tháng 10 nhìn lại. Câu trả lời khi ấy nghe rất sắc sảo nhưng nó dùng thông tin **chưa tồn tại** ở thời điểm được hỏi. Trong tài chính hiện tượng này có tên riêng là rò rỉ nhìn trước, và nó làm mọi đánh giá trở nên vô nghĩa.

{{CODE ch5/fin_core.py | TimeScopedRetriever}}

**Giải thích.** Lớp này giữ nguyên cách chấm điểm lai của RAG thông thường (embedding cộng TF-IDF) nhưng thêm một bước trước đó: `mask` loại bỏ mọi tài liệu nằm ngoài cửa sổ thời gian **trước khi** xếp hạng. Đặt lọc sau khi xếp hạng là không tương đương, vì khi ấy top-k đã bị tài liệu ngoài cửa sổ chiếm chỗ. Tham số `time_filter=False` giữ lại để đo chính tác hại của việc bỏ lọc, kết quả ở mục 5.5.2.

Một vấn đề nữa lộ ra khi thử hệ: kho tin là tiếng Anh còn câu hỏi là tiếng Việt. Kiểm tra cho thấy token duy nhất của câu hỏi lọt vào từ vựng TF-IDF là chữ số năm, còn MiniLM là mô hình tiếng Anh nên véc-tơ của câu tiếng Việt gần như không mang nghĩa. Hệ quả là bộ truy hồi xếp hạng gần như ngẫu nhiên.

{{CODE ch5/fin_core.py | expand_query}}

**Giải thích.** Hàm bắc cầu hai ngôn ngữ bằng cách thêm từ khoá tiếng Anh vào truy vấn, lấy từ hai nguồn: tên tiếng Anh của các thực thể nhận ra được qua `link_entities`, và bảng `VI_EN` ánh xạ các từ chỉ hướng và ý định ("giảm" → *falls drops lower*, "vì sao" → *why*). Đây là giải pháp rẻ; cách căn bản hơn là dùng mô hình nhúng đa ngữ, nhưng cách này đủ để đo được tác dụng và giải thích được từng dòng.

### 5.4.4. Đồ thị dùng để kiểm chứng, không chỉ để mở rộng

**Nguyên lý.** Đây là điểm khác biệt chính của chương. Gần như mọi hệ GraphRAG dùng đồ thị để **mở rộng truy hồi**: tìm thêm tài liệu nhắc tới thực thể lân cận. Hệ này làm thêm chiều ngược lại, dùng đồ thị để **kiểm tra câu trả lời**. Tiêu đề nói "vàng giảm khi lợi suất tăng" tức khẳng định một quan hệ nghịch biến giữa hai chuỗi; mà quan hệ đó thì tính được từ giá, nên kiểm chứng được.

Bước một là trích khẳng định ra khỏi tiêu đề.

{{CODE ch5/fin_core.py | extract_claims}}

**Giải thích.** Tiêu đề tin tài chính có khuôn sẵn: "X giảm **as** Y tăng", "X rơi **after** Y công bố", "X giảm **despite** Y tích cực". Hàm tách tiêu đề tại từ nối, lấy vế trái làm kết quả và vế phải làm nguyên nhân. `kind` phân biệt hai loại: `cause` là khẳng định có quan hệ, còn `contrast` (từ nối `despite`) là người viết đang chỉ ra giá đi **ngược** kỳ vọng nên không dùng để kiểm chứng quan hệ.

Chi tiết quan trọng nhất nằm ở vòng lặp cuối. Hướng của nguyên nhân được tính trong **lân cận của chính thực thể đó** (`WIN_CHARS` ký tự quanh vị trí khớp) chứ không tính trên cả vế. Lý do rất cụ thể: trong tiêu đề *"Treasury Yields and Dollar Offset Softer US Inflation"*, từ chỉ hướng `softer` thuộc về lạm phát; nếu tính hướng trên cả vế thì lợi suất và USD cũng bị gán hướng giảm, và dấu kỳ vọng của khẳng định sai theo. Biến `multi_driver` đánh dấu những vế có từ hai thực thể đo được trở lên, và nhóm này bị loại khỏi phần kiểm chứng ở bước sau.

Bước hai là đối chiếu khẳng định với dữ liệu giá.

{{CODE ch5/fin_core.py | verify_claim}}

**Giải thích.** Hàm trả về một trong bốn kết luận. Trước hết là ba lớp loại trừ: nguyên nhân không có chuỗi giá (Fed, lạm phát, địa chính trị) thì không có gì để đối chiếu; câu nêu nghịch lý thì không khẳng định quan hệ; vế nguyên nhân nhiều thực thể thì không gán riêng được hướng.

Phần còn lại mới được kiểm chứng. Cửa sổ tính tương quan là **45 ngày lịch ngay trước ngày đăng tin**, không lấy ngày sau — bản tin đang giải thích một diễn biến đã xảy ra, nên dữ liệu sau đó không thuộc về điều người viết nhìn thấy, và lấy cả ngày sau chính là rò rỉ nhìn trước mà mục trước vừa chống.

Logic dấu nằm ở dòng `expected = claim["asset_dir"] * claim["driver_dir"]`. Tiêu đề "vàng giảm khi lợi suất tăng" có hướng tài sản là −1 và hướng nguyên nhân là +1, tích bằng −1, tức hàm ý hai chuỗi đi ngược nhau nên tương quan phải âm. Nếu tương quan thực đúng dấu đó thì kết luận là *nhất quán*, ngược dấu là *mâu thuẫn*, còn trị tuyệt đối dưới ngưỡng `CORR_WEAK` thì là *không đủ bằng chứng*.

### 5.4.5. Ghép lại: một câu hỏi đi qua hệ

{{CODE ch5/fin_core.py | answer}}

**Giải thích.** Hàm này là toàn bộ hệ gói lại. Nó gọi bộ định tuyến, rồi rẽ theo nhánh. Nhánh số trả về con số tính được. Nhánh diễn giải trả về tài liệu kèm nguồn. Nhánh nhân quả làm đủ bốn bước: truy hồi tin trong đúng cửa sổ, trích khẳng định từ từng tiêu đề, kiểm chứng từng khẳng định, rồi gom lại thành câu trả lời **hai tầng** — tầng một là báo chí quy cho cái gì, tầng hai là dữ liệu có chống đỡ lời đó không, kèm danh sách nguyên nhân không đo được bằng giá.

Đây là chỗ khác biệt rõ nhất so với RAG thường, vốn chỉ có tầng một và trình bày nó như thể là kết luận. Chạy thật trên câu hỏi "vì sao vàng giảm tháng 6/2026" cho kết quả:

```
Báo chí quy cho : FED (3), GEOPOLITICS (2), INFLATION (1), USD (1)
Không đo được   : FED, GEOPOLITICS, INFLATION
Đối chiếu dữ liệu: USD -> nhất quán, corr = -0,554
Số liệu cùng kỳ : lợi suất vàng quý 2/2026 = -16,09%
```

Hệ còn được đóng gói thành một trang demo tĩnh chạy hoàn toàn trong trình duyệt (`demo/ch5_finance.html`). Bảng giá 2.480 phiên được nhúng thẳng vào trang nên bộ định tuyến và bộ tính chạy tại chỗ bằng JavaScript, không phải đáp án nấu sẵn: gõ "lợi suất của vàng quý 2/2026" thì cách B tính ra −16,09% trên 91 phiên, trong khi cách A truy hồi trúng câu thống kê của cả năm 2026 và trả lời −3,50%. Bản cách A trong demo là bản JavaScript rút gọn, xếp hạng theo độ trùng token thay vì TF-IDF như bản Python; trên câu hỏi này hai bản cho cùng kết quả, nhưng con số tổng hợp ở mục 5.5.1 là của bản Python. Người đọc nhìn thấy ngay chỗ hỏng của cách cũ thay vì chỉ đọc bảng số.

## 5.5. Kết quả

### 5.5.1. Câu hỏi số

Tập câu hỏi được **sinh tự động từ chính bảng giá** nên đáp án đúng có sẵn, không phải gán nhãn tay. Nhờ vậy cỡ mẫu là {{V n(C5('numeric_eval.json')['so_cau_hoi'])}} câu thay vì mười câu như cách gán nhãn thủ công. Hai cách được so: cách A biến bảng thành câu thống kê theo năm rồi truy hồi và rút số bằng biểu thức chính quy, cách B định tuyến sang bộ tính.

{{TAB t_c5_numeric}}

{{FIG ch5_numeric.png | Tỉ lệ trả lời đúng trên câu hỏi số, theo phạm vi thời gian}}

Phải đọc bảng này cho đúng. **Cách B đạt 1,000 không phải vì nó thông minh**, mà vì nó tính trên chính bảng đã sinh ra đáp án; con số đó chỉ xác nhận bộ định tuyến và bộ phân tích câu hỏi chạy đúng, không phải bằng chứng về chất lượng tri thức.

Nội dung thật nằm ở **dạng thất bại của cách A**. Nó trả lời đúng toàn bộ 90 câu hỏi theo năm mà nó có câu thống kê tương ứng, nhưng tụt xuống {{V f(C5('numeric_eval.json')['theo_quy']['A_truy_hoi_cau_thong_ke'],3)}} với câu hỏi theo quý và bằng 0 với câu hỏi tương quan. Không phải vì truy hồi kém, mà vì **không có gì để truy hồi**: kho chỉ chứa câu thống kê theo năm, và tương quan thì không xuất hiện trong bất kỳ câu nào. Đó là giới hạn của việc nhét bảng số vào kho văn bản.

### 5.5.2. Truy hồi tin tức

Hai trục được đo cùng lúc. Trục rò rỉ đếm tài liệu có ngày đăng sau cửa sổ câu hỏi. Trục liên quan so hướng giá nêu trong tiêu đề với hướng mà câu hỏi hỏi: câu "vì sao vàng giảm" mà trả về tin vàng tăng là trượt.

{{TAB t_c5_retrieval}}

{{FIG ch5_retrieval.png | Rò rỉ nhìn trước và độ liên quan ở bốn cấu hình}}

Ba điều đọc được. Thứ nhất, bỏ lọc thời gian thì **gần hai phần ba tài liệu trả về đến từ tương lai** so với câu hỏi. Thứ hai, dòng cuối bảng cho thấy mức độ nghiêm trọng của vấn đề xuyên ngôn ngữ: không mở rộng truy vấn thì độ liên quan bằng **0,00**, tức bộ truy hồi xếp hạng gần như ngẫu nhiên, trong khi chỉ số rò rỉ không hề báo động vì nó chỉ nhìn ngày tháng. Phải đo cả hai trục mới thấy. Thứ ba, lọc thời gian có cái giá thật: độ liên quan giảm từ 0,43 xuống 0,38 vì kho ứng viên bị thu hẹp. Đánh đổi này nên nói rõ chứ không giấu — mất 5 điểm liên quan để loại sạch tài liệu đến từ tương lai.

### 5.5.3. Kiểm chứng khẳng định nhân quả

{{TAB t_c5_claims}}

{{FIG ch5_claims.png | Lý do không kiểm chứng được, và kiểm định vô hiệu}}

Kết quả đáng chú ý nhất không phải tỉ lệ nhất quán mà là **tỉ lệ không kiểm chứng được**: {{V pc(C5('claims_gold.json')['ty_le_nguyen_nhan_khong_co_chuoi_gia'])}} số khẳng định viện tới nguyên nhân không có chuỗi giá để đối chiếu — Fed 77 lần, lạm phát 83 lần, địa chính trị 38 lần. Đó không phải khuyết điểm của hệ kiểm chứng mà là đặc điểm của chính diễn ngôn tài chính: phần lớn lời giải thích viện tới những thứ không đo trực tiếp bằng giá được. Giá trị của bộ kiểm chứng vì thế không nằm ở việc bắt lỗi nhà báo, mà ở việc **đo được bao nhiêu phần lời giải thích có thể đối chiếu với dữ liệu**.

Với phần kiểm chứng được, tỉ lệ nhất quán rất cao. Nhưng con số đó không được phép đọc một mình:

{{TAB t_c5_null}}

Kiểm định vô hiệu gán cho mỗi khẳng định một ngày ngẫu nhiên rồi kiểm chứng lại. Tỉ lệ nhất quán vẫn đạt {{V f(C5('claims_gold.json')['kiem_dinh_vo_hieu_ngay_ngau_nhien']['trung_binh'],3)}}, nghĩa là **phần lớn độ nhất quán đến từ chế độ thị trường chứ không từ bản tin cụ thể**: tương quan vàng–đô âm ở gần hết các quý, nên một khẳng định về đồng đô gán ngày bừa vẫn "nhất quán" phần lớn số lần. Chênh lệch giữa ngày thật và ngày ngẫu nhiên vẫn tồn tại và tương đương khoảng 2,5 độ lệch chuẩn, nên có tín hiệu thật, nhưng nhỏ hơn nhiều so với ấn tượng ban đầu. Không có kiểm định này thì kết luận của mục sẽ sai hẳn về mức độ.

### 5.5.4. Cạnh tĩnh che mất hiện tượng

{{TAB t_c5_regime}}

{{FIG ch5_regime.png | Tương quan theo quý so với một hệ số duy nhất cho cả giai đoạn}}

Ba cặp in trong bảng có hệ số toàn giai đoạn gần 0, đọc lên là "không liên quan". Nhưng cắt theo quý thì chúng liên hệ rất mạnh, chỉ là đổi dấu theo chế độ thị trường nên **triệt tiêu nhau khi lấy trung bình**. Cặp vàng–S&P 500 có hệ số chung +0,122 trong khi biên độ theo quý là 0,950 và đổi dấu 5 lần. Một đồ thị tĩnh sẽ kết luận hai thứ này không liên quan, trong khi sự thật là liên quan mạnh và dấu phụ thuộc giai đoạn.

Cặp vàng–bạc là ca đối chứng: quan hệ thật sự ổn định, không lần nào đổi dấu, nên ở đó một cạnh tĩnh dùng được. Điều này cho thấy cắt lát thời gian không phải lúc nào cũng cần — nó cần đúng ở những cặp mà quan hệ không ổn định, và chỉ có cắt lát mới biết cặp nào thuộc loại nào.

### 5.5.5. Miền chứng khoán: khi bộ trích không dùng được

{{TABL t_c5_stocks}}

Bộ trích khẳng định chạy rất tốt trên tin vàng nhưng gần như không trích được gì từ tin cổ phiếu. Nguyên nhân không nằm ở kỹ thuật mà ở **thể loại tin**: vàng là tài sản vĩ mô, được các ban thị trường viết theo khuôn "giá X giảm vì Y tăng"; còn tin về từng mã cổ phiếu phần lớn là tin doanh nghiệp (bổ nhiệm nhân sự, giải thưởng, công bố kết quả kinh doanh) không nêu quan hệ giá nào.

Đây là kết quả âm nhưng đáng báo cáo, vì nó xác định đúng phạm vi áp dụng của phương pháp: **bộ trích phụ thuộc thể loại văn bản, không phụ thuộc miền dữ liệu**. Với miền chứng khoán, hệ vẫn dùng được ba thành phần còn lại là định tuyến câu hỏi số, cạnh `belongs_to` và cạnh tương quan cắt theo quý; riêng phần kiểm chứng khẳng định thì không áp dụng.

### 5.5.6. Đối chiếu với miền y văn

Đề bài của phần Knowledge Graph và RAG yêu cầu ba miền: bệnh tiểu đường, chứng khoán và vàng. Chương này làm lại hai miền tài chính theo thiết kế mới, còn miền y văn (298 abstract PubMed, đồ thị 319 nút) giữ nguyên kết quả của bài thực hành trước và được dẫn lại ở đây để so sánh.

{{TAB t_c5_diabetes}}

Con số đáng chú ý là mức chênh giữa hai miền. Ở bộ tiểu đường, mở rộng truy hồi bằng đồ thị kéo `context_recall` từ 0,417 lên 0,750 tại k = 2; ở bộ vàng, đồ thị không cải thiện được gì tại k = 2 và k = 4, còn tại k = 6 thì kém hơn truy hồi thường (0,900 so với 1,000), tức việc cộng điểm cho tài liệu nhắc tới thực thể lân cận đã đẩy một tài liệu đúng ra khỏi top-k. Giải thích nằm ở quy mô đồ thị: đồ thị y văn có 319 nút vì thực thể là thuốc, biến chứng, chỉ số xét nghiệm, nên câu hỏi thường phải bắc cầu qua nhiều thực thể mới tới được tài liệu đúng. Đồ thị vàng chỉ có 18 nút, và với một câu hỏi về vàng thì truy hồi theo văn bản thường đã đủ, không còn gì cho đồ thị bổ sung.

Điều này củng cố nhận xét ở mục 5.5.4 từ một hướng khác: giá trị của đồ thị tri thức không đến từ việc có đồ thị, mà đến từ việc đồ thị có đủ phong phú để bắc cầu hay không. Với miền tài chính, đóng góp của đồ thị nằm ở chiều kiểm chứng chứ không ở chiều mở rộng truy hồi.

## 5.6. Những lỗi đã mắc và cách phát hiện

Mục này không phải phần phụ. Trong quá trình làm, năm lỗi đã bị phát hiện và cả năm đều **làm sai lệch kết quả theo hướng có lợi cho phương pháp mới**. Ghi lại ở đây vì cách phát hiện chúng là phần có giá trị nhất của chương.

**Lỗi 1 — tám "mâu thuẫn" đều là lỗi trích.** Vòng chạy đầu báo 8 khẳng định mâu thuẫn với dữ liệu. Kiểm tay từng cái thì cả 8 đều do bộ trích đọc sai hướng: `hits ... low` bị hiểu thành tăng, `pull back from highs` bị từ `highs` lấn, `softer` của lạm phát bị gán cho lợi suất. Nếu không kiểm tay, chương này sẽ báo cáo 8 phát hiện mà thực chất là nhiễu.

**Lỗi 2 — biểu thức chính quy thiếu ranh giới từ.** Các mẫu từ chỉ hướng viết dạng `(gain|lose|peak|low|...)` không có `\b`, nên `gain` khớp vào *again*, `lose` khớp vào *closes*, `peak` khớp vào *speaks*, `low` khớp vào *below*. Lỗi này không báo lỗi gì, chỉ làm đếm sai số từ chỉ hướng. Nó không bị phát hiện trong ba vòng sửa đầu vì các vòng đó **chỉ soi những ca cho kết luận mâu thuẫn**, nên cả một lớp lỗi chưa từng được nhìn tới.

**Lỗi 3 — từ "lợi suất" mang hai nghĩa.** Nó vừa là từ khoá của độ đo *return*, vừa là tên đặt cho chuỗi US10Y. Câu hỏi "lợi suất của vàng năm 2021" vì thế bị cả hai phương pháp hiểu thành câu hỏi về trái phiếu.

**Lỗi 4 — so chuỗi chưa viết thường.** Trong mốc so sánh A, điều kiện `vi in q.lower()` so một chuỗi chưa viết thường với chuỗi đã viết thường, nên tên "S&P 500" không bao giờ khớp và phương pháp A bốc gần như ngẫu nhiên giữa 8 tài sản. Lỗi này nằm ở **phe đối chứng**. Đo riêng tác động của nó bằng cách hoàn tác đúng một dòng: phương pháp A tụt từ 0,283 xuống 0,228. Khi cả lỗi này lẫn lỗi 3 còn nguyên thì A chỉ đạt 0,175. Nói cách khác, hai lỗi nằm ở phe đối chứng cộng lại đã thổi phồng khoảng cách giữa hai phương pháp thêm hơn mười điểm phần trăm.

**Lỗi 5 — bộ truy hồi mắc đúng lỗi xuyên ngôn ngữ vừa sửa cho phe A.** Chỉ số rò rỉ nhìn trước không hề báo động vì nó chỉ nhìn ngày tháng; phải thêm trục đo độ liên quan mới lộ ra rằng độ liên quan bằng 0.

Bài học rút ra có thể phát biểu gọn: khi người làm thí nghiệm tự dựng cả phương pháp mới lẫn phương pháp đối chứng, **lỗi trong phương pháp đối chứng luôn có lợi cho mình nên phải soi kỹ hơn**; và một chỉ số duy nhất thì luôn có một lớp lỗi mà nó không nhìn thấy.

## 5.7. Hạn chế

**Khoảng thời gian của phần diễn giải hẹp.** Tin tức chỉ dày từ tháng 1/2026, nên phần kiểm chứng và diễn giải chạy trên 10 tháng. Mọi kết luận về chúng chỉ có giá trị trong khoảng đó.

**Cỡ mẫu phần kiểm chứng nhỏ.** Chỉ {{V n(C5('claims_gold.json')['kiem_chung_duoc'])}} khẳng định kiểm chứng được, trong đó phần lớn liên quan tới đồng đô. Độ lệch chuẩn của kiểm định vô hiệu là {{V f(C5('claims_gold.json')['kiem_dinh_vo_hieu_ngay_ngau_nhien']['do_lech_chuan'],3)}}, đủ lớn để chênh lệch quan sát được không mang tính quyết định.

**Mẫu soi tay chỉ gồm những ca đã qua lọc.** Độ chính xác trích đạt 30/30 trên mẫu ngẫu nhiên, nhưng mẫu ấy lấy trong nhóm đã loại bỏ các ca đa nguyên nhân và các ca không rõ hướng, tức đúng những ca khó. Đó là độ chính xác sau lọc, không phải độ chính xác của bộ trích nói chung.

**Phần sinh câu trả lời là trích rút, không phải sinh văn bản.** Hệ chọn và ghép câu có sẵn chứ không có mô hình ngôn ngữ viết lại. Ưu điểm là không thể bịa ra số và mọi câu đều truy về nguồn được; nhược điểm là câu trả lời không mượt và không tổng hợp được nhiều nguồn thành một đoạn mạch lạc.

**Tương quan không phải nhân quả.** Bộ kiểm chứng chỉ trả lời câu hỏi "hai chuỗi có đi cùng chiều hay ngược chiều trong giai đoạn đó không". Nó không chứng minh được nguyên nhân, và kết luận "nhất quán" chỉ có nghĩa là lời của người viết không mâu thuẫn với dữ liệu quan sát.

## 5.8. Kết luận chương

Chương này cho thấy ba điều mà một hệ RAG thuần không làm được, và cả ba đều đo được bằng số.

Bảng số nên được **truy vấn** chứ không nên nhét vào kho **truy hồi**: định tuyến câu hỏi số sang bộ tính nâng tỉ lệ trả lời đúng từ {{V f(C5('numeric_eval.json')['tong']['A_truy_hoi_cau_thong_ke'],3)}} lên {{V f(C5('numeric_eval.json')['tong']['B_dinh_tuyen_sang_bo_tinh'],3)}} trên {{V n(C5('numeric_eval.json')['so_cau_hoi'])}} câu hỏi, và phần chênh lệch nằm trọn ở những câu hỏi mà cách cũ không có gì để tìm.

Thời gian phải là một chiều của cả truy hồi lẫn đồ thị: không lọc thời gian thì gần hai phần ba tài liệu trả về đến từ tương lai so với câu hỏi, còn một cạnh đồ thị duy nhất cho sáu năm có thể cho hệ số gần 0 ở những cặp thực ra liên hệ rất mạnh nhưng đổi dấu theo chế độ.

Đồ thị tri thức dùng được theo chiều kiểm chứng chứ không chỉ chiều mở rộng, và điều đáng giá nhất nó đo được là **phần lời giải thích không thể đối chiếu với dữ liệu**, chiếm khoảng hai phần ba số khẳng định trong tin tức về vàng.

Cuối cùng, mục 5.6 cho thấy phần lớn công sức của một chương thực nghiệm không nằm ở việc xây hệ mà ở việc kiểm tra xem con số thu được có thật hay không. Năm lỗi đã mắc đều nghiêng về phía làm phương pháp mới trông tốt hơn thực tế, và chúng chỉ lộ ra khi được soi bằng một chỉ số khác với chỉ số dùng để tối ưu.
