# -*- coding: utf-8 -*-
"""Lõi của hệ hỏi đáp tài chính: định tuyến câu hỏi, truy hồi theo thời gian,
đồ thị tri thức có kiểu, và kiểm chứng khẳng định nhân quả bằng dữ liệu giá.

Khác biệt so với `ragkg_core.py` của bài 07 nằm ở bốn chỗ:

1. Bảng số không nằm trong kho truy hồi. Câu hỏi số được định tuyến sang bộ
   tính toán chạy thẳng trên chuỗi giá, nên trả lời được cả những câu chưa ai
   viết sẵn thành văn bản.
2. Cạnh đồ thị có kiểu và tách nguồn gốc: `correlates_with` là thứ tính được
   từ giá, `claims_cause` là thứ nhà báo nói. Hai loại này không được trộn.
3. Mọi tài liệu và mọi cạnh đều mang khoảng thời gian; truy hồi lọc theo thời
   gian TRƯỚC khi tính độ tương đồng, nhờ đó tránh rò rỉ nhìn trước.
4. Đồ thị được dùng để kiểm chứng câu trả lời, không chỉ để mở rộng truy hồi.
"""
import collections
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# Số phiên tối thiểu để một cửa sổ được tính tương quan. Dưới mức này hệ số
# tương quan dao động quá mạnh, báo một con số ở đó là báo nhiễu.
MIN_DAYS = 15

# Khoảng thời gian có tin tức đủ dày. Google News RSS chỉ trả tin gần đây nên
# trước 2026 kho tin gần như rỗng; mọi câu hỏi diễn giải phải nằm trong khoảng
# này, hỏi ra ngoài là đo lỗ hổng dữ liệu chứ không đo hệ thống.
NEWS_START = pd.Timestamp("2026-01-01", tz="UTC")


# ----------------------------------------------------------------- dữ liệu

def load_prices(name="gold_market.csv"):
    """Bảng giá theo ngày, chỉ số là ngày giao dịch."""
    df = pd.read_csv(os.path.join(DATA, name), parse_dates=["Date"])
    return df.set_index("Date").sort_index()


def load_news(name="gold_news.csv"):
    """Tin tức kèm mốc thời gian, đã khử trùng lặp theo tiêu đề.

    Cùng một bản tin hãng được hàng chục trang đăng lại; không khử thì top-k
    sẽ bị một tin chiếm hết chỗ."""
    df = pd.read_csv(os.path.join(DATA, name))
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True, errors="coerce")
    df = df.dropna(subset=["date", "title"])
    df["key"] = df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    df = df.drop_duplicates(subset="key").drop(columns="key")
    return df.sort_values("date").reset_index(drop=True)


# US10Y là lãi suất, không phải giá. Quy ước trong tài chính là lấy CHÊNH LỆCH
# lãi suất (điểm cơ bản), không lấy lợi suất log: khi lãi suất ở mức 0,5% thì
# tăng lên 0,6% là +18% theo log nhưng thực chất chỉ là 10 điểm cơ bản, và
# tương quan tính theo log sẽ bị méo nặng ở giai đoạn lãi suất thấp.
RATE_SERIES = ("US10Y",)


def returns(prices, col):
    """Biến động theo ngày của một chuỗi.

    Tương quan phải tính trên biến động chứ không trên mức giá: hai chuỗi cùng
    có xu hướng tăng sẽ cho tương quan gần 1 dù chẳng liên quan gì nhau (tương
    quan giả do xu hướng).

    Chuỗi lãi suất dùng chênh lệch, chuỗi giá dùng lợi suất log. Giá trị không
    dương bị loại trước khi lấy log: ngày 20/4/2020 hợp đồng dầu WTI xuống giá
    âm (-37,63), một sự kiện có thật làm phép lấy log trả về NaN.
    """
    s = prices[col].astype(float).dropna()
    if col in RATE_SERIES:
        return s.diff().dropna()
    s = s[s > 0]
    return np.log(s).diff().dropna()


# ------------------------------------------------------- chuẩn hoá thực thể

# Mỗi nút của đồ thị kèm danh sách cách gọi khác nhau trong tin tức. Không có
# bước này thì "gold", "XAU" và "bullion" thành ba nút rời rạc.
ENTITIES = {
    "GOLD":        dict(kind="Asset", series="GOLD",
                        aliases=["gold", "xau", "bullion", "vàng"]),
    "SILVER":      dict(kind="Asset", series="SILVER",
                        aliases=["silver", "xag", "bạc"]),
    "OIL":         dict(kind="Asset", series="OIL",
                        aliases=["oil", "crude", "brent", "wti", "dầu"]),
    "SP500":       dict(kind="Asset", series="SP500",
                        aliases=["s&p", "sp500", "s&p 500", "stocks", "equities",
                                 "wall street"]),
    "BTC":         dict(kind="Asset", series="BTC",
                        aliases=["bitcoin", "btc", "crypto"]),
    "USD":         dict(kind="MacroIndicator", series="USD",
                        aliases=["dollar", "usd", "greenback", "dxy",
                                 "đồng đô", "us dollar"]),
    "US10Y":       dict(kind="MacroIndicator", series="US10Y",
                        aliases=["yield", "yields", "treasury", "treasuries",
                                 "10-year", "bond", "lợi suất trái phiếu"]),
    "VIX":         dict(kind="MacroIndicator", series="VIX",
                        aliases=["vix", "volatility index", "fear gauge"]),
    "FED":         dict(kind="Institution", series=None,
                        aliases=["fed", "federal reserve", "fomc", "powell",
                                 "central bank", "rate hike", "rate cut",
                                 "interest rate", "interest rates", "monetary policy"]),
    "INFLATION":   dict(kind="MacroIndicator", series=None,
                        aliases=["inflation", "cpi", "price pressures", "lạm phát"]),
    "GEOPOLITICS": dict(kind="Theme", series=None,
                        aliases=["war", "tensions", "conflict", "iran", "ukraine",
                                 "middle east", "sanctions", "geopolitical"]),
    "RECESSION":   dict(kind="Theme", series=None,
                        aliases=["recession", "slowdown", "growth fears",
                                 "economic data", "jobs data", "payrolls"]),
}


def _alias_patterns():
    """Gom mọi cách gọi thành một regex cho mỗi nút, ưu tiên cụm dài trước để
    'interest rates' không bị 'rate' cắt mất."""
    out = {}
    for node, meta in ENTITIES.items():
        al = sorted(meta["aliases"], key=len, reverse=True)
        out[node] = re.compile(
            r"(?<![a-z])(" + "|".join(re.escape(a) for a in al) + r")(?![a-z])",
            re.IGNORECASE)
    return out


ALIAS_RX = _alias_patterns()


def link_entities(text):
    """Trả về các nút đồ thị được nhắc tới trong đoạn văn bản."""
    t = str(text).lower()
    return [node for node, rx in ALIAS_RX.items() if rx.search(t)]


def series_of(node):
    """Chuỗi giá tương ứng với một nút, hoặc None nếu nút không đo được bằng số
    (ví dụ GEOPOLITICS). Phân biệt này quyết định khẳng định nào kiểm chứng
    được và khẳng định nào không."""
    return ENTITIES.get(node, {}).get("series")


# --------------------------------------------------- bộ tính toán trên số

METRICS = ("return", "volatility", "drawdown", "correlation", "last", "range")


def window_slice(prices, start, end):
    return prices.loc[str(start):str(end)]


def metric_return(prices, asset, start, end):
    s = window_slice(prices, start, end)[asset].astype(float).dropna()
    if len(s) < 2:
        return None
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def metric_volatility(prices, asset, start, end):
    """Độ biến động quy năm từ lợi suất ngày."""
    r = returns(window_slice(prices, start, end), asset)
    if len(r) < MIN_DAYS:
        return None
    return float(r.std() * np.sqrt(252))


def metric_drawdown(prices, asset, start, end):
    """Mức sụt sâu nhất so với đỉnh trước đó trong cửa sổ."""
    s = window_slice(prices, start, end)[asset].astype(float).dropna()
    if len(s) < 2:
        return None
    return float((s / s.cummax() - 1.0).min())


def metric_correlation(prices, a, b, start, end):
    """Tương quan lợi suất ngày giữa hai chuỗi, kèm số phiên thực dùng.

    Trả về (hệ số, số phiên) để nơi gọi tự quyết có tin con số đó không."""
    w = window_slice(prices, start, end)
    ra, rb = returns(w, a), returns(w, b)
    j = ra.index.intersection(rb.index)
    if len(j) < MIN_DAYS:
        return None, len(j)
    return float(np.corrcoef(ra[j], rb[j])[0, 1]), len(j)


def compute(prices, metric, asset, start, end, other=None):
    """Một cửa duy nhất cho mọi phép tính số, để bộ định tuyến chỉ gọi một hàm."""
    if metric == "return":
        return metric_return(prices, asset, start, end)
    if metric == "volatility":
        return metric_volatility(prices, asset, start, end)
    if metric == "drawdown":
        return metric_drawdown(prices, asset, start, end)
    if metric == "correlation":
        return metric_correlation(prices, asset, other, start, end)[0]
    if metric == "last":
        s = window_slice(prices, start, end)[asset].dropna()
        return float(s.iloc[-1]) if len(s) else None
    if metric == "range":
        s = window_slice(prices, start, end)[asset].astype(float).dropna()
        return (float(s.min()), float(s.max())) if len(s) else None
    raise ValueError("không biết độ đo " + str(metric))


# --------------------------------------- trích khẳng định nhân quả từ tin tức

# Hướng đi của giá, suy từ động từ trong tiêu đề.
UP_WORDS = r"\b(rise|rises|rising|rose|climb|climbs|climbed|gain|gains|jump|jumps|" \
           r"surge|surges|surging|rally|rallies|rebound|rebounds|advance|advances|" \
           r"higher|soar|soars|soaring|firm|firms|strengthen|strengthens|" \
           r"strong|stronger|strength|mount|mounts|mounting|hike|hikes|" \
           r"increase|increases|record|tops|breaks|hawkish|elevated|bullish|" \
           r"boost|boosts|boosted|high|highs|highest|peak|peaks)\b"
DOWN_WORDS = r"\b(fall|falls|fell|drop|drops|slump|slumps|slide|slides|dip|dips|" \
             r"ease|eases|easing|dim|dims|lose|loses|loss|weaken|weakens|" \
             r"weak|weaker|weakness|lower|retreat|retreats|sink|sinks|" \
             r"tumble|tumbles|fade|fades|fading|cool|cooler|cooling|" \
             r"cut|cuts|decline|declines|dovish|soft|softer|bearish|" \
             r"slip|slips|selloff|sell-off|subdued|low|lows|lowest|trough)\b"

# Phủ định làm đảo hướng: "rate-cut bets tempered" là kỳ vọng giảm bị dập đi,
# tức nghiêng về phía tăng. Không xử lý thì dấu kỳ vọng sai hoàn toàn.
NEGATORS = r"\b(tempered|dashed|faded|pared|trimmed|doubts? over|no longer|" \
           r"less likely|scaled back|unwound)\b"

# Cụm động từ phải bắt trước khi bắt từ đơn, vì "pull back from highs" có
# 'highs' là từ tăng trong khi cả cụm lại mang nghĩa giảm.
PHRASES_DOWN = r"\b(pull[s]? back|pulled back|pare[sd]? back|give[s]? back|" \
               r"struggle[sd]? to|fail[sed]* to hold|come[s]? off|back off)\b"
PHRASES_UP = r"\b(bounce[sd]? back|push(?:es|ed)? higher|hold[s]? gains|extend[s]? gains|" \
             r"retrace[sd]? losses|recover[s]? losses|erase[sd]? losses)\b"

# Cụm chỉ thời điểm, không chỉ hướng giá. Phải xoá trước khi đếm, nếu không
# "in advance of the release" bị tính là một từ chỉ hướng tăng vì chứa 'advance'.
NEUTRAL_RX = re.compile(r"\b(in advance of|ahead of|on tap|in focus|in view)\b",
                        re.IGNORECASE)

# Số ký tự lấy quanh vị trí thực thể khi suy hướng của riêng thực thể đó.
WIN_CHARS = 28

# Từ nối tách vế kết quả với vế nguyên nhân.
CAUSE_CONN = r"\b(as|after|amid|on|following|due to|driven by|because of|" \
             r"thanks to|weighed by|helped by|boosted by)\b"
CONTRAST_CONN = r"\b(despite|even as|although|though|but)\b"


def _direction(text):
    """+1 nếu đoạn nói giá lên, -1 nếu nói giá xuống, 0 nếu không rõ.

    Đếm số từ mỗi hướng rồi lấy bên nhiều hơn; hoà thì trả 0 và khẳng định đó
    bị loại khỏi phần kiểm chứng thay vì đoán bừa. Nếu có từ phủ định thì đảo
    dấu, vì "rate-cut bets tempered" nghĩa là kỳ vọng giảm bị dập đi."""
    t = NEUTRAL_RX.sub(" ", str(text).lower())
    if re.search(PHRASES_DOWN, t):
        return -1
    if re.search(PHRASES_UP, t):
        return 1
    up = len(re.findall(UP_WORDS, t))
    down = len(re.findall(DOWN_WORDS, t))
    if up == down:
        return 0
    d = 1 if up > down else -1
    return -d if re.search(NEGATORS, t) else d


def extract_claims(title, asset="GOLD"):
    """Tách tiêu đề thành khẳng định (tài sản, hướng) <- (nguyên nhân, hướng).

    Dựa vào khuôn 'X as Y', 'X after Y', 'X despite Y' vốn có sẵn trong tiêu đề
    tin tài chính. Trả về danh sách vì một tiêu đề có thể nêu nhiều nguyên nhân.

    Loại 'cause' là khẳng định có quan hệ; loại 'contrast' ('despite') là người
    viết đang chỉ ra giá đi NGƯỢC kỳ vọng, nên không dùng để kiểm chứng quan hệ.
    """
    t = str(title)
    m = re.search(CONTRAST_CONN, t, re.IGNORECASE)
    kind = "contrast"
    if m is None:
        m = re.search(CAUSE_CONN, t, re.IGNORECASE)
        kind = "cause"
    if m is None:
        return []

    left, right = t[:m.start()], t[m.end():]
    if asset not in link_entities(left):
        return []                       # vế kết quả không nói về tài sản đang xét
    a_dir = _direction(left)

    # Hướng của nguyên nhân phải tính trong LÂN CẬN của chính thực thể đó, không
    # tính trên cả vế. Tiêu đề "Treasury Yields and Dollar Offset Softer US
    # Inflation" có 'softer' thuộc về lạm phát; tính trên cả vế thì lợi suất và
    # USD cũng bị gán hướng giảm, và dấu kỳ vọng sai theo.
    drivers = [d for d in link_entities(right) if d != asset]
    multi = sum(1 for d in drivers if series_of(d)) > 1
    out = []
    for drv in drivers:
        mm = ALIAS_RX[drv].search(right)
        near = right[max(0, mm.start() - WIN_CHARS): mm.end() + WIN_CHARS] if mm else right
        out.append({
            "asset": asset, "driver": drv, "kind": kind,
            "asset_dir": a_dir, "driver_dir": _direction(near),
            "multi_driver": multi, "context": near.strip(),
            "connector": m.group(0).lower(), "title": t,
        })
    return out


def claims_from_news(news, asset="GOLD"):
    """Chạy bộ trích trên cả kho tin, giữ lại mốc thời gian của từng khẳng định."""
    rows = []
    for _, r in news.iterrows():
        for c in extract_claims(r["title"], asset):
            c["date"] = r["date"]
            c["source"] = r.get("source", "")
            rows.append(c)
    return pd.DataFrame(rows)


# ------------------------------------------------------- kiểm chứng bằng giá

CORR_WEAK = 0.15        # dưới ngưỡng này coi như không đủ bằng chứng


def verify_claim(prices, claim, min_days=MIN_DAYS):
    """Đối chiếu một khẳng định của tin tức với tương quan tính từ giá, trong
    đúng tháng mà tin được đăng.

    Logic dấu: tiêu đề 'vàng giảm khi lợi suất tăng' hàm ý hai chuỗi đi ngược
    nhau, tức tương quan âm. Dấu kỳ vọng vì thế là tích hướng của hai vế.
    """
    drv_series = series_of(claim["driver"])
    if drv_series is None:
        return {"verdict": "không kiểm chứng được", "reason": "nguyên nhân không có chuỗi giá"}
    if claim["kind"] == "contrast":
        return {"verdict": "không kiểm chứng được", "reason": "câu nêu nghịch lý, không khẳng định quan hệ"}
    if claim["asset_dir"] == 0 or claim["driver_dir"] == 0:
        return {"verdict": "không kiểm chứng được", "reason": "không xác định được hướng của một vế"}
    # Vế nguyên nhân nêu từ hai thực thể đo được trở lên thì không gán riêng
    # hướng cho từng cái một cách đáng tin: trong "Treasury Yields and Dollar
    # Offset Softer US Inflation", từ chỉ hướng 'softer' thuộc về lạm phát chứ
    # không thuộc về lợi suất. Loại hẳn nhóm này, và báo cáo số bị loại.
    if claim.get("multi_driver"):
        return {"verdict": "không kiểm chứng được",
                "reason": "vế nguyên nhân có nhiều thực thể, không gán riêng được hướng"}

    # Cửa sổ là 45 ngày lịch NGAY TRƯỚC ngày đăng tin, không lấy ngày sau. Bản
    # tin đang giải thích một diễn biến đã xảy ra, nên dữ liệu sau đó không
    # thuộc về điều người viết nhìn thấy; lấy cả ngày sau là rò rỉ nhìn trước.
    d = pd.Timestamp(claim["date"]).tz_convert(None).normalize()
    start, end = d - pd.Timedelta(days=45), d
    corr, n = metric_correlation(prices, series_of(claim["asset"]), drv_series, start, end)
    if corr is None:
        return {"verdict": "không kiểm chứng được",
                "reason": "cửa sổ chỉ có %d phiên, dưới mức tối thiểu %d" % (n, min_days)}

    expected = claim["asset_dir"] * claim["driver_dir"]
    if abs(corr) < CORR_WEAK:
        verdict = "không đủ bằng chứng"
    elif np.sign(corr) == expected:
        verdict = "nhất quán"
    else:
        verdict = "mâu thuẫn"
    return {"verdict": verdict, "corr": round(corr, 3), "n_days": n,
            "expected_sign": int(expected),
            "window": "%s..%s" % (start.date(), end.date())}


def verify_all(prices, claims):
    """Kiểm chứng toàn bộ khẳng định, trả về bảng để báo cáo thống kê."""
    recs = []
    for _, c in claims.iterrows():
        recs.append({**c.to_dict(), **verify_claim(prices, c)})
    return pd.DataFrame(recs)


# ------------------------------------------- đồ thị tri thức có kiểu, có thời gian

def correlation_edges(prices, freq="QE", min_days=MIN_DAYS):
    """Cạnh `correlates_with` cắt theo từng lát thời gian.

    Bài 07 dùng một hệ số cho cả giai đoạn 2020-2026. Tương quan vàng-USD thật
    ra đổi theo chế độ thị trường, nên một con số cho sáu năm là trung bình hoá
    mất hiện tượng. Ở đây mỗi quý (hoặc tháng) có cạnh riêng, kèm số phiên thực
    dùng để người đọc biết con số đáng tin tới đâu.
    """
    cols = [c for c in prices.columns if prices[c].notna().sum() > min_days]
    rows = []
    for period, chunk in prices.groupby(pd.Grouper(freq=freq)):
        if len(chunk) < min_days:
            continue
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                corr, n = metric_correlation(chunk, a, b,
                                             chunk.index[0], chunk.index[-1])
                if corr is None:
                    continue
                rows.append({"src": a, "dst": b, "rel": "correlates_with",
                             "period": str(period.date()), "corr": round(corr, 3),
                             "n_days": n})
    return pd.DataFrame(rows)


def claim_edges(verified):
    """Cạnh `claims_cause` gộp theo cặp (nguyên nhân -> tài sản) và theo tháng.

    Giữ riêng khỏi `correlates_with`: một bên là thứ tính được từ giá, một bên
    là thứ người viết nói. Trộn hai loại vào một cạnh là trình bày suy đoán của
    phóng viên như thể là dữ kiện đo được.
    """
    if len(verified) == 0:
        return pd.DataFrame()
    v = verified.copy()
    v["month"] = pd.to_datetime(v["date"], utc=True).dt.strftime("%Y-%m")
    g = v.groupby(["driver", "asset", "month"])
    rows = []
    for (drv, asset, month), sub in g:
        rows.append({
            "src": drv, "dst": asset, "rel": "claims_cause", "period": month,
            "n_claims": len(sub),
            "nhat_quan": int((sub["verdict"] == "nhất quán").sum()),
            "mau_thuan": int((sub["verdict"] == "mâu thuẫn").sum()),
            "khong_kiem_chung": int((sub["verdict"] == "không kiểm chứng được").sum()),
        })
    return pd.DataFrame(rows)


def regime_shifts(corr_edges, pair, threshold=0.3):
    """Những lát thời gian mà tương quan của một cặp đổi dấu đáng kể.

    Đây là thứ đồ thị tĩnh của bài 07 không thể hiện được."""
    a, b = pair
    sub = corr_edges[((corr_edges["src"] == a) & (corr_edges["dst"] == b)) |
                     ((corr_edges["src"] == b) & (corr_edges["dst"] == a))]
    sub = sub.sort_values("period").reset_index(drop=True)
    flips = []
    for i in range(1, len(sub)):
        p, c = sub.loc[i - 1, "corr"], sub.loc[i, "corr"]
        if np.sign(p) != np.sign(c) and abs(p - c) > threshold:
            flips.append({"tu": sub.loc[i - 1, "period"], "den": sub.loc[i, "period"],
                          "corr_truoc": p, "corr_sau": c})
    return sub, pd.DataFrame(flips)


# ------------------------------------------------- phân tích và định tuyến câu hỏi

# Thứ tự trong dict là thứ tự xét, nên độ đo cụ thể hơn phải đứng trước: câu
# "tương quan giữa vàng và lợi suất trái phiếu" chứa cả từ khoá của tương quan
# lẫn của lợi suất, xét sai thứ tự là định tuyến sai.
METRIC_WORDS = {
    "correlation": ["tương quan", "correlation", "liên hệ với", "đi cùng"],
    "return":      ["lợi suất", "tăng bao nhiêu", "sinh lời", "return", "performance"],
    "volatility":  ["biến động", "volatility", "độ lệch"],
    "drawdown":    ["sụt", "giảm sâu", "drawdown", "lỗ sâu nhất"],
    "last":        ["giá cuối", "giá đóng cửa", "last price", "closing"],
    "range":       ["cao nhất", "thấp nhất", "khoảng giá", "range"],
}
CAUSAL_WORDS = ["vì sao", "tại sao", "nguyên nhân", "do đâu", "why", "driver",
                "lý do", "điều gì khiến"]


def parse_period(q):
    """Rút khoảng thời gian từ câu hỏi. Trả (start, end) hoặc None nếu không nêu.

    Thiếu bước này thì mọi câu hỏi đều chạy trên toàn bộ 2020-2026 và câu trả
    lời cho 'năm 2022' sẽ giống hệt câu trả lời cho 'năm 2024'."""
    t = str(q).lower()
    m = re.search(r"quý\s*([1-4])\s*[/\- ]\s*(20\d\d)", t)
    if m:
        qt, yr = int(m.group(1)), int(m.group(2))
        start = pd.Timestamp(year=yr, month=3 * qt - 2, day=1)
        return start, start + pd.offsets.QuarterEnd(0)
    m = re.search(r"tháng\s*(1[0-2]|[1-9])\s*[/\- ]\s*(20\d\d)", t)
    if m:
        mo, yr = int(m.group(1)), int(m.group(2))
        start = pd.Timestamp(year=yr, month=mo, day=1)
        return start, start + pd.offsets.MonthEnd(0)
    m = re.search(r"(?:năm|in|of)?\s*\b(20[12]\d)\b", t)
    if m:
        yr = int(m.group(1))
        return pd.Timestamp(year=yr, month=1, day=1), pd.Timestamp(year=yr, month=12, day=31)
    return None


def parse_question(q):
    """Phân loại câu hỏi và rút ra mọi thứ cần để trả lời.

    Đây là bộ định tuyến: câu hỏi số đi sang bộ tính trên bảng giá, câu hỏi
    diễn giải và nhân quả đi sang truy hồi tin tức. Bài 07 không có bước này,
    mọi câu hỏi đều đi qua cùng một đường truy hồi văn bản."""
    t = str(q).lower()
    ents = [e for e in link_entities(q) if series_of(e)]
    metric = next((k for k, ws in METRIC_WORDS.items() if any(w in t for w in ws)), None)
    causal = any(w in t for w in CAUSAL_WORDS)
    if metric:
        kind = "numeric"
    elif causal:
        kind = "causal"
    else:
        kind = "narrative"
    return {"kind": kind, "metric": metric, "entities": ents,
            "period": parse_period(q), "text": q}


def answer_numeric(prices, parsed, full_range=("2020-01-01", "2026-10-15")):
    """Trả lời câu hỏi số bằng cách TÍNH trên bảng giá, không truy hồi văn bản."""
    if not parsed["entities"]:
        return {"value": None, "reason": "không nhận ra tài sản nào trong câu hỏi"}
    start, end = parsed["period"] or (pd.Timestamp(full_range[0]), pd.Timestamp(full_range[1]))
    metric = parsed["metric"]
    asset = parsed["entities"][0]
    other = parsed["entities"][1] if len(parsed["entities"]) > 1 else None
    if metric == "correlation" and other is None:
        return {"value": None, "reason": "câu hỏi tương quan cần hai tài sản"}
    val = compute(prices, metric, series_of(asset), start, end,
                  other=series_of(other) if other else None)
    return {"value": val, "metric": metric, "asset": asset, "other": other,
            "window": (str(start.date()), str(end.date()))}


# ------------------------- mốc so sánh: cách làm của bài 07 (bảng -> câu -> truy hồi)

def make_stats_corpus(prices, freq="YE"):
    """Dựng lại đúng cách bài 07 xử lý bảng số: mỗi tài sản mỗi năm một câu.

    Đây là mốc để so sánh. Hạn chế lộ ra ngay từ cách dựng: chỉ những phép tính
    đã viết sẵn thành câu mới trả lời được, hỏi theo quý hay theo tháng là chịu.
    """
    docs = []
    for period, chunk in prices.groupby(pd.Grouper(freq=freq)):
        if len(chunk) < MIN_DAYS:
            continue
        s, e = chunk.index[0], chunk.index[-1]
        for col in prices.columns:
            r = metric_return(chunk, col, s, e)
            v = metric_volatility(chunk, col, s, e)
            d = metric_drawdown(chunk, col, s, e)
            if r is None or v is None or d is None:
                continue
            last = chunk[col].dropna()
            docs.append({
                "text": ("%s in %d: price return %+.1f%%, annualized volatility "
                         "%.1f%%, maximum drawdown %.1f%%, last price %.2f."
                         % (col, period.year, 100 * r, 100 * v, 100 * d,
                            float(last.iloc[-1]))),
                "source": "stats:%s:%d" % (col, period.year),
                "entities": [col], "year": period.year,
            })
    return docs


# ------------------------------------------- truy hồi tin tức có lọc thời gian

class TimeScopedRetriever:
    """Truy hồi tin tức, lọc theo cửa sổ thời gian TRƯỚC khi tính tương đồng.

    Bỏ bước lọc thì câu hỏi về tháng 3 có thể được trả lời bằng bài viết tháng
    9 nhìn lại. Với bài tập thì sai phương pháp, với hệ thật dùng để ra quyết
    định thì là rò rỉ nhìn trước: câu trả lời dùng thông tin chưa tồn tại ở
    thời điểm được hỏi.
    """

    def __init__(self, news, use_embeddings=True, model_name="all-MiniLM-L6-v2"):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.news = news.reset_index(drop=True)
        self.titles = self.news["title"].astype(str).tolist()
        self.dates = pd.to_datetime(self.news["date"], utc=True)
        self.mode = "tfidf"
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                try:                      # tắt thanh tiến trình cho output gọn
                    from transformers.utils import logging as hf_log

                    hf_log.disable_progress_bar()
                    hf_log.set_verbosity_error()
                except Exception:
                    pass
                self.model = SentenceTransformer(model_name)
                self.emb = self.model.encode(self.titles, normalize_embeddings=True,
                                             batch_size=64, show_progress_bar=False)
                self.mode = "hybrid"
            except Exception as e:
                print("Không dùng được embedding, lùi về TF-IDF:", type(e).__name__)
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                   sublinear_tf=True)
        self.tfidf = self.vec.fit_transform(self.titles)

    def _scores(self, q, expand=True):
        qq = expand_query(q) if expand else q
        lex = (self.tfidf @ self.vec.transform([qq]).T).toarray().ravel()
        if self.mode == "hybrid":
            return self.emb @ self.model.encode([qq], normalize_embeddings=True)[0] + 0.5 * lex
        return lex

    def retrieve(self, q, window=None, k=5, time_filter=True, expand=True):
        """window là cặp (start, end); time_filter=False để đo tác hại của việc
        bỏ lọc thời gian."""
        s = self._scores(q, expand=expand)
        mask = np.ones(len(s), dtype=bool)
        if window and time_filter:
            start, end = [pd.Timestamp(x, tz="UTC") if pd.Timestamp(x).tz is None
                          else pd.Timestamp(x) for x in window]
            mask = ((self.dates >= start) & (self.dates <= end)).to_numpy()
        if mask.sum() == 0:
            return []
        idx = np.argsort(-np.where(mask, s, -np.inf))[:min(k, int(mask.sum()))]
        return [{"i": int(i), "score": float(s[i]), "title": self.titles[i],
                 "date": self.dates.iloc[i], "source": self.news.loc[i, "source"]}
                for i in idx]


def leakage_rate(retriever, questions, k=5):
    """Tỉ lệ tài liệu được truy hồi có ngày đăng SAU cửa sổ câu hỏi, khi tắt
    lọc thời gian. Đây là phần câu trả lời dựa trên thông tin của tương lai."""
    total = late = 0
    for q, window in questions:
        end = pd.Timestamp(window[1], tz="UTC")
        for h in retriever.retrieve(q, window, k=k, time_filter=False):
            total += 1
            late += int(h["date"] > end)
    return (late / total if total else 0.0), total


# ------------------------------------------------ mở rộng truy vấn Việt - Anh

# Kho tin là tiếng Anh, câu hỏi là tiếng Việt. Không bắc cầu thì truy hồi gần
# như vô dụng: token duy nhất của câu hỏi lọt vào từ vựng TF-IDF là chữ số năm,
# còn MiniLM là mô hình tiếng Anh nên véc-tơ của câu tiếng Việt không mang
# nghĩa. Đây đúng là lỗi đã bắt được ở mốc so sánh A, nên phải sửa cho cả hai.
VI_EN = {
    "giảm": "falls drops lower decline",
    "tăng": "rises gains higher climbs",
    "biến động": "volatility swings",
    "vì sao": "why", "tại sao": "why", "nguyên nhân": "driver reason",
    "kỷ lục": "record high", "trú ẩn": "safe haven",
    "tin tức": "news", "dự báo": "forecast outlook",
}


def expand_query(q):
    """Thêm từ khoá tiếng Anh tương ứng vào truy vấn tiếng Việt.

    Gồm hai nguồn: tên tiếng Anh của các thực thể nhận ra được, và bảng từ khoá
    hướng/ý định ở trên."""
    t = str(q).lower()
    extra = []
    for node in link_entities(q):
        al = ENTITIES[node]["aliases"]
        extra.append(al[0])                       # cách gọi tiếng Anh chính
        extra.append(node)
    for vi, en in VI_EN.items():
        if vi in t:
            extra.append(en)
    return q + " " + " ".join(extra) if extra else q


# --------------------------------------------------- đường trả lời hoàn chỉnh

def answer(q, prices, retriever, k=5):
    """Trả lời một câu hỏi, đi theo nhánh mà bộ định tuyến chọn.

    Câu hỏi số  -> tính thẳng trên bảng giá.
    Câu hỏi nhân quả -> truy hồi tin trong đúng cửa sổ, trích khẳng định của
        báo chí, rồi đối chiếu từng khẳng định với tương quan tính từ giá.
    Câu hỏi diễn giải -> truy hồi trong cửa sổ và trả về kèm nguồn.

    Câu trả lời cho nhánh nhân quả có hai tầng, và đó là điểm khác biệt so với
    RAG thường: tầng một là báo chí nói gì, tầng hai là dữ liệu có chống đỡ lời
    đó không. RAG thường chỉ có tầng một và trình bày nó như thể là kết luận.
    """
    parsed = parse_question(q)
    window = parsed["period"]

    if parsed["kind"] == "numeric":
        out = answer_numeric(prices, parsed)
        return {"loai": "số", "ket_qua": out, "cau_hoi": q}

    hits = retriever.retrieve(q, window, k=k) if window else retriever.retrieve(q, k=k)
    if parsed["kind"] == "narrative":
        return {"loai": "diễn giải", "cau_hoi": q, "tai_lieu": hits,
                "nguon": sorted({h["source"] for h in hits if h.get("source")})}

    # nhánh nhân quả
    claims, verdicts = [], []
    for h in hits:
        for c in extract_claims(h["title"], asset="GOLD"):
            c["date"] = h["date"]
            c["source"] = h.get("source", "")
            claims.append(c)
            verdicts.append(verify_claim(prices, c))

    bao_chi = collections.Counter(c["driver"] for c in claims)
    doi_chieu = []
    for c, v in zip(claims, verdicts):
        doi_chieu.append({"nguyen_nhan": c["driver"], "ket_luan": v["verdict"],
                          "corr": v.get("corr"), "ly_do": v.get("reason"),
                          "tieu_de": c["title"]})
    khong_do_duoc = sorted({c["driver"] for c in claims if series_of(c["driver"]) is None})

    return {"loai": "nhân quả", "cau_hoi": q,
            "cua_so": (str(window[0].date()), str(window[1].date())) if window else None,
            "bao_chi_quy_cho": bao_chi.most_common(),
            "doi_chieu_du_lieu": doi_chieu,
            "nguyen_nhan_khong_do_duoc_bang_gia": khong_do_duoc,
            "tai_lieu": hits,
            "nguon": sorted({h["source"] for h in hits if h.get("source")})}


# ----------------------------------------------------- miền chứng khoán

TICKERS = {
    "VNM": ["vinamilk", "vnm", "vietnam dairy"],
    "FPT": ["fpt"],
    "VCB": ["vietcombank", "vcb", "joint stock commercial bank for foreign trade"],
    "HPG": ["hoa phat", "hpg", "hoa phat group"],
    "VIC": ["vingroup", "vic"],
}


def load_company_info(name="company_info.json"):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def belongs_to_edges(info):
    """Cạnh `belongs_to` lấy từ dữ liệu tham chiếu: mã -> ngành -> quốc gia.

    Khác hẳn hai loại cạnh kia về độ tin cậy: đây là sự thật tra cứu được,
    không phải thứ tính ra từ giá cũng không phải thứ nhà báo nói. Giữ riêng
    cả ba loại là để người đọc biết mỗi cạnh đến từ đâu."""
    rows = []
    for tic, meta in info.items():
        code = tic.split(".")[0]
        for field, rel in (("industry", "thuộc ngành"), ("sector", "thuộc nhóm"),
                           ("country", "ở nước")):
            if meta.get(field):
                rows.append({"src": code, "dst": meta[field], "rel": "belongs_to",
                             "chi_tiet": rel, "period": "tĩnh"})
    return pd.DataFrame(rows)


def stock_entities():
    """Bổ sung các mã cổ phiếu vào bảng thực thể để dùng chung bộ máy với vàng."""
    out = dict(ENTITIES)
    for code, al in TICKERS.items():
        out[code] = dict(kind="Company", series=code, aliases=al + [code.lower()])
    return out
