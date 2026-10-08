"""
ERP 주문 데이터 → 시뮬레이션 수요 입력

원천 데이터: UCI Online Retail II (영국 온라인 도매업체 실거래, 2009-12 ~ 2011-12, CC BY 4.0)
  https://archive.ics.uci.edu/dataset/502/online+retail+ii

1) prepare(): 원천 xlsx → ERP 주문 export 형식 CSV (data/erp_orders.csv)
     order_id, order_datetime, customer_id, country, n_lines, quantity
   - 취소 주문(C로 시작), 수량·단가 ≤ 0 라인, 중복 라인 제외 후 주문(Invoice) 단위로 집계
   - 다른 ERP 데이터를 쓸 때는 이 형식으로 export한 CSV만 있으면 됨

2) ERPOrderBook: 주문 CSV → 시뮬레이션 주문 스트림 [(도착 시각, 작업량), ...]
   - 운영 시간축: 주문이 있는 영업일의 07:00~21:00만 이어 붙임, 1 영업일 = SIM_PER_DAY sim time
   - 로트 분할: 주문 수량이 MAX_LOT_QTY를 넘으면 균등 분할 (MRP 로트 사이징)
   - Job Size = 로트 수량 / 학습 기간 평균 로트 수량 (평균 1.0)
   - 타임스탬프가 분 단위라 같은 분의 주문은 그 1분 안에 균등하게 흩뿌림
"""
import csv, math, os, sys
from datetime import date, datetime
import numpy as np

DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "erp_orders.csv")
DAY_START_HOUR, DAY_END_HOUR = 7, 21
SIM_PER_DAY = 70.0      # 1 sim time = 12분, 의사결정 주기 5 = 1시간
MAX_LOT_QTY = 800          # 주문 수량 상위 5% 지점
SPLIT_DATE = date(2011, 7, 1)   # 이전: 학습 / 이후: 평가


def prepare(xlsx_path: str, out_csv: str = DATA_CSV):
    import pandas as pd
    sheets = pd.read_excel(xlsx_path, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    ok = df[~df.Invoice.str.startswith("C") & (df.Quantity > 0) & (df.Price > 0)].drop_duplicates()
    orders = ok.groupby("Invoice").agg(order_datetime=("InvoiceDate", "min"),
                                       customer_id=("Customer ID", "first"),
                                       country=("Country", "first"),
                                       n_lines=("StockCode", "count"),
                                       quantity=("Quantity", "sum"))
    orders = orders.sort_values("order_datetime").reset_index().rename(columns={"Invoice": "order_id"})
    orders["customer_id"] = orders["customer_id"].astype("Int64")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    orders.to_csv(out_csv, index=False)
    print(f"{len(df):,} lines → {len(orders):,} orders → {out_csv}")


class ERPOrderBook:
    def __init__(self, path: str = DATA_CSV):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 없음. 먼저 `python erp_data.py <online_retail_II.xlsx>` 실행")
        day_list, minute, qty, oid = [], [], [], []
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                dt = datetime.fromisoformat(r["order_datetime"])
                m = (dt.hour - DAY_START_HOUR) * 60 + dt.minute
                day_list.append(dt.date())
                minute.append(min(max(m, 0), (DAY_END_HOUR - DAY_START_HOUR) * 60 - 1))
                qty.append(float(r["quantity"]))
                oid.append(r["order_id"])
        self.days = sorted(set(day_list))
        idx = {d: i for i, d in enumerate(self.days)}

        # 로트 분할
        o_day = np.array([idx[d] for d in day_list])
        o_min = np.array(minute, dtype=float)
        o_qty = np.array(qty)
        n_lots = np.ceil(o_qty / MAX_LOT_QTY).astype(int)
        self.lot_order = np.repeat(np.arange(len(o_qty)), n_lots)
        self.lot_day = o_day[self.lot_order]
        self.lot_min = o_min[self.lot_order]
        self.lot_qty = (o_qty / n_lots)[self.lot_order]

        self.train_days = [i for i, d in enumerate(self.days) if d < SPLIT_DATE]
        self.test_days = [i for i, d in enumerate(self.days) if d >= SPLIT_DATE]
        train_mask = self.lot_day < len(self.train_days)
        self.qty_unit = float(self.lot_qty[train_mask].mean())
        self.mean_rate = train_mask.sum() / (len(self.train_days) * SIM_PER_DAY)

    def window(self, start_day: int, n_days: int, rng: np.random.Generator) -> np.ndarray:
        """start_day부터 n_days 영업일의 주문을 [(sim 도착시각, 작업량)] 배열로 반환"""
        sel = (self.lot_day >= start_day) & (self.lot_day < start_day + n_days)
        _, inv = np.unique(self.lot_order[sel], return_inverse=True)
        jitter = rng.random(inv.max() + 1 if len(inv) else 0)[inv]   # 같은 주문의 로트는 동시에 도착
        minutes_per_day = (DAY_END_HOUR - DAY_START_HOUR) * 60
        t = (self.lot_day[sel] - start_day) * SIM_PER_DAY + \
            (self.lot_min[sel] + jitter) / minutes_per_day * SIM_PER_DAY
        size = self.lot_qty[sel] / self.qty_unit
        o = np.argsort(t, kind="stable")
        return np.column_stack([t[o], size[o]])

    def windows(self, days, n_days):
        """겹치지 않는 n_days 단위 구간의 시작 인덱스 목록"""
        return [days[i] for i in range(0, len(days) - n_days + 1, n_days)]

    def period(self, start_day: int, n_days: int) -> str:
        return f"{self.days[start_day]} ~ {self.days[start_day + n_days - 1]}"


_BOOK = None


def order_book() -> ERPOrderBook:
    global _BOOK
    if _BOOK is None:
        _BOOK = ERPOrderBook()
    return _BOOK


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python erp_data.py <online_retail_II.xlsx>")
    prepare(sys.argv[1])
    b = ERPOrderBook()
    print(f"영업일 {len(b.days)}일 (학습 {len(b.train_days)} / 평가 {len(b.test_days)}), "
          f"로트 {len(b.lot_qty):,}개, 평균 로트 수량 {b.qty_unit:.1f}, 학습기간 평균 도착률 {b.mean_rate:.3f}/sim time")
