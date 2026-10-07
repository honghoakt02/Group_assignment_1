# HOSE V4: Dashboard Nghiên Cứu Định Lượng & Nested Walk-Forward Portfolio Optimization

> **Dự án Nghiên cứu Tài chính Định lượng (Quantitative Finance Thesis & Research Framework)**  
> **Chủ đề:** Hệ thống Đa nhân tố Kỹ thuật & Cơ bản kết hợp Khung Nested Walk-Forward, Tối ưu Danh mục (EW, MPT, MinVar, HRP) và Cơ chế Market Regime trên sàn HOSE (2021 – 2025).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 1. Giới thiệu Tổng quan về HOSE V4

Hệ thống **HOSE V4** được phát triển nhằm giải quyết triệt để các hạn chế của các mô hình định lượng truyền thống trên thị trường chứng khoán Việt Nam:
1. **Khắc phục Lookahead Bias (Rò rỉ tương lai):** Thiết kế đường ống phân chia 3 tập thời gian độc lập: **Core Train (60%) → Validation (20%) → Final Test (20%)**. Tín hiệu tính toán tại giá đóng cửa $Close(t-1)$ chỉ được phép gửi lệnh giao dịch tại giá mở cửa $Open(t)$.
2. **Cơ chế Nested Walk-Forward theo Quý:** Thay vì khóa cứng danh mục Top N trong cả năm, hệ thống tái chọn lọc danh mục Top 5 tại mỗi quý bằng dữ liệu chỉ tính đến phiên liền trước, giúp danh mục linh hoạt luân chuyển theo sóng ngành thị trường.
3. **So sánh Đa Mô hình Phân bổ Vốn:** Cạnh tranh trực tiếp giữa **Equal Weight (EW)**, **Modern Portfolio Theory với Ledoit-Wolf Shrinkage (MPT-LW)**, **Minimum Variance (MinVar-LW)**, **Hierarchical Risk Parity (HRP)**, và **Blend50**.
4. **Cơ chế Quản trị Rủi ro Market Regime:** Tự động phòng thủ đưa tỷ trọng về Tiền mặt (Cash) khi chỉ số chung gãy xu hướng dài hạn ($EMA200$ và độ dốc âm).

---

## 🏆 2. Tóm tắt Kết quả Thực nghiệm Out-Of-Sample (Năm 2025)

| Mô hình | Vai trò trong Nghiên cứu | Return [%] | CAGR [%] | Sharpe | Sortino | MaxDD [%] | Turnover [%] | Tiền mặt TB [%] | Excess CAGR vs VN-Index |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DYN_EW_NO_REGIME** | *Pure Stock Selection Alpha* | **+56.79%** | **+57.23%** | **1.691** | **2.124** | -25.51% | 523.8% | 0.00% | **+42.93 pp** |
| **STATIC_EW_BH** | *Internal Benchmark Buy & Hold* | **+21.61%** | **+21.76%** | **0.980** | **1.132** | -17.65% | 99.9% | 0.00% | **+7.46 pp** |
| **VNINDEX (Market Proxy)** | *Thị trường Chung* | **+14.21%** | **+14.30%** | **0.690** | **0.788** | -21.90% | — | — | **0.00 pp** |
| **DIAG_DYN_MPT_LW_REGIME** | *Diagnostic MPT Shrinkage* | **+12.01%** | **+12.09%** | **0.577** | **0.589** | -22.31% | 1653.6% | 38.28% | **-2.21 pp** |
| **DIAG_DYN_BLEND50_REGIME** | *Diagnostic 50% MPT + 50% EW* | **+2.23%** | **+2.24%** | **0.215** | **0.207** | -24.28% | 1363.5% | 39.02% | **-12.06 pp** |
| **DIAG_DYN_HRP_REGIME** | *Diagnostic HRP Risk Parity* | **-0.28%** | **-0.28%** | **0.095** | **0.090** | -22.42% | 1171.3% | 38.95% | **-14.58 pp** |
| **DIAG_DYN_MINVAR_REGIME** | *Diagnostic Minimum Variance* | **-2.24%** | **-2.25%** | **-0.007** | **-0.007** | -21.53% | 1091.6% | 38.44% | **-16.55 pp** |
| **LOCKED_CHAMPION (DYN_EW_REGIME)** | *Khóa từ Validation (Official)* | **-7.41%** | **-7.45%** | **-0.222** | **-0.210** | -26.36% | 1089.5% | **39.77%** | **-21.75 pp** |

> 💡 **Điểm sáng học thuật:** Thuật toán chọn cổ phiếu đa nhân tố tạo ra mức alpha vượt bậc (+56.79% vs +14.21% VN-Index). Đồng thời nghiên cứu thảo luận thẳng thắn hiện tượng **"Cash Drag"** của bộ lọc Market Regime trong thị trường sideway biên độ hẹp năm 2025.

---

## 📂 3. Cấu trúc Thư mục Repository

```text
├── app.py                                                  # Ứng dụng Web Dashboard tương tác (Streamlit)
├── requirements.txt                                        # Danh mục thư viện cần cài đặt
├── README.md                                               # Tài liệu hướng dẫn chi tiết dự án
├── HOSE-01.01.2021-31.12.2025.csv                          # Bộ dữ liệu 95 mã cổ phiếu HOSE (14.9 MB)
└── HOSE_V4_GENERIC_NESTED_WALK_FORWARD_STEP_BY_STEP_VI.ipynb # Notebook nghiên cứu gốc
```

---

## 🚀 4. Hướng dẫn Chạy Ứng dụng Cục bộ (Local Run)

### Yêu cầu hệ thống:
- Python 3.9, 3.10, 3.11 hoặc 3.12.
- Git (nếu clone từ repository).

### Bước 1: Clone hoặc tải repository về máy
```bash
git clone https://github.com/<your-username>/hose-v4-quant-dashboard.git
cd hose-v4-quant-dashboard
```

### Bước 2: Tạo môi trường ảo (Khuyến nghị)
```bash
python -m venv venv
# Trên Windows:
venv\Scripts\activate
# Trên macOS / Linux:
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 4: Khởi chạy Streamlit Dashboard
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở tại địa chỉ `http://localhost:8501`.

---

## 🌐 5. Hướng dẫn Đẩy lên GitHub & Deploy Miễn phí trên Streamlit Community Cloud

### Bước 1: Khởi tạo Git và Commit file
Mở Terminal trong thư mục dự án và chạy các lệnh sau:
```bash
git init
git add app.py requirements.txt README.md HOSE-01.01.2021-31.12.2025.csv
git commit -m "feat: Initial commit for HOSE V4 Quantitative Dashboard"
```

### Bước 2: Tạo Repository trên GitHub
1. Truy cập [https://github.com](https://github.com) và đăng nhập.
2. Nhấn nút **New repository** (hoặc dấu `+` ở góc trên bên phải).
3. Đặt tên repository (ví dụ: `hose-v4-dashboard`). Chọn chế độ **Public**.
4. Nhấn **Create repository**.

### Bước 3: Đẩy mã nguồn lên GitHub
Sao chép các lệnh mà GitHub hướng dẫn (thay `<your-username>` bằng tên tài khoản của bạn):
```bash
git branch -M main
git remote add origin https://github.com/<your-username>/hose-v4-dashboard.git
git push -u origin main
```

### Bước 4: Deploy trên Streamlit Community Cloud
1. Truy cập [https://share.streamlit.io/](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"New app"** (hoặc **"Create app"**).
3. Chọn:
   - **Repository:** `<your-username>/hose-v4-dashboard`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Nhấn **"Deploy!"**.
5. Streamlit sẽ tự động cài đặt `requirements.txt` và khởi chạy web app trong khoảng 1–2 phút. Bạn sẽ nhận được đường link web công khai để chia sẻ cho hội đồng phản biện và giảng viên!

---

## 🧭 6. Các Tính năng Chi tiết trong Dashboard

Dashboard được thiết kế theo cấu trúc mô-đun khoa học bao gồm **16 chuyên trang tương tác**:
1. **🏛️ Tổng quan & Luận điểm:** Các thẻ chỉ số hiệu năng (KPIs), biểu đồ tăng trưởng NAV tương tác và tóm lược luận điểm.
2. **📊 Dữ liệu Đầu vào:** Bảng kiểm tra chất lượng dữ liệu 95 mã HOSE, số phiên giao dịch, tỷ lệ thiếu volume, giá trị giao dịch trung vị.
3. **⏳ Chia tập & Leakage Audit:** Bảng phân chia Core Train/Val/Test và ma trận kiểm toán 7 bước chống rò rỉ dữ liệu.
4. **🔍 Phân tích Cơ bản (FA):** Thuyết minh nguyên tắc Point-in-time và cơ chế tự động chuyển đổi TA-only khi thiếu ngày công bố BCTC an toàn.
5. **📈 Phân tích Kỹ thuật & TA Score:** Bóc tách 4 trụ cột (Trend, Momentum, Volume, Volatility) và bảng điểm mẫu.
6. **🎯 Tối ưu Ngưỡng BUY / EXIT:** Bản đồ nhiệt quét lưới 3×3 tham số trên Core Train với phạt độ lệch chuẩn (Stability Penalty).
7. **🏆 Mô hình Xếp hạng Cổ phiếu:** Công thức chấm điểm đa nhân tố, 3 cổng lọc rủi ro biến động và trần tương quan cặp ≤ 0.80.
8. **🔄 Nested Walk-Forward & Top N:** Bảng lịch sử luân chuyển danh mục Top 5 theo từng quý trong Validation (2024) và Final Test (2025).
9. **⚖️ So sánh Các Mô hình Phân bổ:** Bảng đua tài giữa EW, MPT-LW, MinVar-LW, HRP và Blend50 trên tập Validation để khóa Champion.
10. **🛡️ Market Regime & Tiền mặt:** Công thức EMA200 + Slope confirmation, cơ chế giữ tiền mặt và quy tắc không tái phân bổ vốn.
11. **📉 Đánh giá Final Test & Benchmark:** Báo cáo 11 chỉ số định lượng đầy đủ (Return, CAGR, Sharpe, Sortino, Calmar, MaxDD, Turnover, Excess Return).
12. **🌊 Phân tích Drawdown:** Biểu đồ Underwater Drawdown đo lường độ sâu và thời gian sụt giảm của từng chiến lược.
13. **💸 Turnover & Chi phí Giao dịch:** Đánh giá tác động của phí hoa hồng 0.15% và trượt giá 0.05% lên hiệu quả thực tế.
14. **📜 Sổ lệnh & Giao dịch Chi tiết:** Bộ lọc tra cứu toàn bộ các lệnh mua/bán khớp tại giá Open theo chuẩn Event-Driven.
15. **📋 Trung tâm Xuất Báo cáo:** Tải xuống từng bảng CSV hoặc xuất toàn bộ 8 sheet kết quả vào một file Excel đa sheet (`HOSE_V4_GENERIC_RESULTS.xlsx`).
16. **🎓 Hướng dẫn Thuyết trình Luận văn:** Kịch bản 6 bước bảo vệ luận văn chuẩn mực trước hội đồng chấm đề tài.

---

## 👨‍💻 7. Thông tin Tác giả & Bản quyền

- **Môn học / Đề tài:** Nghiên cứu Định lượng Danh mục Đầu tư Chứng khoán HOSE.
- **Phiên bản:** HOSE V4 Generic Nested Walk-Forward Framework.
- **Bản quyền:** Mã nguồn được phân phối theo giấy phép mã nguồn mở MIT. Bạn hoàn toàn có thể tự do mở rộng, sử dụng cho luận văn tốt nghiệp hoặc nghiên cứu độc lập.
