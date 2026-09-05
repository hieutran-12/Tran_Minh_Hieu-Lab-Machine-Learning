## 2. Về lựa chọn dữ liệu

Bài nộp có 2 nguồn dữ liệu:

1. **`data/advertising.csv`** (200 dòng, TV/Radio/Newspaper/Sales) — dữ liệu thật đính kèm.
   Kiểm tra tương quan: `r(TV,Radio)=0.05`, `r(TV,Newspaper)=0.06`, `r(Radio,Newspaper)=0.35`
   → **không có đa cộng tuyến đáng kể**. Bộ này chỉ dùng để đối chiếu (mục 1 trong notebook/script),
   vì nếu dùng nó làm dữ liệu chính thì không thể minh hoạ được tác dụng cốt lõi của Ridge.

2. **Dữ liệu mô phỏng** (500 dòng, TV/Facebook/Google → DoanhThu), sinh đúng theo công thức
   khuyến nghị ở mục 3 README gốc, với Facebook được đặt tương quan cao với TV
   (`fb = 0.6*tv + noise`, r ≈ 0.98) — mô phỏng tình huống "chiến dịch lớn thì bơm tiền tất cả
   kênh cùng lúc". Đây là dữ liệu dùng cho toàn bộ phân tích chính (VIF, bootstrap, coefficient
   path, RidgeCV, đề xuất ngân sách).

---

## 3. Kết quả chính

### 3.1. VIF — bằng chứng đa cộng tuyến

| Biến     | VIF   |
| -------- | ----- |
| TV       | 26.44 |
| Facebook | 26.44 |
| Google   | 1.00  |

TV và Facebook đều VIF ≈ 26 (>> ngưỡng 10) → đa cộng tuyến nghiêm trọng giữa hai kênh này.

### 3.2. ⭐ Vì sao Ridge ổn định hơn Linear (bootstrap 100 lần, 80% dữ liệu/lần)

| Biến     | Std hệ số (Linear) | Std hệ số (Ridge, α=10) | Giảm dao động                                 |
| -------- | ------------------ | ----------------------- | --------------------------------------------- |
| TV       | 7.08               | 4.06                    | **−42.6%**                                    |
| Facebook | 5.92               | 3.91                    | **−33.9%**                                    |
| Google   | 2.34               | 2.31                    | −1.5% (không đổi, vì không tương quan với ai) |

**Giải thích cơ chế:** khi TV và Facebook tương quan cao, Linear Regression phải "chia" cùng
một phần thông tin về doanh thu cho hai biến gần như trùng nhau — bài toán trở nên gần suy biến
(ill-conditioned), khiến hệ số cực kỳ nhạy với việc mẫu dữ liệu thay đổi (chỉ cần bớt/thêm vài
dòng dữ liệu, hệ số TV có thể nhảy vọt hoặc thậm chí đổi dấu). Số hạng phạt `λ·Σwᵢ²` trong Ridge
kéo các hệ số lớn về gần 0 một cách đồng đều, làm bài toán tối ưu "well-conditioned" hơn → hệ số
ổn định qua các mẫu khác nhau. Biến Google không tương quan với ai nên không có gì để Ridge phải
ổn định — đúng như thực nghiệm cho thấy.

Xem `reports/bootstrap_he_so.png`.

### 3.3. Coefficient path

`reports/coefficient_path.png` — không hệ số nào bị đưa về đúng 0 dù α tăng đến 10⁴ (khác Lasso,
vốn dùng phạt L1 và có thể đưa hệ số về đúng 0). Đáng chú ý, hệ số Facebook **tăng nhẹ** trước khi
co lại: khi TV bị phạt giảm, Ridge san sẻ một phần "trách nhiệm giải thích doanh thu" sang Facebook
vì hai biến mang thông tin gần giống nhau.

### 3.4. RMSE theo alpha & so sánh Linear vs Ridge

`reports/rmse_theo_alpha.png` cho thấy RMSE gần như phẳng từ α ≈ 0.001 đến α ≈ 10–30, rồi mới tăng
dần. RidgeCV (5-fold CV trên tập train, không đụng vào tập test) chọn **α ≈ 0.001** — gần như không
phạt.

| Model                | RMSE test |
| -------------------- | --------- |
| Linear Regression    | 47.946    |
| Ridge (α từ RidgeCV) | 47.946    |

**Vì sao RMSE gần như không đổi?** Với 500 dòng dữ liệu cho chỉ 3 biến và nhiễu tương đối nhỏ so
với tín hiệu, Linear Regression đã có phương sai thấp sẵn — Ridge không còn nhiều "phương sai dư
thừa" để cắt giảm cho mục tiêu dự đoán. Đây là một bài học quan trọng và trung thực (không phải kết
quả bị "làm đẹp"): **lợi ích chính của Ridge trong bài toán này không phải là giảm RMSE dự đoán, mà
là làm hệ số ổn định và diễn giải được** — đúng thứ CMO cần để ra quyết định phân bổ ngân sách, khác
với chỉ cần dự đoán doanh thu chính xác. Vùng RMSE phẳng (α đến ~10–30) cũng cho thấy ta có thể "mua"
sự ổn định hệ số gần như miễn phí về mặt độ chính xác — đó là lý do bootstrap ở trên dùng α=10 làm
minh hoạ.

### 3.5. ✍️ Đề xuất phân bổ ngân sách (dựa trên |hệ số Ridge| đã chuẩn hoá)

Giả định ngân sách 2 tỷ VND/tháng như ví dụ trong bài:

| Kênh     | Hệ số Ridge (chuẩn hoá) | Tỷ lệ đề xuất | Số tiền           |
| -------- | ----------------------- | ------------- | ----------------- |
| TV       | 406.9                   | 53.9%         | 1,078,000,000 VND |
| Google   | 206.7                   | 27.4%         | 548,000,000 VND   |
| Facebook | 141.0                   | 18.7%         | 374,000,000 VND   |

---

## 4. Hạn chế cần nêu rõ

- **Hệ số hồi quy (kể cả Ridge) phản ánh tương quan, không phải quan hệ nhân quả.** Để khẳng định
  kênh nào thực sự _tạo ra_ doanh thu tăng thêm, cần thí nghiệm thật (geo-test, A/B test theo khu
  vực/thời điểm) — hệ số ở đây chỉ nên dùng làm điểm khởi đầu cho giả thuyết, không phải kết luận
  cuối cùng để dồn toàn bộ ngân sách.
- Mô hình chưa tính **hiệu ứng trễ (adstock)**: quảng cáo TV hôm nay có thể ảnh hưởng doanh thu
  tuần sau, không chỉ ngày hôm đó.
- Mô hình chưa tính **hiệu ứng bão hoà**: chi gấp đôi ngân sách một kênh không tạo ra doanh thu
  gấp đôi (lợi suất giảm dần).
- Dữ liệu mô phỏng là tuyến tính theo thiết kế — trên dữ liệu thật, quan hệ giữa ngân sách và
  doanh thu có thể phi tuyến, cần thử thêm biến đổi log hoặc mô hình phi tuyến.

## 5. Mở rộng có thể làm tiếp (chưa triển khai trong bài này)

1. Thêm adstock: `x_t + 0.5·x_{t-1} + 0.25·x_{t-2}`.
2. Thêm hiệu ứng bão hoà: dùng `log(1+x)` thay vì `x`.
3. So sánh với Bayesian Ridge để có khoảng tin cậy cho từng hệ số.
4. TT-13 (Lasso) và TT-14 (ElasticNet) để so sánh 3 kiểu regularization trên cùng bộ dữ liệu.
