# TT-10 — MLP CLASSIFIER (sklearn)

## Đọc số viết tay trên séc / phiếu chuyển khoản

## 1. Mô tả

Bài toán: nhận diện chữ số viết tay (0-9) trên séc/phiếu chuyển khoản bằng
`MLPClassifier` (Multi-Layer Perceptron) của scikit-learn, có cơ chế
**human-in-the-loop** (chuyển cho người kiểm tra khi model không đủ tự tin) vì
sai 1 chữ số trong số tiền là sai tiền thật.

**Nguồn chân lý của pipeline là `notebooks/mlp_digits.ipynb`** — notebook này
**độc lập hoàn toàn, không import từ file nào khác**, chạy được từ đầu đến
cuối (end-to-end, Run All) chỉ với `pip install -r requirements.txt`. Toàn
bộ 13 bước của đề (baseline, so sánh chuẩn hoá, 4 kiến trúc, loss curve, 3
activation, 3 learning rate, confusion matrix, ảnh dự đoán sai,
human-in-the-loop) nằm trong notebook.

`src/train.py` là **bản script được xuất tự động từ notebook** (Jupyter →
Export as Executable Script), để chạy không cần Jupyter, ví dụ khi huấn
luyện trên server. Nếu cần sửa logic, sửa trong notebook trước rồi export lại
— đừng sửa hai nơi lệch nhau.

## 2. Cài đặt

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Chạy

### Cách 1 — Notebook (khuyến nghị, xem từng bước + biểu đồ trực tiếp)

```bash
jupyter notebook notebooks/mlp_digits.ipynb
```

Mở cell cấu hình đầu tiên, chỉnh:

- `DATASET = "mnist"` (mặc định, chính thức nộp bài) — lần đầu chạy sẽ tải MNIST
  từ OpenML (~1-2 phút tuỳ mạng) rồi cache lại.
- `DATASET = "digits"` — chạy nhanh trên `load_digits()` (1.797 ảnh 8x8) để
  kiểm tra pipeline khi không có mạng hoặc máy yếu.
- `QUICK = True` — nếu `DATASET = "mnist"` nhưng máy yếu, chỉ lấy 6.000 mẫu để
  chạy thử nhanh trước khi chạy full.

Sau đó **Run All**.

### Cách 2 — Script (không cần Jupyter)

```bash
python src/train.py
```

Chỉnh `DATASET` / `QUICK` / `CONFIDENCE_THRESHOLD` ngay trong phần cấu hình ở
đầu file `src/train.py`.

---

Kết quả (bảng số liệu `results_summary.json`, các ảnh loss curve/confusion
matrix/ảnh dự đoán sai) được lưu vào `reports/`, model chính (`(128, 64)`) lưu
vào `models/mlp_pipeline.joblib`.

> **Lưu ý:** repo này không kèm sẵn model/ảnh/số liệu — bạn cần tự chạy trên
> máy mình để `reports/` và `models/` được sinh ra. Cả notebook lẫn script đã
> được chạy thử end-to-end (13 bước, không lỗi) với `DATASET="digits"` trong
> lúc phát triển để đảm bảo logic đúng; nhánh `DATASET="mnist"` dùng chung
> code nên logic giống hệt, chỉ khác nguồn dữ liệu — bạn cần tự chạy vì máy
> phát triển ở đây không truy cập được OpenML.

## 4. Kết quả (điền sau khi chạy `--dataset mnist`)

### 4.1. So sánh CÓ / KHÔNG chuẩn hoá

| Cấu hình | Accuracy | Số vòng lặp (n_iter) | Ghi chú |
| -------- | -------- | -------------------- | ------- |

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>time_s</th>
      <th>n_iter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>logistic_regression</th>
      <td>0.866667</td>
      <td>0.032598</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mlp_khong_chuan_hoa</th>
      <td>0.966667</td>
      <td>0.652478</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>mlp_co_chuan_hoa</th>
      <td>0.905556</td>
      <td>1.611204</td>
      <td>66.0</td>
    </tr>
  </tbody>
</table>
</div>

### 4.2. So sánh kiến trúc

| Kiến trúc | Số tham số | Accuracy | Thời gian train (s) |
| --------- | ---------- | -------- | ------------------- |

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>architecture</th>
      <th>n_params</th>
      <th>accuracy</th>
      <th>train_time_s</th>
      <th>n_iter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(64,)</td>
      <td>4810</td>
      <td>0.7694</td>
      <td>0.13</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(128,)</td>
      <td>9610</td>
      <td>0.8972</td>
      <td>1.41</td>
      <td>100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(128, 64)</td>
      <td>17226</td>
      <td>0.9056</td>
      <td>3.75</td>
      <td>66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(256, 128, 64)</td>
      <td>58442</td>
      <td>0.9639</td>
      <td>4.02</td>
      <td>78</td>
    </tr>
  </tbody>
</table>
</div>

### 4.3. So sánh activation

| Activation | Accuracy | n_iter | Nhận xét |
| ---------- | -------- | ------ | -------- |

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activation</th>
      <th>accuracy</th>
      <th>n_iter</th>
      <th>train_time_s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>relu</td>
      <td>0.9056</td>
      <td>66</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tanh</td>
      <td>0.9444</td>
      <td>59</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>logistic</td>
      <td>0.1000</td>
      <td>14</td>
      <td>0.56</td>
    </tr>
  </tbody>
</table>
</div>

### 4.4. So sánh learning_rate_init

| learning_rate | Accuracy | n_iter |
| ------------- | -------- | ------ |

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>learning_rate</th>
      <th>accuracy</th>
      <th>n_iter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0100</td>
      <td>0.9639</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0010</td>
      <td>0.9056</td>
      <td>66</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0001</td>
      <td>0.0972</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>

### 4.5. Ma trận nhầm lẫn — cặp số hay nhầm nhất

Xem `reports/confusion_10x10.png`. Top 5 cặp nhầm lẫn: (điền từ
`reports/results_summary.json` → `top_cap_nham_lan`)

### 4.6. ⭐ Human-in-the-loop (ngưỡng tin cậy 99%)

| Ngưỡng | % séc tự động xử lý | Accuracy trên phần tự động | % chuyển cho người kiểm tra |
| ------ | ------------------- | -------------------------- | --------------------------- |
| ≥ 99%  |                     |                            |                             |

### 4.7. Accuracy cuối cùng

Model `(128, 64)` trên tập test MNIST: **accuracy = \_\_\_** (mục tiêu ≥ 0,96;
mức tham chiếu MLP: 0,97–0,98, CNN TT-26: > 0,99).

## 5. Hạn chế

MLP xử lý ảnh dưới dạng vector 1 chiều (flatten) nên **mất cấu trúc không
gian** của ảnh — không có tính bất biến dịch chuyển (translation invariance)
như CNN. Đây là lý do CNN (TT-26) thường vượt trội hơn trên bài toán ảnh.
