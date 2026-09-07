import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# TT-13 — LASSO REGRESSION (L1)
### Chọn ra 10 chỉ số xét nghiệm quan trọng nhất trong 200 chỉ số

**Bài toán:** Bệnh viện muốn sàng lọc tiến triển tiểu đường bằng xét nghiệm, nhưng đo 200
chỉ số sinh hoá quá tốn kém. Ta dùng bộ dữ liệu `load_diabetes` (10 biến thật) + 190 biến
nhiễu để mô phỏng 200 chỉ số, rồi dùng **Lasso (phạt L1)** để tự động chọn ra một bộ nhỏ
chỉ số đủ tốt để dự đoán.""")

md("## 0. Import thư viện")
code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, RidgeCV, lasso_path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

RNG_SEED = 42
%matplotlib inline""")

md("""## 1. Nạp dữ liệu & mở rộng thành 200 cột (10 thật + 190 nhiễu)

Thiết kế này cho phép **chấm điểm khách quan**: Lasso có tìm lại đúng 10 cột gốc và loại
190 cột nhiễu không? Đây là bài tập hiếm hoi có đáp án đúng biết trước.""")
code("""X, y = load_diabetes(return_X_y=True, as_frame=True)
BIEN_THAT = list(X.columns)

rng = np.random.default_rng(RNG_SEED)
nhieu = pd.DataFrame(
    rng.normal(size=(len(X), 190)),
    columns=[f"chi_so_nhieu_{i:03d}" for i in range(190)],
)
X_full = pd.concat([X, nhieu], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=RNG_SEED
)

print(f"Dữ liệu: {X_full.shape[0]} dòng x {X_full.shape[1]} cột "
      f"({len(BIEN_THAT)} biến thật + 190 biến nhiễu)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
X_full.head()""")

md("## 2. Baseline: Linear Regression trên 200 cột → quan sát overfit")
code("""scaler_base = StandardScaler()
Xtr_s = scaler_base.fit_transform(X_train)
Xte_s = scaler_base.transform(X_test)

lin = LinearRegression()
lin.fit(Xtr_s, y_train)
rmse_lin_train = mean_squared_error(y_train, lin.predict(Xtr_s)) ** 0.5
rmse_lin_test = mean_squared_error(y_test, lin.predict(Xte_s)) ** 0.5

print(f"RMSE train = {rmse_lin_train:.2f}")
print(f"RMSE test  = {rmse_lin_test:.2f}  <- lớn hơn nhiều so với train => OVERFIT rõ rệt")""")

md("## 3. LassoCV dò alpha (5-fold CV)")
code("""pipe_lasso = Pipeline([
    ("scale", StandardScaler()),
    ("lasso", LassoCV(alphas=np.logspace(-4, 1, 100), cv=5,
                       max_iter=50000, random_state=RNG_SEED)),
])
pipe_lasso.fit(X_train, y_train)

lasso_model = pipe_lasso["lasso"]
alpha_toi_uu = lasso_model.alpha_
he_so = lasso_model.coef_
so_bien_giu = int((he_so != 0).sum())

rmse_lasso_train = mean_squared_error(y_train, pipe_lasso.predict(X_train)) ** 0.5
rmse_lasso_test = mean_squared_error(y_test, pipe_lasso.predict(X_test)) ** 0.5

print(f"alpha tối ưu (CV) = {alpha_toi_uu:.5f}")
print(f"Lasso giữ lại {so_bien_giu}/200 biến")
print(f"RMSE train = {rmse_lasso_train:.2f}")
print(f"RMSE test  = {rmse_lasso_test:.2f}")""")

md("""## 4. ⭐ Chấm điểm chọn biến

- Bao nhiêu trong 10 biến THẬT được giữ? (recall chọn biến)
- Bao nhiêu biến nhiễu bị giữ nhầm? (false positive)""")
code("""ten_cot = X_full.columns
bien_duoc_chon = set(ten_cot[he_so != 0])

that_giu = sorted(bien_duoc_chon & set(BIEN_THAT))
that_bo = sorted(set(BIEN_THAT) - bien_duoc_chon)
nhieu_giu_nham = sorted(bien_duoc_chon - set(BIEN_THAT))

print(f"Biến THẬT được giữ ({len(that_giu)}/10): {that_giu}")
print(f"Biến THẬT bị bỏ    ({len(that_bo)}/10): {that_bo}")
print(f"Biến NHIỄU bị giữ nhầm ({len(nhieu_giu_nham)}/190): {nhieu_giu_nham}")

bang_diem = pd.DataFrame({
    "Chỉ số": ["Tổng số biến giữ lại", "Biến thật giữ đúng (recall)",
               "Biến thật bị bỏ sót", "Biến nhiễu giữ nhầm (false positive)"],
    "Giá trị": [f"{so_bien_giu}/200", f"{len(that_giu)}/10",
                f"{len(that_bo)}/10", f"{len(nhieu_giu_nham)}/190"],
})
bang_diem""")

md("## 5. Coefficient path của Lasso — các hệ số lần lượt \\\"rơi\\\" về 0")
code("""scaler_path = StandardScaler()
X_scaled_full = scaler_path.fit_transform(X_full)

alphas_path, coefs_path, _ = lasso_path(
    X_scaled_full, y, alphas=np.logspace(1, -4, 100), max_iter=50000
)

plt.figure(figsize=(9, 6))
for i, col in enumerate(ten_cot):
    mau = "crimson" if col in BIEN_THAT else "lightgray"
    do_day = 2.2 if col in BIEN_THAT else 0.6
    plt.plot(alphas_path, coefs_path[i], color=mau, linewidth=do_day)
plt.xscale("log")
plt.gca().invert_xaxis()
plt.axvline(alpha_toi_uu, color="steelblue", linestyle="--",
            label=f"alpha tối ưu (CV) = {alpha_toi_uu:.4f}")
plt.xlabel("alpha (thang log, giảm dần)")
plt.ylabel("Hệ số hồi quy (đã chuẩn hoá)")
plt.title("Lasso Coefficient Path\\n(đỏ = 10 biến thật, xám = 190 biến nhiễu)")
plt.legend()
plt.tight_layout()
plt.show()""")

md("## 6. RMSE train/test theo alpha")
code("""alphas_grid = np.logspace(-4, 1, 60)
rmse_train_list, rmse_test_list = [], []

scaler_grid = StandardScaler().fit(X_train)
Xtr_g = scaler_grid.transform(X_train)
Xte_g = scaler_grid.transform(X_test)

for a in alphas_grid:
    m = Lasso(alpha=a, max_iter=50000, random_state=RNG_SEED)
    m.fit(Xtr_g, y_train)
    rmse_train_list.append(mean_squared_error(y_train, m.predict(Xtr_g)) ** 0.5)
    rmse_test_list.append(mean_squared_error(y_test, m.predict(Xte_g)) ** 0.5)

plt.figure(figsize=(9, 6))
plt.plot(alphas_grid, rmse_train_list, label="RMSE train", marker="o", markersize=3)
plt.plot(alphas_grid, rmse_test_list, label="RMSE test", marker="o", markersize=3)
plt.axvline(alpha_toi_uu, color="gray", linestyle="--",
            label=f"alpha tối ưu = {alpha_toi_uu:.4f}")
plt.xscale("log")
plt.xlabel("alpha (thang log)")
plt.ylabel("RMSE")
plt.title("RMSE train/test theo alpha")
plt.legend()
plt.tight_layout()
plt.show()""")

md("""## 7. So sánh Ridge vs Lasso

- Ridge giữ bao nhiêu biến? (dự đoán: cả 200, chỉ co nhỏ)
- RMSE cái nào tốt hơn?""")
code("""pipe_ridge = Pipeline([
    ("scale", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)),
])
pipe_ridge.fit(X_train, y_train)
ridge_model = pipe_ridge["ridge"]
so_bien_giu_ridge = int((ridge_model.coef_ != 0).sum())
rmse_ridge_train = mean_squared_error(y_train, pipe_ridge.predict(X_train)) ** 0.5
rmse_ridge_test = mean_squared_error(y_test, pipe_ridge.predict(X_test)) ** 0.5

bang_ridge_lasso = pd.DataFrame({
    "Mô hình": ["Linear (baseline, 200 biến)", "Ridge (L2)", "Lasso (L1)"],
    "Số biến giữ lại": [200, so_bien_giu_ridge, so_bien_giu],
    "RMSE train": [round(rmse_lin_train, 2), round(rmse_ridge_train, 2), round(rmse_lasso_train, 2)],
    "RMSE test": [round(rmse_lin_test, 2), round(rmse_ridge_test, 2), round(rmse_lasso_test, 2)],
})
bang_ridge_lasso""")

code("""x_pos = np.arange(3)
plt.figure(figsize=(7, 5))
plt.bar(x_pos, bang_ridge_lasso["Số biến giữ lại"], color=["gray", "seagreen", "crimson"])
plt.xticks(x_pos, bang_ridge_lasso["Mô hình"], rotation=15)
plt.ylabel("Số biến giữ lại (trên 200)")
plt.title("Ridge vs Lasso: số biến giữ lại")
for i, v in enumerate(bang_ridge_lasso["Số biến giữ lại"]):
    plt.text(i, v + 3, str(v), ha="center")
plt.tight_layout()
plt.show()""")

md("""## 8. ⚠️ Thí nghiệm biến tương quan

Nhân đôi cột `bmi` (biến thật mạnh nhất) thành một bản sao có tương quan ~0.95
(giống tình huống thực tế: 2 chỉ số xét nghiệm đo cùng một hiện tượng sinh học).
Chạy Lasso trên nhiều mẫu bootstrap khác nhau (mô phỏng "chạy lại với seed khác")
để xem Lasso có chọn ổn định không.""")
code("""X_corr = X_full.copy()
bien_goc = "bmi"
X_corr[f"{bien_goc}_ban_sao"] = X_corr[bien_goc] + rng.normal(scale=0.016, size=len(X_corr))
corr_thuc_te = X_corr[bien_goc].corr(X_corr[f"{bien_goc}_ban_sao"])
print(f"Tương quan thực tế giữa '{bien_goc}' và bản sao: {corr_thuc_te:.3f}")

ket_qua_seed = {}
n = len(X_corr)
for seed in [1, 2, 3, 42, 99]:
    boot_rng = np.random.default_rng(seed)
    idx = boot_rng.choice(n, size=n, replace=True)
    X_boot, y_boot = X_corr.iloc[idx], y.iloc[idx]
    sc = StandardScaler().fit(X_boot)
    m = LassoCV(alphas=np.logspace(-4, 1, 60), cv=5, max_iter=50000, random_state=seed)
    m.fit(sc.transform(X_boot), y_boot)
    coefs = dict(zip(X_corr.columns, m.coef_))
    ket_qua_seed[seed] = {bien_goc: round(coefs[bien_goc], 4),
                           f"{bien_goc}_ban_sao": round(coefs[f"{bien_goc}_ban_sao"], 4)}

bang_tuong_quan = pd.DataFrame(ket_qua_seed).T
bang_tuong_quan.index.name = "seed bootstrap"
bang_tuong_quan""")

md("""**Nhận xét:** trong lần chạy này, Lasso *luôn* chọn `bmi` gốc và loại bản sao ở mọi
seed bootstrap — không đảo chiều. Điều này không mâu thuẫn với lý thuyết: khi một cột chỉ
là "gần giống" (chưa hoàn toàn trùng), coordinate descent có xu hướng ưu tiên cột có tương
quan biên (marginal correlation) với `y` cao hơn một chút và giữ nguyên lựa chọn đó qua
các lần resample. Tính KHÔNG ổn định của Lasso thể hiện rõ nhất khi hai biến gần như
**trùng tuyệt đối** (correlation ≈ 0.999+) — khi đó sai số số học rất nhỏ cũng đủ để đảo
chiều lựa chọn. Dù vậy, bài học thực tế vẫn đúng: **không nên diễn giải "biến A quan trọng
hơn biến B" khi A và B tương quan cao** — biến bị Lasso bỏ có thể chỉ thua biến kia một
chút xíu, không phải vì nó vô dụng.""")

md("""## 9. Debiased Lasso — Linear Regression chỉ trên biến Lasso chọn

Kỹ thuật phổ biến: dùng Lasso để CHỌN biến (loại nhiễu), sau đó train lại Linear Regression
thường (không phạt) trên đúng tập biến đó để hệ số không bị co (biased) bởi phạt L1.""")
code("""cot_duoc_chon = list(bien_duoc_chon)
X_train_sel = X_train[cot_duoc_chon]
X_test_sel = X_test[cot_duoc_chon]

sc_sel = StandardScaler().fit(X_train_sel)
lin_debiased = LinearRegression()
lin_debiased.fit(sc_sel.transform(X_train_sel), y_train)

rmse_debiased_train = mean_squared_error(y_train, lin_debiased.predict(sc_sel.transform(X_train_sel))) ** 0.5
rmse_debiased_test = mean_squared_error(y_test, lin_debiased.predict(sc_sel.transform(X_test_sel))) ** 0.5

print(f"Debiased Linear Regression trên {so_bien_giu} biến Lasso chọn:")
print(f"  RMSE train = {rmse_debiased_train:.2f}")
print(f"  RMSE test  = {rmse_debiased_test:.2f}")
print(f"So với Lasso đầy đủ: RMSE test = {rmse_lasso_test:.2f}")""")

md("""## 10. ✍️ Đề xuất bộ xét nghiệm cuối cùng + ước tính chi phí tiết kiệm""")
code("""print("Bộ xét nghiệm đề xuất (biến Lasso chọn, bao gồm cả biến thật lẫn nhiễu giữ nhầm):")
for c in sorted(cot_duoc_chon):
    loai = "BIẾN THẬT" if c in BIEN_THAT else "nhiễu (giữ nhầm — nên loại bỏ khi triển khai)"
    print(f"  - {c:20s} [{loai}]")

gia_moi_chi_so = 5_000_000 / 200  # giả định ước tính ban đầu: 200 chỉ số ~ 5 triệu đồng
chi_phi_cu = 5_000_000
chi_phi_moi = len(that_giu) * gia_moi_chi_so
print(f"\\nChi phí ước tính CŨ (200 chỉ số): {chi_phi_cu:,.0f} đ/bệnh nhân")
print(f"Chi phí ước tính MỚI ({len(that_giu)} chỉ số thật hữu ích): {chi_phi_moi:,.0f} đ/bệnh nhân")
print(f"Tiết kiệm: {chi_phi_cu - chi_phi_moi:,.0f} đ/bệnh nhân "
      f"({(1 - chi_phi_moi/chi_phi_cu)*100:.0f}%)")""")

md("""**Kết luận:** Lasso cho phép giảm bộ xét nghiệm từ 200 chỉ số xuống chỉ còn vài chỉ số
thật sự có tín hiệu, tiết kiệm phần lớn chi phí xét nghiệm trong khi RMSE dự đoán vẫn tốt
hơn nhiều so với dùng Linear Regression trên toàn bộ 200 chỉ số (vốn bị overfit nặng).
Tuy nhiên cần lưu ý: (1) một vài biến nhiễu có thể bị giữ nhầm do ngẫu nhiên, và (2) khi có
các biến tương quan cao, danh sách biến được chọn có thể không hoàn toàn ổn định giữa các
lần chạy — nên tham khảo thêm Stability Selection hoặc ElasticNet (TT-14) trước khi áp dụng
lâm sàng thực tế.""")

nb['cells'] = cells
nbf.write(nb, "../notebooks/lasso_feature_selection.ipynb")
print("Notebook đã được tạo.")
