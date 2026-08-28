## Kết quả chính

| Bước                                                      | F1-score  |
| --------------------------------------------------------- | --------- |
| Dummy (most_frequent)                                     | 0.000     |
| 1 stump (depth=1)                                         | 0.915     |
| **AdaBoost 300 stumps**                                   | **0.984** |
| GradientBoosting (150 cây, depth=3)                       | 0.998     |
| RandomForest (300 cây)                                    | 0.999     |
| AdaBoost trên **test gốc NSL-KDD** (có tấn công zero-day) | **0.759** |

→ Đúng như mức tham chiếu của đề (~0.95+ trên CV, ~0.75–0.80 trên test gốc).

## THÍ NGHIỆM NHIỄU NHÃN (đảo ngẫu nhiên 5% nhãn train)

| Model         | F1 sạch | F1 nhiễu 5% | Tụt (%)   |
| ------------- | ------- | ----------- | --------- |
| AdaBoost      | 0.9843  | 0.9730      | **1.15%** |
| Random Forest | 0.9990  | 0.9929      | 0.62%     |

**Giải thích:** AdaBoost tăng trọng số các mẫu bị phân loại sai qua từng vòng lặp. Một mẫu bị **gán nhãn sai** sẽ liên tục "sai" dưới góc nhìn của các stump kế tiếp, khiến trọng số của nó tăng gần như không kiểm soát — model dồn nhiều "vòng" để cố gắng khớp đúng một điểm rác, làm giảm hiệu năng tổng thể. Random Forest dùng bagging (mỗi cây chỉ thấy một mẫu con dữ liệu, độc lập với các cây khác) nên một vài nhãn nhiễu chỉ ảnh hưởng cục bộ tới một phần các cây, rồi được trung bình hoá — vì vậy ổn định hơn trước nhiễu nhãn. Kết quả thực nghiệm khớp với lý thuyết: AdaBoost tụt điểm theo tỷ lệ % gần gấp đôi Random Forest.

## Chênh lệch CV vs test gốc

CV (cùng phân bố train): F1 = 0.984. Test gốc NSL-KDD: F1 = 0.759 (chênh **0.225**). Tập test chứa các loại tấn công **không xuất hiện trong train** (mô phỏng zero-day) — model chưa từng thấy các pattern này nên recall giảm mạnh. Đây là hành vi được thiết kế có chủ đích của bộ dữ liệu, không phải lỗi.

## Alert fatigue

FPR = 7.29% trên test gốc (TP=8282, FP=708, FN=4551, TN=9003). Với giả định ~1.44 triệu kết nối bình thường/ngày → **~105.000 báo động giả/ngày** ở ngưỡng mặc định (0.5) — quá cao để SOC vận hành thực tế; cần hiệu chỉnh ngưỡng quyết định hoặc thêm lớp lọc quy tắc trước khi đẩy cảnh báo cho nhân viên.

Dữ liệu: NSL-KDD (`KDDTrain+.txt`, `KDDTest+.txt`) — **không có header**, cần nạp bằng `header=None` + tự đặt tên 43 cột (41 đặc trưng + label + difficulty). \*Lưu ý: bản CSV gốc bạn đã thử nạp trước đó bị thiếu `header=None`, khiến pandas lấy dòng dữ liệu đầu tiên làm tên cột (`"13", "tcp", "telnet", "SF", ...` chính là các giá trị của dòng 1, không phải tên cột thật) — đây là nguyên nhân info() ra 43 cột với tên số/chuỗi vô nghĩa trong file bạn gửi.
