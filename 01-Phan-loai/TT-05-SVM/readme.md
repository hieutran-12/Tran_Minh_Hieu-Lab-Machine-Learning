1. Nạp dữ liệu thành công và xác nhận 0 = ác tính và có 212 ác tính và 357 lành tính

2. Giải thích vì sao SVM chỉ phụ thuộc vào support vectors
   SVM không cần tất cả điểm dữ liệu để xác định boundary vì nó chủ yếu dựa vào những điểm gần decision boundary nhất, Các điểm nằm rất xa boundary thường không ảnh hưởng đến vị trí boundary.
3. Hạn chế khi sử dụng SVM
   SVM khó giải thích trực tiếp ảnh hưởng của từng feature hoặc lý do cụ thể khiến một mẫu được phân loại thành malignant/benign. Ngoài ra, chi phí tính toán và bộ nhớ của SVM có thể tăng đáng kể khi số lượng mẫu lớn, đặc biệt với kernel phi tuyến như RBF.
