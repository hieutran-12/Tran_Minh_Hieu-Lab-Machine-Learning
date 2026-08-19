Quy tắc 1 — Không làm thêm giờ + ít kinh nghiệm
OverTime = No
AND TotalWorkingYears <= 2.5

→ Cây dự đoán Yes (nghỉ việc).

Diễn giải:

Nhân viên không làm thêm giờ nhưng có ≤ 2.5 năm tổng kinh nghiệm làm việc được cây dự đoán có nguy cơ nghỉ việc.

Quy tắc 2 — Không làm thêm giờ + nhiều kinh nghiệm + không hài lòng
OverTime = No
AND TotalWorkingYears > 2.5
AND StockOptionLevel <= 0.5
AND JobSatisfaction <= 1.5

→ Yes

Diễn giải:

Nhân viên không làm thêm giờ, có trên 2.5 năm kinh nghiệm, mức Stock Option thấp và Job Satisfaction ≤ 1 có nguy cơ nghỉ việc cao hơn theo cây.

⚠️ Ở đây JobSatisfaction <= 1.5 vì biến có giá trị nguyên 1–4 nên thực tế nghĩa là:

JobSatisfaction = 1
Quy tắc 3 — Làm thêm giờ + cấp bậc thấp
OverTime = Yes
AND JobLevel <= 1.5

→ Yes

Vì JobLevel cũng là biến nguyên:

JobLevel <= 1.5

thực tế nghĩa là:

JobLevel = 1

Diễn giải:

Nhân viên làm thêm giờ và đang ở Job Level 1 được cây dự đoán có nguy cơ nghỉ việc.

Đây là một quy tắc rất đáng chú ý vì nó khá dễ giải thích cho HR.

Quy tắc 4 — Làm thêm giờ + cấp bậc cao + ở xa
OverTime = Yes
AND JobLevel > 1.5
AND MaritalStatus_Single <= 0.5
AND DistanceFromHome > 17.5

→ Yes

MaritalStatus_Single <= 0.5 nghĩa là:

MaritalStatus != Single

Diễn giải:

Nhân viên làm thêm giờ, có Job Level từ 2 trở lên, không thuộc nhóm độc thân và sống cách nơi làm việc trên khoảng 17 km được cây dự đoán có nguy cơ nghỉ việc.

Quy tắc 5 — Làm thêm giờ + cấp bậc cao + độc thân
OverTime = Yes
AND JobLevel > 1.5
AND MaritalStatus_Single > 0.5

→ Yes

MaritalStatus_Single > 0.5 nghĩa là:

MaritalStatus = Single

Diễn giải:

Nhân viên làm thêm giờ, có Job Level từ 2 trở lên và độc thân được cây dự đoán có nguy cơ nghỉ việc.
