# TÀI LIỆU ÔN TẬP BẢO VỆ ĐỒ ÁN: CHUYÊN SÂU & PHẢN BIỆN (DEFENSE Q&A)

Đây là tài liệu được biên soạn đặc biệt để giúp bạn "phòng thủ" và trả lời xuất sắc những câu hỏi phản biện cực khó, mang tính đào sâu từ các giảng viên trong Hội đồng bảo vệ đồ án.

---

## PHẦN 1: TIỀN XỬ LÝ & KỸ NGHỆ DỮ LIỆU (DATA ENGINEERING)

### Q1: Tại sao trong `merger.py`, em lại dùng thuật toán `ffill(limit=3)` để điền dữ liệu khuyết thay vì dùng giá trị Trung bình (Mean) hoặc Nội suy (Interpolation)?
*   **Cách trả lời:** 
    "Thưa thầy/cô, thời tiết và bức xạ có tính liên tục cục bộ (local continuity). Nhiệt độ hay lượng mây ở giờ T thường rất giống với giờ T-1. Việc dùng `Forward-fill` (lấy giá trị quá khứ gần nhất đắp vào) giữ lại được tính chân thực của khoảnh khắc đó. 
    Lý do em dùng tham số `limit=3` (tối đa 3 tiếng) là vì đây là ngưỡng an toàn. Nếu khuyết quá 3 tiếng, bầu trời đã chuyển sang một buổi khác (ví dụ từ sáng sang trưa), việc copy dữ liệu cũ sẽ làm sai lệch mô hình. Nếu dùng Mean của cả tháng thì sẽ phá vỡ hoàn toàn tính chất cục bộ của thời tiết ngày hôm đó."

### Q2: Tại sao phải dùng sin/cos để mã hóa thời gian (`hour_sin`, `hour_cos`)? Việc để nguyên số giờ (0, 1, 2... 23) có vấn đề gì đối với Machine Learning?
*   **Cách trả lời:**
    "Thuật toán Machine Learning hiểu dữ liệu theo khoảng cách toán học. Nếu ta để nguyên số giờ (0-23), khoảng cách giữa 23h đêm và 0h sáng là 23 đơn vị (cực kỳ xa nhau về mặt số học). Nhưng trên thực tế, 23h và 0h chỉ cách nhau 1 tiếng (rất gần).
    Việc em chuyển sang trục tọa độ vòng tròn lượng giác (sin/cos) giúp khoảng cách giữa 23h và 0h được thu hẹp lại liên tục, đúng với bản chất tuần hoàn tự nhiên của thời gian. Mô hình sẽ học được sự mượt mà này thay vì bị gián đoạn ở chu kỳ ngày mới."

### Q3: Đặc trưng `solar_cloud_interaction` được tính bằng phép nhân: `cos(zenith) * (1 - cloud_cover)`. Tại sao lại dùng phép nhân mà không dùng cộng hay để hai biến đó hoạt động độc lập?
*   **Cách trả lời:**
    "Phép nhân ở đây nhằm mô phỏng hiệu ứng 'kìm hãm' (Interaction effect) trong Vật lý học. Góc thiên đỉnh (zenith) cao tạo ra tiềm năng bức xạ rất lớn, nhưng lượng bức xạ này bắt buộc phải xuyên qua mây. 
    Nếu bầu trời nhiều mây 100% (`cloud_cover` = 1), thì `(1 - 1) = 0`, phép nhân trả về 0. Lúc này, dù mặt trời có đứng trên đỉnh đầu đi chăng nữa, giá trị đặc trưng vẫn lập tức bị kéo về 0. Thay vì ép mô hình học máy tự tìm ra sự thật vật lý này, em đã chủ động cung cấp sẵn cho nó, giúp giảm sai số đáng kể."

### Q4: Hệ thống có xảy ra Data Leakage (Rò rỉ dữ liệu tương lai) khi em tính các biến độ trễ (Lag / Rolling) không? Em xử lý rủi ro đó như thế nào?
*   **Cách trả lời:**
    "Em hoàn toàn ý thức được rủi ro Data Leakage ở bài toán Time-Series. Trong file `engr.py`, khi tính toán hàm `shift()` (lag) hay `rolling()`, em đã bắt buộc gọi `.groupby('location_id')`. 
    Nếu không group, dữ liệu cuối ngày của trạm A sẽ bị lấy làm 'quá khứ' cho dữ liệu đầu ngày của trạm B – đây là lỗi leakage kinh điển. Hơn nữa, với các mốc giờ đầu tiên không có quá khứ, chúng sẽ trả về `NaN` và được xử lý thay vì lấy số liệu giả mạo từ tương lai."

---

## PHẦN 2: MÔ HÌNH HỌC MÁY & ĐÁNH GIÁ (MACHINE LEARNING)

### Q5: Tại sao em lại chọn hệ quy chiếu Metric là RMSE để đánh giá độ tốt của mô hình UV? Tại sao không dùng MAE cho dễ hiểu?
*   **Cách trả lời:**
    "Bài toán dự báo UV liên quan trực tiếp đến cảnh báo sức khỏe y tế. Nếu mô hình đoán sai ở mức thấp (đoán UV 2 lên 3) thì không gây hậu quả gì. Nhưng nếu đoán sai đỉnh UV (đoán UV 7 lên 11) thì có thể khiến người dùng bị cháy nắng nặng.
    Chỉ số RMSE sử dụng hàm bình phương sai số, do đó nó 'phạt' (penalize) cực kỳ nặng các dự đoán trượt xa khỏi đỉnh UV thực tế. Chọn mô hình có RMSE thấp nhất đảm bảo hệ thống có độ an toàn và ổn định cao nhất khi dự báo đỉnh."

---

## PHẦN 3: HỆ THỐNG GỢI Ý (RECOMMENDER SYSTEM) & KIỂM THỬ

### Q6: Trong `recommendation.py`, tại sao công thức tính Score lại dùng phép Nhân: `Average_Safe_Ratio * Shade_Bonus * Indoor_Bonus` thay vì phép Cộng?
*   **Cách trả lời:**
    "Em sử dụng phép Nhân vì nó đóng vai trò như toán tử logic **AND**. 
    Nếu `Average_Safe_Ratio` = 0 (nghĩa là khung giờ đó ra ngoài 100% bị cháy nắng, không an toàn chút nào), thì dù địa điểm đó có đẹp hay bóng râm (`Shade_Bonus`) nhiều cỡ nào, tổng điểm Score sẽ lập tức bị kéo về 0. 
    Nếu em dùng phép Cộng, một địa điểm nguy hiểm chết người nhưng có hệ số `Indoor_Bonus` cao vẫn có thể lọt lên Top đầu gợi ý. Phép nhân chính là chốt chặn an toàn (Veto Power)."

### Q7: Chỉ số NDCG@5 trong `evaluation.py` hoạt động thế nào? Tại sao nó ưu việt hơn Precision@5?
*   **Cách trả lời:**
    "Chỉ số `Precision@5` chỉ quan tâm việc Địa điểm Tốt có nằm trong Top 5 hay không (trả lời Có hoặc Không). Nhưng với người dùng, vị trí đứng của địa điểm là cực kỳ quan trọng. 
    `NDCG@5` (Normalized Discounted Cumulative Gain) ưu việt hơn vì nó quan tâm đến **Thứ tự xếp hạng**. NDCG áp dụng hàm chia logarit: Nó sẽ 'trừ điểm' nặng nề nếu hệ thống đẩy một địa điểm hoàn hảo xuống tận Top 4 thay vì đặt nó ở Top 1. Hệ thống của em đạt NDCG cao, chứng minh nó không chỉ tìm đúng, mà còn biết cách xếp cái tốt nhất lên đầu tiên."

### Q8: Hãy giải thích logic y khoa `ScanSkinAI` và công thức Thời gian An toàn?
*   **Cách trả lời:**
    "Thang đo Y khoa Fitzpatrick chia da người thành 6 loại. Da sáng màu (Loại 1, 2) có rất ít sắc tố Melanin bảo vệ, do đó Base Time (thời gian phơi nắng tiêu chuẩn trước khi da bị tổn thương) rất ngắn, chỉ khoảng 15-20 phút. Da sậm màu (Loại 5, 6) Base Time có thể lên đến 90 phút.
    Công thức của hệ thống là: `Safe_Minutes = Base_Time_SkinType / max(1, UV_Index)`
    Ví dụ: Da loại 3 (Base Time = 30), UV lúc 12h trưa đang là 10. Vậy thời gian an toàn = 30 / 10 = 3 phút. Nghĩa là người dùng chỉ có đúng 3 phút trước khi tế bào da bắt đầu bị bỏng rát. Hệ thống sẽ ngay lập tức cảnh báo người dùng và gợi ý họ tìm địa điểm trong nhà (Indoor)."

### Q9: Làm sao chứng minh hệ thống của em tốt hơn việc chỉ đơn giản 'Gợi ý các địa điểm du lịch gần nhất'?
*   **Cách trả lời:**
    "Trong tab System Assessment, em đã triển khai Hệ thống Đánh giá Baseline (Baseline Comparator). Em đã đo lường và hiển thị minh bạch sự đối đầu giữa hệ thống của em với 3 phương pháp khác: Lựa chọn Ngẫu nhiên (Random), Gợi ý Nơi nổi tiếng nhất (Popularity), và Gợi ý Nơi gần nhất (Distance Only).
    Biểu đồ chứng minh rõ ràng: Khi kết hợp nhận thức UV (UV-aware), hệ thống của em đạt Precision và NDCG cao hơn hẳn việc chỉ chọn theo khoảng cách, vì những nơi gần nhất đôi khi lại là những bãi biển trống trải đang chịu ngưỡng tia cực tím hủy diệt vào lúc 12h trưa."
