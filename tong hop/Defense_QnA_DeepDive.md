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

### Q6: Trong `recommendation.py`, tại sao công thức tính Score hiện tại dùng `Average_Safe_Ratio * Thermal_Modifier`, còn indoor/shade lại xử lý bằng Effective UV?
*   **Cách trả lời:**
    "Phiên bản hiện tại không còn cộng hoặc nhân bonus cố định cho indoor/shade. Em chuyển các yếu tố này về cùng đại lượng vật lý là UV hiệu dụng. Nếu địa điểm trong nhà, hệ thống dùng `effective_uv = uv * 0.05`; nếu là ngoài trời có bóng râm, hệ thống dùng `uv * ((1 - shade_ratio) + shade_ratio * 0.3)`. Sau đó hệ thống tính lại estimated exposure minutes bằng công thức MED/UVI. Em trình bày 0.05 và 0.3 là xấp xỉ triển khai có cơ sở từ tài liệu UV attenuation, không phải hằng số tuyệt đối cho mọi loại kính hoặc bóng râm. Score cuối là `Average_Safe_Ratio * Thermal_Modifier`; nhiệt độ cao ngoài trời giảm điểm bằng hệ số 0.5 như một comfort penalty triển khai."

### Q7: Chỉ số NDCG@5 trong `evaluation.py` hoạt động thế nào? Tại sao nó ưu việt hơn Precision@5?
*   **Cách trả lời:**
    "Chỉ số `Precision@5` chỉ quan tâm việc Địa điểm Tốt có nằm trong Top 5 hay không (trả lời Có hoặc Không). Nhưng với người dùng, vị trí đứng của địa điểm là cực kỳ quan trọng. 
    `NDCG@5` (Normalized Discounted Cumulative Gain) hữu ích vì nó quan tâm đến **thứ tự xếp hạng**. NDCG áp dụng hàm chia logarit: nó giảm điểm nếu kết quả phù hợp bị đẩy xuống dưới. Sau khi sửa `IDCG` theo candidate-level ideal ranking, NDCG@5 hiện là 0.9493 và không vượt 1. Tuy nhiên, NDCG chỉ chứng minh chất lượng ranking theo ground truth scenario, không chứng minh an toàn y khoa ngoài thực tế."

### Q8: Hãy giải thích logic MED/Fitzpatrick và công thức estimated exposure limit?
*   **Cách trả lời:**
    "Thang Fitzpatrick chia da thành 6 nhóm theo phản ứng cháy nắng/rám nắng. Công thức của hệ thống là `Estimated_Exposure_Minutes = MED_Value / (Effective_UV * 1.5)`. Trong đó 1 UVI tương đương 1.5 J/(m²·phút) bức xạ gây đỏ da, còn `MED_Value` là giá trị đại diện theo nhóm da: loại I = 200 J/m², loại III = 300 J/m², loại VI = 1000 J/m². Ví dụ da loại 3 tại UV=10 có estimated exposure limit xấp xỉ `300 / (10 * 1.5) = 20 phút`. Em không trình bày đây là bảo đảm y khoa, vì MED thay đổi theo từng cá nhân và WHO cảnh báo không nên hiểu burn time là quyền phơi nắng không bảo vệ."

### Q9: Làm sao chứng minh hệ thống của em tốt hơn việc chỉ đơn giản 'Gợi ý các địa điểm du lịch gần nhất'?
*   **Cách trả lời:**
    "Trong tab System Assessment, em đã triển khai Hệ thống Đánh giá Baseline (Baseline Comparator). Em đã đo lường và hiển thị minh bạch sự đối đầu giữa hệ thống của em với 3 phương pháp khác: Lựa chọn Ngẫu nhiên (Random), Gợi ý Nơi nổi tiếng nhất (Popularity), và Gợi ý Nơi gần nhất (Distance Only).
    Biểu đồ cho thấy trong 33 scenario offline, hệ thống UV-aware đạt Precision@5 = 83.03% và NDCG@5 = 0.9493, cao hơn distance-only ở Precision@5 = 55.15%. Em chỉ diễn giải đây là bằng chứng ranking theo scenario, không phải bằng chứng y khoa ngoài thực tế."
