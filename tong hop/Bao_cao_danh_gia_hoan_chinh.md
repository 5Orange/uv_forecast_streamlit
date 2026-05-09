# BÁO CÁO ĐÁNH GIÁ KỸ THUẬT VÀ HỌC THUẬT: HỆ THỐNG DỰ BÁO UV, CẢNH BÁO VÀ GỢI Ý DU LỊCH

## 1. TỔNG QUAN DỰ ÁN (OVERALL PROJECT REVIEW)

### 1.1. Kiến trúc hệ thống
*   **WHAT:** Hệ thống được thiết kế chia làm hai phần rõ rệt: Backend xử lý dữ liệu/mô hình (`UV_analysis`) và Frontend phục vụ tương tác người dùng (`UV_analysis_UI` sử dụng Streamlit).
*   **WHY:** Thiết kế Decoupled (tách rời) này giúp dễ dàng mở rộng (scalability). Người phát triển có thể thay đổi, huấn luyện lại mô hình ở Backend mà không làm sập giao diện UI.
*   **HOW:** Dữ liệu thô từ nhiều nguồn được xử lý qua Data Pipeline (`merger.py`, `engr.py`). Sau đó, 11 mô hình hồi quy (Regression Models) được huấn luyện để dự báo cường độ UV. Frontend truy xuất các kết quả dự báo này (qua batch hoặc API serving) và kết nối với module địa điểm (Tourist Catalog) để đưa ra các phân tích (EDA, Results) và gợi ý (Forecast, Recommendation).

### 1.2. Đánh giá Ưu nhược điểm & Tính ứng dụng
*   **Ưu điểm:** Tích hợp đa nguồn dữ liệu thời gian thực và lịch sử (Open-Meteo, OpenUV, WeatherBit). Áp dụng tính toán toán học chuẩn mực cho cả vị trí địa lý (pvlib) và sinh học (ScanSkinAI, WHO).
*   **Nhược điểm:** Mức độ phụ thuộc vào API bên thứ 3 cao; nếu một trong các API thay đổi cấu trúc, pipeline `merger.py` có nguy cơ bị lỗi.
*   **Ứng dụng thực tế:** Đặc biệt hữu ích ở các nước nhiệt đới (như Việt Nam). Có giá trị lớn đối với du lịch bền vững và y tế cộng đồng (phòng chống ung thư da).

---

## 2. PHÂN TÍCH BACKEND (UV_analysis)

### 2.1. Feature Engineering (Kỹ nghệ Đặc trưng)

#### A. Pipeline Hợp nhất Dữ liệu (`merger.py`)
*   **WHAT:** Đây là luồng tích hợp dữ liệu thô (ETL Pipeline) từ Open-Meteo (forecast/historical), WeatherBit (real-time), và OpenUV.
*   **WHY:** Khắc phục nhược điểm của dữ liệu đơn lẻ (ví dụ Open-Meteo mạnh về khí tượng nhưng chỉ số UV đôi khi nội suy thiếu chính xác so với đo lường thực tế của OpenUV).
*   **HOW:**
    *   **Data merging logic:** Mã nguồn sử dụng pandas `merge()` dựa trên các khóa `timestamp` và `location_id`.
    *   **Xử lý Missing Data:** Áp dụng forward-fill (`ffill(limit=3)`) cho các biến liên tục như nhiệt độ, độ ẩm để lấp đầy các khoảng thời gian bị thiếu sót (gap) ngắn.
    *   **Solar Position (pvlib):** Thuật toán gọi `pvlib.solarposition` để tính toán góc phương vị (`azimuth`), góc cao (`elevation`), và góc thiên đỉnh (`zenith`). `cos(zenith)` tỷ lệ thuận với lượng bức xạ mặt trời xuyên qua khí quyển.

#### B. 23 Đặc trưng Cốt lõi (23 Core Features) trong `engr.py`
Để mô hình học máy đạt độ chính xác cao nhất, hệ thống đã kỹ sư hóa 23 đặc trưng (features) cốt lõi, được phân loại thành 4 nhóm chính. Việc lựa chọn các đặc trưng này dựa chặt chẽ trên tính chất vật lý khí quyển và chuỗi thời gian:

**Nhóm 1: Đặc trưng Thời gian (Temporal Features)**
*Tại sao sử dụng:* Thời gian tuyến tính (1, 2, 3... 23, 24) làm mất đi tính tuần hoàn tự nhiên (ví dụ 23h đêm rất gần với 1h sáng). Việc biến đổi thời gian giúp mô hình học được chu kỳ ngày/đêm và chu kỳ các mùa trong năm một cách trơn tru.
*   **1. `hour_sin`**: Khía cạnh hình sin của giờ trong ngày.
*   **2. `hour_cos`**: Khía cạnh hình cos của giờ trong ngày.
*   **3. `month_sin`**: Khía cạnh hình sin của tháng trong năm.
*   **4. `month_cos`**: Khía cạnh hình cos của tháng trong năm.
*   **5. `doy_sin`**: Khía cạnh hình sin của ngày trong năm (day of year).
*   **6. `doy_cos`**: Khía cạnh hình cos của ngày trong năm.
*   **7. `day_fraction`**: Tỷ lệ phần thời gian của ban ngày đã trôi qua. Đặc trưng này chuẩn hóa được vị trí thời điểm hiện tại so với lúc mặt trời mọc và lặn (điều mà `hour` đơn thuần không làm được vì bình minh thay đổi theo mùa).
*   **8. `hours_since_sunrise`**: Số giờ tính từ lúc mặt trời mọc.

**Nhóm 2: Bức xạ & Vị trí Mặt trời (Solar & Radiation Features)**
*Tại sao sử dụng:* Tia UV tới mặt đất phụ thuộc tuyệt đối vào việc mặt trời đang đứng ở góc nào so với mặt đất và khoảng cách tia sáng phải xuyên qua lớp khí quyển.
*   **9. `cos_solar_zenith`**: Cosin của góc thiên đỉnh. Đây là **đặc trưng quan trọng nhất**, tỷ lệ thuận trực tiếp với cường độ bức xạ.
*   **10. `solar_elevation`**: Góc cao của mặt trời so với chân trời.
*   **11. `air_mass`**: Khối lượng không khí mà tia sáng phải đi qua, tính bằng công thức: (1 / cos_zenith). Lớp không khí càng dày (lúc bình minh/hoàng hôn), UV càng bị cản lại nhiều.
*   **12. `clearness_index`**: Chỉ số độ trong suốt của bầu trời (tỷ lệ bức xạ thực tế so với bức xạ lý thuyết ngoài không gian).
*   **13. `solar_cloud_interaction`**: Sự tương tác trực tiếp: `cos_solar_zenith * (1 - cloud_cover)`. Đặc trưng này kết hợp mô phỏng vật lý sự che khuất của mây vào vị trí mặt trời.

**Nhóm 3: Khí quyển & Môi trường (Atmospheric Features)**
*Tại sao sử dụng:* Mây, tầng ozone và khói bụi là những "tấm khiên" tự nhiên hấp thụ hoặc tán xạ tia tử ngoại trước khi chúng chạm đất.
*   **14. `temperature_2m`**: Nhiệt độ không khí.
*   **15. `relative_humidity_2m`**: Độ ẩm tương đối.
*   **16. `cloud_opacity`**: Độ đục của mây. Được tính bằng trung bình có trọng số của mây tầng thấp, trung và cao. Mây tầng thấp thường đặc và cản UV mạnh hơn.
*   **17. `ozone_anomaly`**: Độ lệch của hàm lượng Ozone hiện tại so với trung bình tháng. Ozone là yếu tố chủ chốt hấp thụ tia UV-B độc hại.
*   **18. `aerosol_uv_attenuation`**: Độ suy giảm UV do bụi mịn và các hạt lơ lửng (Aerosol Optical Depth). Rất hữu ích tại các khu vực đô thị ô nhiễm.

**Nhóm 4: Đặc trưng Chuỗi thời gian (Lag, Rolling & Trend Features)**
*Tại sao sử dụng:* UV có tính tự tương quan (autocorrelation) cao. Bức xạ của giờ hiện tại phụ thuộc rất lớn vào xu hướng của các giờ trước đó.
*   **19. `uv_lag_1h`**: Chỉ số UV của 1 giờ trước. Giúp mô hình nắm bắt "quán tính" của hệ thống thời tiết.
*   **20. `uv_lag_24h`**: Chỉ số UV của cùng giờ ngày hôm qua. Hữu ích cho việc dự báo các hình thái thời tiết ổn định kéo dài qua nhiều ngày.
*   **21. `uv_rolling_mean_3h`**: Trung bình UV trong 3 giờ gần nhất. Tính năng này giúp làm mượt các nhiễu động ngẫu nhiên.
*   **22. `uv_diff_1h`**: Đạo hàm bậc 1 của UV so với giờ trước (tốc độ thay đổi). Chỉ báo mô hình biết UV đang trong pha tăng tốc hay lao dốc.
*   **23. `cloud_cover_change_1h`**: Sự biến thiên của độ che phủ mây. Nếu đám mây đột ngột kéo đến che lấp bầu trời, UV sẽ rớt mạnh dù mặt trời đang ở trên đỉnh đầu.

### 2.2. Huấn luyện Mô hình
*   **WHAT:** Benchmark 11 mô hình hồi quy khác nhau (Linear, Tree-based, Ensemble như XGBoost, LightGBM...).
*   **WHY:** Do bản chất dữ liệu UV có yếu tố phi tuyến tính mạnh (phụ thuộc vào tương tác mây - ozone), các mô hình Linear thường hoạt động kém, việc so sánh nhiều mô hình giúp tìm ra thuật toán tối ưu.
*   **HOW:** Quá trình huấn luyện đánh giá qua tập validation/test độc lập. Việc này giảm rủi ro Overfitting. Các tham số được chọn lựa để cân bằng giữa RMSE và R².

---

## 3. PHÂN TÍCH FRONTEND (Streamlit)

### 3.1. Tab EDA (RẤT QUAN TRỌNG)
*   **WHAT:** Tập hợp các biểu đồ Line chart theo thời gian, Scatter plot, Heatmap tương quan.
*   **WHY:** Xác nhận trực quan các giả thuyết khí tượng. Điều này giúp tăng niềm tin trước khi đưa vào mô hình học máy.
*   **HOW:** Hệ thống lọc bỏ các giá trị ban đêm (Nhiễu) trước khi vẽ. Biểu đồ Heatmap biểu diễn ma trận hệ số tương quan Pearson, giúp phát hiện ra `cos_solar_zenith` là feature mang tính quyết định lớn nhất. Nhận xét: Các biểu đồ có tính thực tiễn cao, không bị dư thừa.

### 3.2. Tab Model Results
*   **WHAT:** Bảng so sánh 11 mô hình qua các chỉ số RMSE, MAE, R².
*   **WHY:** Chỉ số RMSE nhạy cảm với các sai số lớn, rất phù hợp với bài toán cảnh báo sức khỏe (đoán sai đỉnh UV cực kỳ nguy hiểm). 
*   **HOW:** Trích xuất kết quả đánh giá đã chạy offline. Việc trực quan hóa các metrics này qua Bar chart cho phép hội đồng giám khảo dễ dàng đối chiếu tính công bằng (Fairness) giữa các thuật toán.

### 3.3. Tab Forecast
*   **WHAT:** Giao diện dự báo UV 7 ngày với các thẻ (cards) tương tác.
*   **WHY:** Giúp người dùng lên kế hoạch bảo vệ bản thân khi đi du lịch hoặc hoạt động ngoài trời.
*   **HOW:** Lấy input từ thời tiết 7 ngày tới, cho chạy qua mô hình đã được chọn. **Điểm đáng khen ngợi:** Thuật toán vẽ biểu đồ đã chủ động thay thế giá trị dự báo UV ban đêm bằng `NaN` (`_add_night_gaps(df)`). Điều này ngăn chặn Plotly vẽ một đường chéo dốc ngược nối liền từ hoàng hôn hôm nay sang bình minh ngày mai, làm biểu đồ trở nên chính xác và chuyên nghiệp.

---

## 4. TAB TOURISM RECOMMENDATION (ĐIỂM NHẤN CỦA HỆ THỐNG)

### 4.1. Khai phá Công thức và Logic (Formulas & Logic)

#### A. Safe Exposure Time (Dựa trên ScanSkinAI & Thang đo Fitzpatrick)
*   **WHAT (Công thức):**
```text
Safe_Minutes = Base_Time_SkinType / max(1, UV_Index)
```
*   **WHY:** WHO và các tổ chức Da liễu phân loại da người thành 6 loại (Fitzpatrick Scale). Da loại 1-2 (sáng màu) có ít Melanin, dễ cháy nắng trong thời gian ngắn. Hàm max(1, UV) đảm bảo không xảy ra lỗi chia cho 0 khi không có UV.
*   **HOW (Thực thi):** `SKIN_BASE_SAFE_TIME = {1: 15, 2: 20, 3: 30, 4: 40, 5: 60, 6: 90}`. File `recommendation.py` duyệt qua từng mốc giờ dự báo, chia số Base Time cho mức UV để trả về số phút an toàn.

#### B. Recommendation Score (Điểm Gợi ý Điểm đến)
*   **WHAT (Công thức cốt lõi hiện tại):**
```text
Score = Average_Safe_Ratio * (1 + Shade_Coverage / 200) * Indoor_Bonus
```
*   **WHY:** Thời gian an toàn không phải là yếu tố duy nhất. Các địa điểm có độ che phủ cây xanh (Shade) hoặc có khu vực trong nhà (Indoor) sẽ giảm lượng tia UV trực tiếp, do đó an toàn hơn.
*   **HOW:** Hệ thống tính toán `safe_ratio = min(safe_minutes / activity_duration, 1.0)`. Nơi có bóng râm được nhân hệ số `shade_bonus = 1.0 + (shade_cov / 200)`. Nơi trong nhà nhân `indoor_bonus = 1.3`. Cuối cùng sắp xếp (Sort) các địa điểm theo điểm số.

### 4.2. Đánh giá Tính khoa học và Đề xuất Tối ưu (Advanced Formulas)
*   **Đánh giá độ chính xác (Correctness):** Logic tính toán hiện tại hoàn toàn đúng đắn về mặt sinh học phân tử và vật lý học (sự che phủ râm mát).
*   **Đề xuất Nâng cấp (Weighted Scoring Optimization):**
    Để hệ thống có khả năng cá nhân hóa cao hơn (Personalized Recommender), thuật toán tính `Score` nên được mở rộng thành một hàm tối ưu tổng trọng số (Weighted Sum):
```text
Total_Score = (w1 * UV_Safety_Score) + (w2 * Distance_Score) + (w3 * User_Preference_Match)
```
    Trong đó:
    *   w1, w2, w3: Các tham số trọng số (có thể điều chỉnh bởi user hoặc mô hình Machine Learning).
    *   `Distance_Score`: Sử dụng hàm suy giảm theo khoảng cách (Distance Decay), ví dụ: `e^(-lambda * distance)`.
    *   `User_Preference_Match`: Mức độ phù hợp với sở thích loại hình du lịch (Tính bằng Cosine Similarity).

---

## 5. TAB SYSTEM ASSESSMENT (ĐÁNH GIÁ OFFLINE HỆ THỐNG GỢI Ý)

Dựa trên cơ sở lý luận từ `Danh_gia_he_khuyen_nghi` và đối chiếu vào source code `evaluation.py`:

*   **WHAT & WHY:** Hệ thống không chỉ gợi ý suông mà còn tự đánh giá bản thân (Offline Evaluation) dựa trên tập dữ liệu kiểm thử (Test Scenarios) kết hợp Ground Truth. Điều này mang ý nghĩa sống còn trong nghiên cứu Khoa học máy tính.
*   **HOW (Chi tiết các chỉ số):**
    1.  **Precision@5 (Độ chính xác Top 5):** `(Số địa điểm phù hợp trong Top 5) / 5`. Đo lường có bao nhiêu địa điểm trong Top 5 thực sự đáp ứng tiêu chuẩn của user.
    2.  **Recall@5 (Độ bao phủ Top 5):** `(Số địa điểm phù hợp trong Top 5) / (Tổng số địa điểm phù hợp có thể có)`. Đo lường xem hệ thống có bỏ sót địa điểm tốt nào không.
    3.  **NDCG@5 (Normalized Discounted Cumulative Gain):** Vị trí gợi ý rất quan trọng. Thuật toán này sẽ "phạt" điểm nếu địa điểm tốt nhất bị đẩy xuống hạng 3 hoặc hạng 4 thay vì hạng 1.
    4.  **MRR (Mean Reciprocal Rank):** Vị trí trung bình của kết quả đúng đầu tiên.
*   **Đánh giá độ tin cậy:** Giao diện Evaluation trình bày việc so sánh thuật toán Baseline (Random, Popularity, Distance Only) với Hệ thống UV hiện tại. Biểu đồ chỉ ra rằng thuật toán UV kết hợp mang lại Precision@5 vượt trội. Đây là minh chứng học thuật vô cùng mạnh mẽ cho Hội đồng bảo vệ.

---

## 6. ĐÁNH GIÁ CHẤT LƯỢNG CODE (CODE IMPLEMENTATION REVIEW)

*   **Tính toàn vẹn của Feature Engineering:** Hoàn toàn chính xác, không xảy ra rò rỉ dữ liệu (Data Leakage) vì thao tác xử lý Time-Series Lag/Rolling được làm rất cẩn thận, nhóm (groupby) đúng theo `location_id`.
*   **Logic Hệ thống Khuyến nghị:** Cấu trúc phân lớp tốt. Tách biệt được hàm tính khoảng cách (Haversine), tính an toàn da (`_compute_safe_minutes`), và chấm điểm (`_score_places`).
*   **Hiệu suất (Inefficiencies & Refactoring):**
    *   *Thuật toán:* Hàm lọc `_filter_nearby_places` hiện đang sử dụng vòng lặp duyệt qua tất cả danh mục địa điểm. Nếu quy mô mở rộng lên 1.000.000 địa điểm, hàm này sẽ gây thắt cổ chai. Đề xuất ứng dụng Spatial Indexing (như KD-Tree của thư viện `scipy.spatial` hoặc Geohash) để tăng tốc query lên ngưỡng `O(log n)` thay vì `O(n)`.
    *   *Refactor:* Trong `forecast.py`, các hằng số cấu hình màu sắc và text tiếng Việt nên được tách riêng ra file `constants.py`.

---

## 7. ĐỊNH HƯỚNG MỞ RỘNG VÀ BÀI BÁO KHOA HỌC (BONUS)

Dự án này sở hữu chiều sâu chuyên môn xuất sắc. Để tiến xa hơn:

1.  **Hướng phát triển Nghiên cứu học thuật (Research Paper Direction):**
    *   *Tên bài báo gợi ý:* "A Context-Aware Tourism Recommendation System Integrating UV Radiation Forecasting and Biological Skin Sensitivity".
    *   *Nơi đăng bài tiềm năng:* Các hội thảo KSE, RIVF. Đóng góp (Contribution) chính của bài báo sẽ là việc kết nối thành công dữ liệu Khí tượng học với Hệ thống Recommender cá nhân hóa.
2.  **Hướng Triển khai Production:**
    *   Sử dụng Redis để cache dữ liệu API thời tiết (giảm latency).
    *   Thiết lập Airflow DAG để tự động lấy dữ liệu và retrain Model định kỳ (chống Data Drift).
