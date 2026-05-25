# BÁO CÁO ĐÁNH GIÁ KỸ THUẬT VÀ HỌC THUẬT: HỆ THỐNG DỰ BÁO UV, CẢNH BÁO VÀ GỢI Ý DU LỊCH

## 1. TỔNG QUAN DỰ ÁN (OVERALL PROJECT REVIEW)

### 1.1. Kiến trúc hệ thống
*   **WHAT:** Hệ thống được thiết kế chia làm hai phần rõ rệt: Backend xử lý dữ liệu/mô hình (`UV_analysis`) và Frontend phục vụ tương tác người dùng (`UV_analysis_UI` sử dụng Streamlit).
*   **WHY:** Thiết kế Decoupled (tách rời) này giúp dễ dàng mở rộng (scalability). Người phát triển có thể thay đổi, huấn luyện lại mô hình ở Backend mà không làm sập giao diện UI.
*   **HOW:** Dữ liệu thô từ nhiều nguồn được xử lý qua Data Pipeline (`merger.py`, `engr.py`). Sau đó, 11 mô hình tối ưu hóa được huấn luyện/đánh giá để dự báo cường độ UV: `rf`, `dt`, `xgb`, `lgb`, `catboost`, `lstm`, `gru`, `bilstm`, `cnn_lstm`, `attention_lstm`, `prophet_lgb`. Frontend tải mô hình cục bộ hoặc gọi Databricks Serving, lấy dữ liệu dự báo trực tiếp từ Open-Meteo, và kết nối với Tourist Catalog để hiển thị EDA, model results, forecast, recommendation và system assessment.

### 1.2. Đánh giá Ưu nhược điểm & Tính ứng dụng
*   **Ưu điểm:** Backend tích hợp Open-Meteo, WeatherBit, OpenUV và Open-Meteo Air Quality trong pipeline dữ liệu. Frontend forecast hiện dùng Open-Meteo Forecast + Air Quality API để tạo dữ liệu đầu vào theo thời gian thực, sau đó dùng `pvlib` và công thức MED/WHO để tính khuyến nghị an toàn.
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

#### B. Bộ đặc trưng hiện tại trong `config.FINAL_22_FEATURES`
Mã hiện tại dùng 22 đặc trưng chống rò rỉ dữ liệu cho suy luận frontend và huấn luyện tối ưu. Các biến UV lag/rolling và các biến bức xạ đo được (`shortwave_radiation`, `direct_radiation`, `clearness_index`, `uv_clear_sky_ratio`, v.v.) được loại khỏi `FINAL_22_FEATURES` vì có nguy cơ leakage hoặc không có sẵn ổn định tại thời điểm dự báo.

**Nhóm 1: Đặc trưng Thời gian (Temporal Features)**
*Tại sao sử dụng:* Thời gian tuyến tính (1, 2, 3... 23, 24) làm mất đi tính tuần hoàn tự nhiên (ví dụ 23h đêm rất gần với 1h sáng). Việc biến đổi thời gian giúp mô hình học được chu kỳ ngày/đêm và chu kỳ các mùa trong năm một cách trơn tru.
*   **Trong `FINAL_22_FEATURES`:** chỉ còn `doy_sin` được dùng trực tiếp để biểu diễn mùa. Các biến `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `doy_cos`, `day_fraction`, `hours_since_sunrise` vẫn có thể được tạo trong `FeatureEngineer`, nhưng không nằm trong bộ đặc trưng cuối cho mô hình tối ưu hiện tại.

**Nhóm 2: Bức xạ & Vị trí Mặt trời (Solar & Radiation Features)**
*Tại sao sử dụng:* Tia UV tới mặt đất phụ thuộc tuyệt đối vào việc mặt trời đang đứng ở góc nào so với mặt đất và khoảng cách tia sáng phải xuyên qua lớp khí quyển.
*   **9. `cos_solar_zenith`**: Cosin của góc thiên đỉnh. Đây là **đặc trưng quan trọng nhất**, tỷ lệ thuận trực tiếp với cường độ bức xạ.
*   **10. `cos_zenith_squared`**: Thành phần phi tuyến của góc mặt trời.
*   **11. `solar_cloud_interaction`**: `cos_solar_zenith * (1 - cloud_cover)`, mô phỏng tương tác giữa góc mặt trời và mây.
*   **12. `cloud_attenuation_exp`**: Cloud Modification Factor theo Kasten & Czeplak: `1 - 0.75 * (cloud_cover/100)^3.4`.
*   **Ghi chú:** `solar_elevation`, `air_mass`, `clearness_index` có thể được tạo trong pipeline nhưng không thuộc `FINAL_22_FEATURES`.

**Nhóm 3: Khí quyển & Môi trường (Atmospheric Features)**
*Tại sao sử dụng:* Mây, tầng ozone và khói bụi là những "tấm khiên" tự nhiên hấp thụ hoặc tán xạ tia tử ngoại trước khi chúng chạm đất.
*   **14. `temperature_2m`**: Nhiệt độ không khí.
*   **15. `relative_humidity_2m`**: Độ ẩm tương đối.
*   **16. `ozone_anomaly`**: Độ lệch ozone so với baseline trung bình theo tháng và địa điểm.
*   **17. `pressure_msl`**: Áp suất mực nước biển, đại diện trạng thái hệ thời tiết.
*   **18. `wind_speed_10m`**: Tốc độ gió, liên quan phân tán aerosol và trạng thái thời tiết.
*   **19. `temp_humidity_product`**: Tương tác nhiệt độ - độ ẩm.
*   **20. `pressure_cloud_interaction`**: Tương tác áp suất - mây.
*   **21. `temperature_2m_ema`, `cloud_cover_ema`, `ozone_ema`**: Trung bình mũ để mô tả quán tính gần đây của thời tiết/khí quyển.
*   **22. `altitude_solar_interaction`, `ozone_depletion_risk`, `air_quality_combined`, `is_raining`, `cloud_cover_change_1h`**: Các biến tương tác, rủi ro ozone thấp, chất lượng không khí, mưa và xu hướng mây.

**Nhóm 4: Đặc trưng Chuỗi thời gian (Lag, Rolling & Trend Features)**
*Tại sao sử dụng:* UV có tính tự tương quan (autocorrelation) cao. Bức xạ của giờ hiện tại phụ thuộc rất lớn vào xu hướng của các giờ trước đó.
*   **Hiện trạng:** `uv_lag_*`, `uv_rolling_*`, `uv_diff_*`, `uv_max_today_so_far` vẫn được tạo cho một số bảng phân tích trong backend, nhưng bị loại khỏi danh sách regression cuối để tránh dùng tín hiệu UV quan sát/quá khứ không nhất quán với live forecast.

### 2.2. Huấn luyện Mô hình
*   **WHAT:** Benchmark 11 mô hình tối ưu hóa: Random Forest, Decision Tree, XGBoost, LightGBM, CatBoost, LSTM, GRU, BiLSTM, CNN-LSTM, Attention-LSTM và Prophet+LGB.
*   **WHY:** Do bản chất dữ liệu UV có yếu tố phi tuyến tính mạnh (phụ thuộc vào tương tác mây - ozone), các mô hình Linear thường hoạt động kém, việc so sánh nhiều mô hình giúp tìm ra thuật toán tối ưu.
*   **HOW:** Kết quả hiện tại trong `results/optimized/consolidated_results.csv` có 33 dòng (11 model x train/val/test). Trên test set, `attention_lstm` đang tốt nhất theo RMSE: MAE=0.3361, RMSE=0.5737, R²=0.9326, MAPE=13.29%.

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

#### A. Estimated Exposure Limit (Dựa trên WHO UVI, MED và thang Fitzpatrick)
*   **WHAT (Công thức):**
```text
Estimated_Exposure_Minutes = MED_Value / (Effective_UV * 1.5)
```
*   **WHY:** WHO/ICNIRP định nghĩa 1 UVI = 0.025 W/m² bức xạ gây đỏ da, tương đương 1.5 J/(m²·phút). `MED_VALUES_JM2` là giá trị đại diện theo nhóm da Fitzpatrick, không phải ngưỡng lâm sàng tuyệt đối cho mọi cá nhân.
*   **HOW (Thực thi):** `MED_VALUES_JM2 = {1: 200, 2: 250, 3: 300, 4: 450, 5: 600, 6: 1000}` trong `src/recommendation/safe_time_policy.py`. Nếu `Effective_UV <= 0.01`, hệ thống trả về trần 480 phút để tránh chia cho gần 0. Đây là ước lượng hỗ trợ quyết định, không phải tư vấn y khoa.

#### B. Recommendation Score (Điểm Gợi ý Điểm đến)
*   **WHAT (Công thức cốt lõi hiện tại):**
```text
Live Score = Average_Safe_Ratio * Thermal_Modifier
```
*   **WHY:** Phiên bản hiện tại không còn dùng `indoor_bonus = 1.3` hoặc `shade_bonus = 1 + shade/200`. Indoor/shade được đưa vào bằng cách giảm UV hiệu dụng trước khi tính estimated exposure minutes.
*   **HOW:** Trong `_score_places()`, địa điểm indoor dùng `effective_uv = uv_predicted * 0.05`; địa điểm outdoor dùng `effective_uv = uv_predicted * ((1 - shade_ratio) + shade_ratio * 0.3)`. Sau đó `safe_ratio = min(estimated_exposure_minutes / activity_duration, 1.0)`, điểm live là trung bình safe ratio nhân `thermal_modifier` (0.5 nếu outdoor và nhiệt độ trung bình >= 35°C).

### 4.2. Đánh giá Tính khoa học và Đề xuất Tối ưu (Advanced Formulas)
*   **Đánh giá độ chính xác (Correctness):** Lõi chuyển đổi UVI -> liều gây đỏ da có cơ sở khoa học, nhưng các giá trị MED theo loại da, hệ số indoor/shade, thermal modifier và rain modifier là các xấp xỉ/giả định triển khai cần ghi rõ giới hạn.
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

*   **WHAT & WHY:** Hệ thống có tab Offline Evaluation dựa trên `data/evaluation/test_scenarios.json`. Hiện có 33 kịch bản, gồm cả các edge case SC026-SC035. Lưu ý: evaluation dùng hàm giả lập `_make_fake_recommendations_from_scenario()` với UV scalar trong scenario, không gọi trực tiếp live `_score_places()` trên dữ liệu forecast 7 ngày.
*   **HOW (Chi tiết các chỉ số):**
    1.  **Precision@5 (Độ chính xác Top 5):** `(Số địa điểm phù hợp trong Top 5) / 5`. Đo lường có bao nhiêu địa điểm trong Top 5 thực sự đáp ứng tiêu chuẩn của user.
    2.  **Recall@5 (Độ bao phủ Top 5):** `(Số địa điểm phù hợp trong Top 5) / (Tổng số địa điểm phù hợp có thể có)`. Đo lường xem hệ thống có bỏ sót địa điểm tốt nào không.
    3.  **NDCG@5 (Normalized Discounted Cumulative Gain):** Vị trí gợi ý rất quan trọng. Thuật toán này sẽ "phạt" điểm nếu địa điểm tốt nhất bị đẩy xuống hạng 3 hoặc hạng 4 thay vì hạng 1.
    4.  **MRR (Mean Reciprocal Rank):** Vị trí trung bình của kết quả đúng đầu tiên.
*   **Đánh giá độ tin cậy:** Chạy bằng môi trường `conda uv_forecast`, kết quả hiện tại sau khi sửa NDCG candidate-level là Precision@5=83.03%, Recall@5=66.16%, NDCG@5=0.9493, MRR=0.9286, Coverage=86.67%, Diversity=0.8515, Pass Rate=93.94% (31/33). Hai kịch bản fail: SC028 và SC032. Baseline Precision@5 trung bình có seed cố định: Random=36.97%, Popular=42.42%, Distance-only=55.15%, Current=83.03%.

---

## 6. ĐÁNH GIÁ CHẤT LƯỢNG CODE (CODE IMPLEMENTATION REVIEW)

*   **Tính toàn vẹn của Feature Engineering:** Hoàn toàn chính xác, không xảy ra rò rỉ dữ liệu (Data Leakage) vì thao tác xử lý Time-Series Lag/Rolling được làm rất cẩn thận, nhóm (groupby) đúng theo `location_id`.
*   **Logic Hệ thống Khuyến nghị:** Cấu trúc phân lớp tốt. Tách biệt được hàm tính khoảng cách (Haversine), ước lượng giới hạn phơi nhiễm (`_compute_safe_minutes`), và chấm điểm (`_score_places`).
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
