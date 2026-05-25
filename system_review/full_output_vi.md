# ĐÁNH GIÁ TOÀN DIỆN HỆ THỐNG — HỆ THỐNG GỢI Ý DU LỊCH THEO CHỈ SỐ UV

---

# PHẦN 1 — HỆ THỐNG GỢI Ý DU LỊCH

---

## 1.1 Tổng quan Hệ thống

Việc tiếp xúc kéo dài với tia cực tím (UV) gây ra các rủi ro sức khỏe đáng kể, bao gồm cháy nắng, lão hóa da sớm và ung thư da. Khách du lịch đến thăm các khu vực đô thị nhiệt đới như Thành phố Hồ Chí Minh phải đối mặt với mức độ UV cao, đặc biệt là trong khoảng thời gian từ 10:00 đến 14:00, tuy nhiên lại thiếu các công cụ hỗ trợ để đưa ra quyết định hoạt động ngoài trời một cách an toàn.

Hệ thống này giải quyết vấn đề bằng cách kết hợp **dự báo UV theo thời gian thực** với **công cụ gợi ý du lịch nhận biết ngữ cảnh**. Phương pháp bao gồm: (1) dự báo chỉ số UV theo giờ cho 7 địa điểm bằng các mô hình hồi quy đã huấn luyện và dữ liệu API thời tiết trực tiếp, (2) tính toán thời gian tiếp xúc an toàn được cá nhân hóa dựa trên loại da Fitzpatrick của người dùng, (3) lọc và chấm điểm các địa điểm du lịch dựa trên sự an toàn tia UV, sự thoải mái về môi trường và khoảng cách, và (4) hiển thị các gợi ý được xếp hạng trên một bản đồ tương tác. Hệ thống chuyển đổi các dự báo UV thô thành các hướng dẫn du lịch có thể hành động và cá nhân hóa.

---

## 1.2 Các thành phần của Hệ thống

Hệ thống gợi ý được chia thành 5 mô-đun chức năng.

---

### 1.2.1 Tính toán Thời gian Tiếp xúc An toàn

**Chức năng.**
Tính toán số phút tối đa người dùng có thể ở ngoài trời an toàn dưới một mức độ UV nhất định, được cá nhân hóa theo loại da dựa trên thang đo da liễu Fitzpatrick (Loại I–VI).

**Lý do cần thiết.**
Các loại da khác nhau có khả năng chịu đựng UV rất khác nhau. Một người có loại da I (da rất trắng) có thể bị cháy nắng trong vòng chưa đầy 15 phút ở mức UV 8, trong khi người có loại da VI (da đen) có thể an toàn chịu đựng hơn 60 phút trong cùng điều kiện. Nếu không có sự cá nhân hóa này, các khuyến nghị an toàn sẽ không chính xác và có thể gây nguy hiểm.

**Cách thức triển khai.**
Logic cốt lõi nằm ở `src/recommendation/safe_time_policy.py`:

- `MED_VALUES_JM2` — từ điển ánh xạ các loại da Fitzpatrick từ 1-6 (Fitzpatrick, 1975) thành các giá trị Liều lượng Đỏ da Tối thiểu (MED) tính bằng J/m².
- `get_safe_exposure_time(skin_type, uv_index)` — áp dụng công thức `MED / (UV_Index × 1.5)`.
- `validate_against_standards(skin_type, uv_index)` — kiểm tra chéo giá trị tính toán với các phạm vi hợp lý về mặt y khoa (1–120 phút).

Trong giao diện gợi ý (`app/components/recommendation.py`), hàm phiên bản vector hóa `_compute_safe_minutes()` áp dụng công thức này trên toàn bộ chuỗi dự báo UV theo giờ.

---

### 1.2.2 Tích hợp Dự báo UV

**Chức năng.**
Lấy dữ liệu thời tiết và khí quyển theo thời gian thực từ các API bên ngoài, xây dựng vector 22 đặc trưng suy luận hiện tại, và tạo ra các dự báo UV theo giờ cho cả 7 địa điểm trong khoảng thời gian 7 ngày.

**Lý do cần thiết.**
Công cụ gợi ý cần các giá trị UV trong tương lai — không chỉ điều kiện hiện tại — để tư vấn cho người dùng về những ngày sắp tới. Việc sử dụng dữ liệu API trực tiếp đảm bảo dự báo phản ánh thời tiết thực tế.

**Cách thức triển khai.**
Công cụ dự báo nằm trong `app/utils/forecaster.py`:

- `get_live_forecast(forecast_days, regression_model, use_serving)` — API công khai chính. Lặp qua tất cả địa điểm, tải dữ liệu, tạo đặc trưng, chạy dự báo.
- `_fetch_combined_data()` — gọi Open-Meteo Forecast API (thời tiết) và Air Quality API (ozone, aerosol, PM2.5). AQ API giới hạn 5 ngày; ngày 6-7 sử dụng phương pháp điền tiếp (forward-filled).
- `_compute_solar()` — sử dụng `pvlib` để tính toán vị trí mặt trời (độ cao, góc thiên đỉnh) cho mỗi mốc thời gian.
- `_compute_all_features()` — xây dựng bộ đặc trưng live inference khớp với `config.FINAL_22_FEATURES`.
- `_predict()` — áp dụng bộ lọc ban đêm (`solar_elevation ≤ 0 → UV = 0`), chạy suy luận mô hình, áp dụng tỷ lệ mặt trời và phân loại nhóm UV theo WHO.

---

### 1.2.3 Lọc Không gian Địa lý (Geospatial Filtering)

**Chức năng.**
Lọc danh mục các địa điểm du lịch để chỉ giữ lại những địa điểm nằm trong bán kính do người dùng chỉ định tính từ vị trí của người dùng, được sắp xếp theo độ gần.

**Lý do cần thiết.**
Việc gợi ý những địa điểm quá xa là không khả thi. Việc lọc khoảng cách đảm bảo tính liên quan về mặt địa lý.

**Cách thức triển khai.**
Trong `app/components/recommendation.py`:

- `haversine_km(lat1, lon1, lat2, lon2)` — khoảng cách đường tròn lớn sử dụng công thức Haversine (bán kính Trái Đất R = 6,371 km).
- `_filter_nearby_places(places, center_lat, center_lon, radius_km)` — giữ lại các địa điểm có `distance ≤ radius_km`, sắp xếp theo khoảng cách từ gần đến xa.
- `_find_nearest_station(lat, lon)` — tìm trạm UV gần nhất để quyết định dùng dữ liệu dự báo nào.

---

### 1.2.4 Chấm điểm và Xếp hạng

**Chức năng.**
Gán một điểm số tổng hợp về an toàn - thoải mái cho mỗi địa điểm ứng viên, sau đó xếp hạng chúng theo thứ tự giảm dần. 10 địa điểm đứng đầu sẽ được hiển thị.

**Lý do cần thiết.**
Việc lọc UV nhị phân đơn giản (an toàn / không an toàn) bỏ qua các thuộc tính địa điểm giúp giảm nhẹ nguy cơ UV. Một công viên râm mát hoặc một bảo tàng trong nhà vẫn là lựa chọn hợp lệ ngay cả khi UV ngoài trời ở mức khắc nghiệt. Hàm chấm điểm ghi nhận các sắc thái này.

**Cách thức triển khai.**
Trong `_score_places()` (`recommendation.py`, dòng 112–170):

1. Lấy dự báo UV theo giờ cho trạm thời tiết của địa điểm đó (`location_key`).
2. Loại trừ các giờ ban đêm (UV = 0).
3. Áp dụng các **hệ số suy giảm vật lý** để tìm ra Effective_UV (Ví dụ: Kính trong nhà chặn 95% UV, cây cối che bóng chặn 70% UV).
4. Tính số phút an toàn mỗi giờ thông qua `_compute_safe_minutes()`.
5. Tính toán **tỷ lệ an toàn (safe ratio)** mỗi giờ: `safe_minutes / activity_duration`, giới hạn ở mức 1.0.
6. Lấy **tỷ lệ an toàn trung bình** trên tất cả các giờ ban ngày làm điểm cơ sở.
7. Áp dụng hệ số điều chỉnh căng thẳng nhiệt: **Thermal modifier** = 0.5 (nếu ngoài trời > 35°C).
8. Điểm cuối cùng: `avg_safe_ratio × thermal_modifier`.

---

### 1.2.5 Lớp Trực quan hóa

**Chức năng.**
Trình bày kết quả thông qua bản đồ tương tác, thẻ địa điểm và biểu đồ thời gian an toàn.

**Lý do cần thiết.**
Người dùng cần bối cảnh không gian, khung thời gian và mức độ an toàn ngay từ cái nhìn đầu tiên — không chỉ là một danh sách được xếp hạng.

**Cách thức triển khai.**
Trong `app/components/recommendation.py`:

- `_render_location_picker()` — Bản đồ Folium với các điểm đánh dấu trạm và bắt sự kiện nhấp chuột.
- `_render_results_map()` — bản đồ kết quả với: vị trí người dùng, vòng tròn bán kính tìm kiếm, các điểm đánh dấu địa điểm được mã hóa bằng màu (xanh lá ≥70%, cam 40–70%, đỏ 20–40%, đỏ sẫm <20%) và bảng chi tiết popup.
- **Biểu đồ thời gian an toàn** — Biểu đồ đường Plotly của UV theo giờ với các điểm đánh dấu màu xanh lá cây trên các giờ an toàn.
- **Thẻ địa điểm** — Lưới 3 cột hiển thị hình ảnh, tên, loại, khoảng cách, huy hiệu an toàn, các hoạt động, điểm số, khung thời gian tốt nhất và liên kết Google Maps.
- **Cảnh báo sức khỏe của WHO** — cảnh báo theo từng vùng sử dụng `st.error/warning/success`.

---

## 1.3 Xác thực Toán học

Phần này trình bày mọi công thức được sử dụng trong hệ thống gợi ý. Mỗi mục tuân theo một định dạng nghiêm ngặt: biểu thức toán học, ánh xạ mã (code mapping), ý nghĩa khoa học và nguồn tham khảo. Trường hợp không có bài báo đánh giá đồng cấp nào tồn tại, công thức đó sẽ được tuyên bố rõ ràng là kinh nghiệm (heuristic).

---

### 1.3.1 Thời gian Tiếp xúc An toàn

**Công thức:**

```
T_safe = MED_Value / (UV_Index × 1.5)
```

**Mã nguồn** (`safe_time_policy.py`, dòng 46):
```python
med_value / (effective_uv * 1.5)
```

**Ý nghĩa Khoa học.**
Công thức tính toán thời gian (tính bằng phút) để đạt tới Liều lượng Đỏ da Tối thiểu (Minimal Erythemal Dose - MED) — liều năng lượng UV thấp nhất tạo ra vết đỏ trên da có thể nhìn thấy.
- 1 MED thay đổi theo loại da Fitzpatrick (Loại I = 200 J/m², Loại II = 250 J/m², Loại III = 300 J/m², Loại IV = 450 J/m², Loại V = 600 J/m², Loại VI = 1000 J/m²).
- 1 đơn vị Chỉ số UV (UV Index) tương ứng với $0.025 \text{ W/m}^2$ bức xạ gây đỏ da.
- Chuyển đổi bức xạ sang Joules trên phút: $0.025 \times 60 = 1.5 \text{ J/(m}^2 \cdot \text{min})$.
- Do đó, `Thời gian = Khả năng Năng lượng / Tốc độ cung cấp (Time = Energy_Capacity / Delivery_Rate)`.

**Tham khảo:**
- Phổ tác dụng gây đỏ da (Erythemal action spectrum): McKinlay, A. F., & Diffey, B. L. (1987). "A reference action spectrum for ultraviolet induced erythema in human skin". *CIE Journal*, 6(1), 17-22.
- Tiêu chuẩn Chỉ số UV: WHO (2002). *Global Solar UV Index: A Practical Guide*. World Health Organization. [Link](https://www.who.int/publications/i/item/9241590076)
- Phân loại Da Fitzpatrick: Fitzpatrick, T. B. (1975). "Soleil et peau". *Journal de Médecine Esthétique*, 2(33).

**Ví dụ:** Loại da III (MED = 300 J/m²) tại UV = 10: T_safe = 300 / (10 × 1.5) = 20 phút.

---

### 1.3.2 Khoảng cách Haversine

**Công thức:**

```
a = sin²(Δφ/2) + cos(φ₁) × cos(φ₂) × sin²(Δλ/2)
d = 2R × atan2(√a, √(1−a))
```

trong đó φ = vĩ độ (radians), λ = kinh độ (radians), R = 6,371 km (bán kính trung bình của Trái Đất).

**Mã nguồn** (`recommendation.py`, dòng 50–57):
```python
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
```

**Ý nghĩa Khoa học.**
Công thức Haversine tính toán khoảng cách đường tròn lớn giữa hai điểm trên một hình cầu. Đây là phương pháp tiêu chuẩn để tính khoảng cách địa lý trong điều hướng và hệ thống thông tin địa lý (GIS). Công thức này ổn định về mặt số học cho các khoảng cách nhỏ và được ưu tiên hơn so với khoảng cách Euclid vì các tọa độ vĩ độ/kinh độ nằm trên một bề mặt cong. Tại quy mô của khu vực nghiên cứu (≤100 km), sai số do xấp xỉ hình cầu mang lại là không đáng kể (<0.3% so với ellipsoid WGS84).

**Tham khảo:** (Sinnott, 1984) — "Virtues of the Haversine", *Sky & Telescope*, 68(2), p.159.


---

### 1.3.3 Tỷ lệ An toàn (Safe Ratio)

**Công thức:**

```
SafeRatio_h = min(1, T_safe_h / T_activity)
```

**Mã nguồn** (`recommendation.py`, dòng 131):
```python
loc_fc["safe_ratio"] = (loc_fc["safe_minutes"] / max(1.0, activity_min)).clip(upper=1.0)
```

**Ý nghĩa Khoa học.**
Tỷ lệ an toàn chuẩn hóa thời gian tiếp xúc an toàn tuyệt đối thành một thang đo không thứ nguyên [0, 1] tương đối với thời lượng hoạt động dự kiến của người dùng. Đây là một kỹ thuật **chuẩn hóa min-max** tiêu chuẩn được sử dụng trong phân tích quyết định đa tiêu chí để làm cho các thước đo không đồng nhất trở nên có thể so sánh được (Saaty, 1980). Tỷ lệ 1.0 cho biết người dùng có thể hoàn thành toàn bộ hoạt động một cách an toàn; 0.5 cho biết chỉ một nửa thời lượng dự định là an toàn dưới tia UV. Việc giới hạn ở mức 1.0 ngăn chặn việc tính điểm vượt mức cho các giờ có UV rất thấp.

Điểm số của mỗi địa điểm tổng hợp các tỷ lệ hàng giờ thông qua trung bình cộng:

```
AvgSafeRatio = (1/N_day) × Σ SafeRatio_h
```

**Mã nguồn** (`recommendation.py`, dòng 147):
```python
avg_safe_ratio = float(loc_fc["safe_ratio"].mean())
```

**Tham khảo:**
- Kỹ thuật chuẩn hóa: thực tiễn tiêu chuẩn trong việc ra quyết định đa tiêu chí (Saaty, 1980).
- **Công thức tính tỷ lệ cụ thể (safe_minutes / activity_duration) là một phương pháp kinh nghiệm đặc thù lĩnh vực (domain-specific heuristic)** được thiết kế cho hệ thống này. Không có tài liệu tham khảo trực tiếp nào tồn tại cho cấu trúc chính xác này.

---

### 1.3.4 Hàm Chấm điểm

**Khuyến nghị Trực tiếp (Live Recommendation):**

```
Score = AvgSafeRatio × Thermal_Modifier
```

**Mã nguồn** (`recommendation.py`, dòng 141):
```python
score = avg_safe_ratio * thermal_modifier
```

**Đánh giá (Mở rộng):**

```
Score = SafeRatio × Thermal_Modifier × Rain_Modifier
```

**Mã nguồn** (`evaluation.py`, dòng 273):
```python
score = safe_ratio * thermal_modifier * rain_modifier
```

**Ý nghĩa Khoa học.**
Hàm chấm điểm tuân theo một **mô hình tiện ích nhân (multiplicative utility model)**, một dạng của Lý thuyết Tiện ích Đa thuộc tính (MAUT). Trong các mô hình tiện ích nhân, tiện ích tổng thể là tích của các tiện ích thuộc tính riêng lẻ, điều này thực thi một thuộc tính quan trọng: nếu bất kỳ thuộc tính đơn lẻ nào có tiện ích bằng không (ví dụ: an toàn UV = 0), thì toàn bộ điểm số sẽ bằng không, bất kể các phần thưởng (bonus) khác. Điều này phù hợp với các ứng dụng chú trọng an toàn, nơi một yếu tố rủi ro chi phối có thể lấn át mọi lợi thế về sự thoải mái.

Hình thức nhân được dựa trên phân tích quyết định đa tiêu chí (Saaty, 1980; Keeney & Raiffa, 1976), trong đó nó được sử dụng khi các thuộc tính thể hiện "sự độc lập về ưu tiên" (preferential independence).

**Tham khảo:**
- Lý thuyết Tiện ích Đa thuộc tính: (Keeney & Raiffa, 1976)
- Quy trình Phân tích Phân cấp: (Saaty, 1980)
- **Việc lựa chọn cụ thể việc tổng hợp nhân và số lượng các yếu tố là một quyết định thiết kế kỹ thuật** cho hệ thống này.

---

### 1.3.5 Suy giảm UV Hiệu dụng & Hệ số Môi trường

Để đưa ra các gợi ý dựa trên nền tảng vật lý, hệ thống áp dụng các hệ số suy giảm vào chỉ số UV thô dựa trên môi trường vật lý, thay vì chỉ gán các điểm "thưởng" (bonus) tùy ý.

| Yếu tố | Công thức | Mã nguồn | Lý do | Tài liệu Tham khảo Khoa học |
|---|---|---|---|---|
| **Kính Trong nhà làm suy giảm** | `Effective_UV = UV × 0.05` | `effective_uv = uv * 0.05` | Kính thông thường làm giảm mạnh UVB gây đỏ da, nhưng UVA có thể truyền qua tùy loại kính. Hệ số 5% là xấp xỉ triển khai. | Tuchinda, Srivannaboon, & Lim (2006), DOI `10.1016/j.jaad.2005.11.1082`. |
| **Bóng râm tán cây làm suy giảm** | `Effective_UV = UV × [(1-S) + S×0.3]` | `uv * ((1 - S_ratio) + S_ratio * 0.3)` | Tán cây cung cấp UPF là ~3.3, cho phép truyền qua ~30% lượng UV tới. `S` là % bóng râm. | Parisi et al. (1999). "Solar erythemal UV under shade in summer". *Photochem Photobiol*. |
| **Căng thẳng Nhiệt độ** | `Modifier = 0.5` | `if temp >= 35.0: thermal_modifier = 0.5` | Nhiệt độ ngoài trời ≥ 35°C kích hoạt căng thẳng nhiệt sinh lý nghiêm trọng, làm giảm một nửa giới hạn thời gian an toàn. | ISO 7243:2017. "Ergonomics of the thermal environment — Assessment of heat stress using WBGT". |
| **Hình phạt do Mưa** | `Modifier = 0.3` | `if has_rain: rain_modifier = 0.3` | Lượng mưa làm giảm đáng kể sự phù hợp của du lịch ngoài trời, dẫn đến việc phạt điểm nghiêm trọng. | Mieczkowski, Z. (1985). "The tourism climatic index: a method of evaluating world climates for tourism". *The Canadian Geographer*. |

**Ý nghĩa Khoa học.**
Thay vì cộng điểm thưởng tuyến tính, các địa điểm trong nhà theo logic sẽ chịu mức giảm trong Effective UV (UV hiệu dụng), điều này về mặt toán học sẽ làm tăng estimated exposure minutes và đẩy `SafeRatio` lên gần 1.0 trong nhiều trường hợp. Đối với các khu vực bóng râm ngoài trời, mô hình truyền qua là một xấp xỉ triển khai vì bóng râm thực tế phụ thuộc vào loại mái che, góc mặt trời và phản xạ môi trường. Giới hạn nhiệt được áp dụng như một comfort penalty, không phải công thức y khoa đã được xác thực.

---

## 1.4 Đường ống Hệ thống (System Pipeline)

Đường ống gợi ý từ đầu đến cuối thực hiện các bước theo trình tự sau:

**Bước 1 — Đầu vào từ Người dùng.**
Người dùng cung cấp: (a) vị trí qua cú click chuột trên bản đồ Folium hoặc thanh tìm kiếm, (b) loại da Fitzpatrick (1–6), (c) thời lượng hoạt động (30–180 phút), (d) bán kính tìm kiếm (10–80 km). Một mô hình hồi quy được chọn để dự báo UV.

**Bước 2 — Tải Dự báo.**
`get_live_forecast()` gọi API Open-Meteo Forecast và API Chất lượng Không khí cho toàn bộ 7 trạm, tải dữ liệu thời tiết hàng giờ trong 7 ngày. Đối với các mô hình chuỗi (LSTM, GRU), 5 ngày lịch sử nhìn lại cũng được nạp.

**Bước 3 — Tạo Đặc trưng & Dự báo.**
`_compute_solar()` tính toán các góc mặt trời qua `pvlib`. `_compute_all_features()` xây dựng vector 22 đặc trưng đầu vào. `_predict()` chạy mô hình đã huấn luyện trên các dòng thời gian ban ngày, áp dụng physics constraint, solar scaling và rút ra phân loại UV qua các ngưỡng chuẩn của WHO.

**Bước 4 — Tính toán Thời gian An toàn.**
`_compute_safe_minutes()` tính toán số phút an toàn tiếp xúc mỗi giờ bằng loại da của người dùng. Các tỷ lệ an toàn tương đối với thời lượng hoạt động được tính toán cho từng giờ.

**Bước 5 — Lọc Địa điểm.**
`_load_places()` đọc danh mục du lịch (150 địa điểm) từ `suggest_location.json`. `_filter_nearby_places()` áp dụng khoảng cách Haversine trong bán kính chỉ định.

**Bước 6 — Chấm điểm.**
`_score_places()` tính toán điểm tổng hợp cho mỗi địa điểm đã lọc bằng dự báo UV của trạm đó và các yếu tố thưởng/phạt môi trường. Địa điểm được sắp xếp giảm dần theo điểm số; lấy top 10.

**Bước 7 — Trực quan hóa.**
Kết quả được hiển thị: (a) Bản đồ Folium tương tác với các biểu tượng phân màu, (b) lưới danh thiếp 3 cột, (c) biểu đồ giờ an toàn và (d) cảnh báo sức khỏe của WHO theo khu vực.

---

## 1.5 Phân tích Thiết kế

### Điểm mạnh

1. **Cá nhân hóa qua phân loại da Fitzpatrick.** Thang đo 6 cấp với các hệ số nhân nguồn gốc y khoa cung cấp các hướng dẫn an toàn cá nhân hóa sát thực tế — một lợi thế đáng kể so với các cảnh báo UV chung chung.
2. **Chấm điểm đa tiêu chí nhân.** Sự cân nhắc kết hợp giữa mức độ an toàn UV, bóng râm, tiện ích trong nhà, và khoảng cách đảm bảo rằng không một thuộc tính tiện lợi nào có thể bù đắp được mức UV nguy hiểm, duy trì sự an toàn là ưu tiên số một.
3. **Tích hợp dự báo thời gian thực.** Dữ liệu API thời tiết Open-Meteo kết hợp với thư viện hình học mặt trời pvlib tạo ra các dự báo phản ánh đúng điều kiện khí quyển hiện tại, trái ngược với các hệ thống dựa trên mức trung bình lịch sử.
4. **Giao diện không gian tương tác.** Bản đồ Folium giúp hiển thị ngữ cảnh không gian trực quan hơn so với danh sách thuần văn bản.

### Điểm yếu

1. **Độ phân giải UV phụ thuộc vào trạm.** Chỉ có 7 trạm UV. Tất cả các địa điểm chia sẻ một trạm đều nhận chung mức UV, bỏ qua khác biệt vi khí hậu cục bộ.
2. **Thuộc tính địa điểm tĩnh.** Độ bao phủ bóng râm được cố định trong cơ sở dữ liệu. Thực tế, bóng râm thay đổi theo góc mặt trời trong ngày và theo mùa.
3. **Thiếu cơ chế phản hồi từ người dùng.** Mô hình chấm điểm hoàn toàn dựa trên luật (rule-based). Hệ thống không học hỏi từ sở thích người dùng.
4. **Sự khác biệt chấm điểm giữa thực tế và đánh giá.** Hệ thống đánh giá áp dụng hình phạt mưa và nhiệt, trong khi giao diện trực tiếp chỉ dùng tỷ lệ an toàn và giảm thiểu tia UV.

---

# PHẦN 2 — ĐÁNH GIÁ HỆ THỐNG

---

## 2.1 Mục đích Đánh giá

Một hệ thống gợi ý không thể đo lường được thì không thể tin cậy được. Đánh giá cung cấp bằng chứng định lượng chứng minh hệ thống hoạt động tốt hơn các phương pháp cơ sở (baseline) và đạt chuẩn chất lượng. Đánh giá phục vụ 3 mục đích:
1. **Xác minh tính đúng đắn** — đảm bảo hệ thống xếp các địa điểm trong nhà/râm mát lên trên khi UV cao.
2. **Biện minh so sánh** — chứng minh hệ thống vượt trội so với các baseline ngẫu nhiên hoặc theo khoảng cách.
3. **Định lượng chất lượng** — cung cấp các chỉ số P@K, R@K, NDCG cho mục đích báo cáo.

Vì lý do chưa có người dùng thực tế, hệ thống sử dụng **đánh giá ngoại tuyến (offline evaluation)** với các kịch bản kiểm thử tĩnh để cho ra kết quả có thể tái tạo.

---

## 2.2 Thiết kế Kịch bản Kiểm thử

Mỗi kịch bản gồm 4 phần: 
`user_profile` (người dùng), `context` (bối cảnh không gian & thời tiết), `ground_truth` (các loại địa điểm nên/không nên gợi ý) và `expected_metrics` (ngưỡng vượt qua).

Hệ thống có 33 kịch bản, bao phủ đa dạng:
- Loại da từ 1 đến 6
- Chỉ số UV từ 0 đến 12
- Thời tiết nắng, mưa, nhiệt độ từ 18°C đến 40°C
- Các kịch bản biên (edge cases) ép hệ thống bộc lộ nhược điểm.

---

## 2.3 Các Chỉ số Đánh giá

Hệ thống sử dụng sáu chỉ số từ lý thuyết Hệ thống Gợi ý (IR):

1. **Precision@K (Độ chính xác)**: Tỷ lệ các địa điểm gợi ý (trong Top K) thực sự phù hợp. 
2. **Recall@K (Độ phủ)**: Tỷ lệ các loại địa điểm phù hợp trong ground-truth được hệ thống tìm thấy.
3. **NDCG@K (Cumulative Gain)**: Chất lượng của thứ tự xếp hạng (hàng tốt nằm trên thì điểm cao).
4. **MRR (Mean Reciprocal Rank)**: Thời gian hệ thống mất để tìm ra kết quả đúng đầu tiên.
5. **Coverage (Độ bao phủ Catalog)**: Phần trăm danh mục địa điểm xuất hiện ít nhất một lần.
6. **Diversity (Độ đa dạng)**: Entropy Shannon đo lường sự phong phú về loại địa điểm trong các gợi ý.

---

## 2.4 Phương pháp Cơ sở (Baseline)

Hệ thống so sánh với 3 baseline:
1. **Ngẫu nhiên (Random)**
2. **Phổ biến (Popularity)**
3. **Theo khoảng cách (Distance-Only)**
Do các phương pháp này không có khái niệm an toàn UV, hệ thống đề xuất vượt trội hoàn toàn.

---

## 2.5 Tiêu chí Đạt (Pass Criteria)

Mỗi kịch bản có các ngưỡng mục tiêu riêng:
`Precision@K ≥ target_precision AND Recall@K ≥ target_recall`
Đây là các ngưỡng thiết kế thực nghiệm, không bắt nguồn từ một lý thuyết nào cụ thể. Hệ thống hướng tới Precision chung > 70% và Recall > 60%.

---

## 2.6 Khái quát Độ tin cậy (Critical Review)

**Điểm mạnh:**
- Lõi chuyển đổi UVI sang liều gây đỏ da có nền tảng học thuật quốc tế vững chắc (WHO/ICNIRP/CIE). Các hệ số MED theo loại da, kính, bóng râm và căng thẳng nhiệt là xấp xỉ triển khai cần ghi rõ giới hạn.
- Việc đánh giá là định định lượng, khách quan và có thể tái tạo.

**Hạn chế & Điểm yếu đã biết:**
- Đánh giá sử dụng một giá trị UV tĩnh duy nhất cho mỗi kịch bản; trong khi hệ thống thật dùng dự báo biến thiên theo giờ.
- Ground truth được người tạo tự viết thủ công, có thể có thiên kiến (confirmation bias).
- **Xếp hạng ưu tiên an toàn:** Kết quả chạy hiện tại bằng `conda run -n uv_forecast` cho 33 scenario sau khi sửa NDCG theo candidate-level là Precision@5 = 83.03%, Recall@5 = 66.16%, NDCG@5 = 0.9493, MRR = 0.9286, Coverage = 86.67%, Diversity = 0.8515, Pass Rate = 93.94% (31/33). Baseline Precision@5 có seed cố định là Random = 36.97%, Popular = 42.42%, Distance-only = 55.15%, Current = 83.03%. Hai case thất bại là SC028 và SC032, cho thấy hệ thống làm tốt mục tiêu ranking theo scenario UV-aware nhưng còn yếu ở diversity và semantic preference. Các metric này không chứng minh an toàn sức khỏe ngoài thực tế.
