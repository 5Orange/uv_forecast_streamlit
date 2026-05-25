# TÀI LIỆU CHUYÊN SÂU: KIỂM ĐỊNH VÀ ĐÁNH GIÁ HỆ THỐNG (SYSTEM ASSESSMENT)

Để trả lời thuyết phục các câu hỏi liên quan đến Tab Assessment và chứng minh hệ thống của bạn hoạt động dựa trên **Cơ sở khoa học (Scientific Evidence)** chứ không phải giả định cảm tính (Assumption), hãy nắm vững 5 luận điểm dưới đây:

---

## 1. Làm thế nào để định nghĩa một Test Case (Kịch bản kiểm thử)?

Chúng ta không test hệ thống bằng cách "chạy thử vài lần xem có ra kết quả đẹp không". Chúng ta định nghĩa các Test Case một cách nghiêm ngặt trong file `data/evaluation/test_scenarios.json`. 

Mỗi Test Case (Scenario) được cấu thành từ 4 phần (Ví dụ kịch bản `SC001`):
1. **User Profile (Hồ sơ người dùng):** Người da sáng màu (Loại 2), muốn đi chơi 60 phút.
2. **Context (Ngữ cảnh môi trường):** 12h trưa, UV = 11 (Cực kỳ nguy hiểm), Nhiệt độ 35°C.
3. **Ground Truth (Sự thật gốc - Căn cứ đánh giá):** 
   - `should_recommend`: Bắt buộc phải là `indoor_attraction` (Khu trong nhà).
   - `should_avoid`: Tuyệt đối cấm `coastal_beach` (Bãi biển).
   - *Lý do y khoa:* UV 11 thì thời gian an toàn của da loại 2 chỉ là 1.8 phút. Đi biển 60 phút là nguy hiểm tính mạng.
4. **Expected Metrics (Kỳ vọng):** Hệ thống phải đạt Precision $\ge$ 0.6 cho case này.

**=> Kết luận cho Hội đồng:** "Thưa thầy cô, các Test Case của em là các phép thử ranh giới (Edge-cases) được suy chuẩn từ bảng thời gian cháy nắng của WHO. Em đặt ra Ground Truth nghiêm ngặt, bắt buộc hệ thống phải bảo vệ tính mạng người dùng trong các điều kiện khắc nghiệt nhất (mưa lớn, nắng gắt, da trắng)."

---

## 2. Hệ thống tính toán Precision, Recall như thế nào? (Kèm Code)

Mã nguồn tại `src/recommendation/evaluation.py` thực hiện vòng lặp chấm điểm tự động cho tất cả 33 test cases trong `data/evaluation/test_scenarios.json`.

### Precision@K (Độ chính xác Top-K)
Trọng tâm: Trong K địa điểm hệ thống gợi ý, có bao nhiêu địa điểm **thực sự an toàn và phù hợp** (nằm trong `should_recommend`)?
```python
# Code thực tế trong hệ thống:
@staticmethod
def precision_at_k(recommendations, ground_truth, k=5):
    should = set(ground_truth.get("should_recommend", []))
    if not should:
        return 0.0
    
    top_k = recommendations[:k]
    # Đếm số địa điểm có loại hình nằm trong danh sách an toàn
    relevant = sum(1 for r in top_k if r.get("type") in should)
    
    # Chia cho K (VD: Có 3 địa điểm đúng trong Top 5 -> Precision = 3/5 = 0.6)
    return relevant / max(1, len(top_k))
```

### Recall@K (Độ bao phủ Top-K)
Trọng tâm: Trong tất cả các loại hình địa điểm an toàn, hệ thống có tìm ra được sự đa dạng không, hay chỉ tìm được 1 loại?
```python
@staticmethod
def recall_at_k(recommendations, ground_truth, k=5):
    should = set(ground_truth.get("should_recommend", []))
    top_k = recommendations[:k]
    
    # Tìm tập giao (intersection) giữa các loại hình được gợi ý và Ground Truth
    found_types = {r.get("type") for r in top_k} & should
    return len(found_types) / len(should)
```

---

## 3. Làm sao để phân tích và khẳng định Hệ thống là TỐT & ĐÚNG?

Hệ thống được chứng minh là "Tốt" khi nó vượt qua bài Test về sự **So sánh Baseline (Baseline Comparison)**. 

Trong `evaluation.py`, class `BaselineComparator` sẽ ép hệ thống UV của bạn thi đấu trực tiếp với 3 thuật toán cơ bản:
1. **Random:** Gợi ý ngẫu nhiên bừa bãi.
2. **Popularity:** Luôn luôn gợi ý chỗ nổi tiếng nhất (Chợ Bến Thành, Dinh Độc Lập...) bất kể nắng mưa.
3. **Distance Only:** Luôn gợi ý chỗ GẦN NHẤT.

**Cách phân tích kết quả:**
Trên giao diện Tab Assessment, biểu đồ Bar Chart sẽ cho thấy hệ thống UV của bạn có `Precision@5` cao hơn hẳn thuật toán "Distance Only".
Tại sao? Bởi vì vào lúc 12h trưa UV mức 11, thuật toán "Gần nhất" sẽ đẩy người dùng ra bãi biển vì bãi biển cách đó 1km, bỏ qua rủi ro cháy nắng. Trong khi đó, hệ thống UV sẽ "phạt" điểm bãi biển về 0, và hướng người dùng tới khu mua sắm trong nhà (Indoor) dù cách xa 5km. Điều này minh chứng hệ thống của bạn hoạt động hoàn toàn Đúng (Correct) với logic y khoa học.

---

## 4. Trả lời Hội đồng: "Làm sao chúng tôi tin được hệ thống và các Test Case của em? Bằng chứng khoa học ở đâu?"

**Đây là câu hỏi "sát thủ". Bạn hãy tự tin trả lời theo 3 luồng luận điểm (Arguments) sau:**

**Luận điểm 1: Test Case phi thiên vị (Unbiased Ground Truth)**
"Thưa Hội đồng, các Test Case không phải do em ngồi tự tưởng tượng để hệ thống dễ dàng pass qua. Ground Truth (đáp án chuẩn) được định nghĩa chặt chẽ dựa trên Định luật vật lý và Y khoa:
- **Y khoa:** Áp dụng chuẩn ScanSkinAI & Thang Fitzpatrick. Da loại 1 phơi UV 10 chỉ được 1.5 phút. Em cài Ground Truth bắt buộc phải cấm (should_avoid) các điểm ngoài trời.
- **Vật lý:** Ở nhiệt độ 39 độ C, hoặc khi có cờ `is_raining = true`, Ground Truth bắt buộc hệ thống phải ưu tiên `indoor_option` (trong nhà). 
Đây là các tham chiếu khách quan, độc lập với code của em."

**Luận điểm 2: Minh bạch về thuật toán đánh giá (Quantitative Evaluation)**
"Để chứng minh, em không dùng ảnh chụp màn hình (screenshot) chọn lọc những kết quả đẹp. Em đã code một module `evaluation.py` chạy offline tự động tính toán bằng Toán học (Precision, Recall, NDCG, MRR, Coverage, Diversity) trên 33 kịch bản. Tuy nhiên, em cũng ghi rõ đây là scenario-based simulator dùng UV scalar, không phải kiểm thử trực tiếp toàn bộ live recommender 7 ngày."

**Luận điểm 3: Sự đánh đổi Khoảng cách - An toàn (Distance vs Safety Trade-off)**
"Bằng chứng đắt giá nhất để tin tưởng hệ thống là sự đánh đổi. Trong các kịch bản thực nghiệm, hệ thống của em sẵn sàng hi sinh tiêu chí Khoảng cách (chấp nhận gợi ý một nơi xa hơn 5km) để đổi lấy sự an toàn (nơi đó có máy lạnh và mái che) khi UV chạm ngưỡng hủy diệt. Một hệ thống chỉ dựa trên giả định (assumption) sẽ không thể tự động xử lý được bài toán tối ưu hóa đa biến này."

---

## 5. Quy mô Test Case: 25 Kịch bản đã đủ chưa? Có cần thêm không?

Đây là câu hỏi phản biện cực kỳ xuất sắc để kiểm tra xem bạn có thực sự hiểu về **Phương pháp luận Đánh giá (Evaluation Methodology)** hay không.

### 25 Test Case đã đủ chưa?
*   **Với Khóa luận tốt nghiệp (Proof of Concept):** 25 Test Case là **TẠM ĐỦ**.
*   **Với Hệ thống Thực tế (Production / Bài báo khoa học):** Con số này là **CHƯA ĐỦ**.

**Cách lập luận bảo vệ (Dành cho Hội đồng):**
"Thưa Hội đồng, nếu đây là một mô hình Machine Learning thuần túy (như Thuật toán gợi ý của Netflix hay Shopee) dựa trên dữ liệu lịch sử hàng triệu người dùng, thì 33 test case là con số còn nhỏ. Tuy nhiên, hệ thống của em là **Hệ thống Gợi ý dựa trên Luật và Ngữ cảnh (Rule-based & Context-Aware Recommender System)**, được neo vào các quy tắc y khoa vật lý cứng.

Do đó, 33 kịch bản của em không được tạo ra ngẫu nhiên, mà em sử dụng kỹ thuật **Phân tích Giá trị Biên (Boundary Value Analysis)** và **Phân vùng Tương đương (Equivalence Partitioning)** trong Kiểm thử Phần mềm. Em tập trung test thẳng vào các 'điểm mù' như nắng cực gắt, mưa lớn, da cực trắng, UV thấp nhưng outdoor nên thắng, diversity, và preference theo loại địa điểm. Kết quả hiện tại là 31/33 pass; hai fail case giúp chỉ ra hướng cải tiến preference-aware và diversity-aware ranking."

### Nên thêm bao nhiêu Test Case thì hệ thống mới đạt chuẩn hoàn hảo?
Bạn hãy chủ động đề xuất hướng giải quyết này trước khi Hội đồng kịp bắt bẻ:
"Dù 33 test case đã chứng minh được phần lớn tính đúng đắn (Correctness) của logic, nhưng em nhận thức được để hệ thống hoàn thiện ở mức độ thương mại (Production) hoặc xuất bản bài báo khoa học, em cần mở rộng lên khoảng **50 đến 100 kịch bản** và thêm kiểm thử trực tiếp cho live recommender."

### Thêm như thế nào cho khoa học? (Tuyệt đối không thêm bừa)
Đừng nói là "Em sẽ ngồi nghĩ thêm 75 cái nữa". Hãy dùng thuật ngữ học thuật:
"Để tạo ra 100 kịch bản này, em sẽ không sinh ngẫu nhiên, mà sẽ dùng phương pháp **Kiểm thử Tổ hợp (Combinatorial Testing)** hoặc **Ma trận Trực giao (Orthogonal Array Testing)**. Cụ thể, em sẽ cho hệ thống tự động sinh các tổ hợp chéo giữa các chiều không gian:
*   **6 Loại Da** (Từ sáng đến tối).
*   **5 Mức độ UV** (Low, Moderate, High, Very High, Extreme).
*   **3 Trạng thái Thời tiết** (Clear, Rain, Heatwave).
*   **3 Thời điểm** (Sáng, Trưa, Chiều tối).

$(6 \times 5 \times 3 \times 3 = 270 \text{ tổ hợp lý thuyết})$. Từ 270 tổ hợp này, em sẽ tự động hóa việc lọc ra khoảng 70-100 tổ hợp có ý nghĩa thực tiễn nhất để bổ sung vào bộ `test_scenarios.json`. Việc phủ kín ma trận (Matrix Coverage) chỉ giúp tăng độ bao phủ kiểm thử scenario; không được diễn giải là đảm bảo hệ thống an toàn 100% trong mọi tình huống thực tế."
