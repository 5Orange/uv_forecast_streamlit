# Giai thich Tab System Assessment trong Streamlit

Tai lieu nay giai thich tab **"Danh gia he thong" / "System Assessment"** trong ung dung Streamlit cua do an UV Forecast UI. Muc tieu la giup trinh bay voi hoi dong cham thi: tab nay la gi, testcase duoc dinh nghia nhu the nao, "expectation" nghia la gi, va cac chi so danh gia nen duoc hieu ra sao.

## 1. Tab System Assessment la gi?

Tab **System Assessment** la phan danh gia dinh luong cho he thong goi y du lich an toan theo chi so UV. No khong phai la tab du bao UV truc tiep, ma la tab kiem tra xem:

> Voi cac tinh huong gia lap da duoc dinh nghia truoc, he thong co goi y dung loai dia diem an toan hay khong?

Trong source code, tab nay duoc khai bao trong `app/streamlit_app.py` voi ten:

```python
"Danh gia he thong"
```

Phan giao dien chinh nam trong:

```text
app/components/evaluation.py
```

Logic tinh toan cac metric nam trong:

```text
src/recommendation/evaluation.py
```

Cac testcase nam trong:

```text
data/evaluation/test_scenarios.json
```

Co the giai thich voi hoi dong nhu sau:

> Tab Assessment la module danh gia offline. Em tao cac kich ban kiem thu co ground truth, sau do he thong tu sinh top-5 goi y va tinh Precision@5, Recall@5, NDCG@5, MRR, Coverage, Diversity de chung minh he thong goi y hoat dong co do luong, khong chi dua vao cam tinh.

## 2. Testcase la gi?

Trong project nay, mot testcase duoc goi la mot **scenario**. Moi scenario mo phong mot tinh huong nguoi dung cu the, gom:

- Nguoi dung la ai: loai da, thoi gian hoat dong, so thich.
- Moi truong ra sao: dia diem, UV, nhiet do, mua hay khong.
- He thong nen goi y loai dia diem nao.
- He thong can dat nguong metric nao de duoc xem la pass.

Vi du mot testcase co dang:

```json
{
  "id": "SC001",
  "description": "Fair skin (type 2) at extreme UV noon - must go indoor",
  "user_profile": {
    "skin_type": 2,
    "activity_duration_minutes": 60,
    "preferences": ["sightseeing"]
  },
  "context": {
    "location_id": "hcm",
    "lat": 10.7769,
    "lon": 106.7009,
    "timestamp": "2025-07-15T12:00:00",
    "uv_forecast": 11,
    "temperature": 35,
    "weather_condition": "sunny",
    "is_raining": false
  },
  "ground_truth": {
    "should_recommend": [
      "observation_deck",
      "historical_site",
      "indoor_attraction",
      "market",
      "river_side"
    ],
    "should_avoid": [
      "coastal_beach",
      "urban_park",
      "nature_spot"
    ],
    "explanation": "UV=11 extreme. Indoor glass blocks most UV. Indoor places should be preferred."
  },
  "expected_metrics": {
    "target_precision": 0.6,
    "target_recall": 0.5
  }
}
```

File hien tai co nhieu scenario tu `SC001` den `SC035`, bao phu:

- Loai da Fitzpatrick tu 1 den 6.
- UV tu 0 den 12.
- Dieu kien nang, mua, nhiet do cao.
- Nhieu dia diem khac nhau nhu HCM, Vung Tau, Can Gio, Long Hai.
- Thoi gian hoat dong khac nhau.

## 3. Cach dinh nghia mot testcase

Moi testcase gom 4 phan chinh.

## 3.1 `user_profile`

Day la ho so nguoi dung:

```json
"user_profile": {
  "skin_type": 2,
  "activity_duration_minutes": 60,
  "preferences": ["sightseeing"]
}
```

Y nghia:

- `skin_type`: loai da Fitzpatrick, tu 1 den 6. Loai 1 la da rat sang va de chay nang; loai 6 la da sam mau, chiu UV tot hon.
- `activity_duration_minutes`: thoi gian nguoi dung du dinh hoat dong.
- `preferences`: so thich cua nguoi dung, vi du `sightseeing`, `walking`, `beach`.

Trong danh gia, `skin_type` va `activity_duration_minutes` rat quan trong vi chung anh huong truc tiep den thoi gian phoi nang an toan.

## 3.2 `context`

Day la ngu canh moi truong:

```json
"context": {
  "location_id": "hcm",
  "lat": 10.7769,
  "lon": 106.7009,
  "uv_forecast": 11,
  "temperature": 35,
  "is_raining": false
}
```

Y nghia:

- `location_id`: khu vuc dang xet.
- `lat`, `lon`: toa do nguoi dung, dung cho baseline theo khoang cach.
- `uv_forecast`: chi so UV trong kich ban.
- `temperature`: nhiet do.
- `is_raining`: co mua hay khong.

Trong assessment, `uv_forecast` la mot gia tri tinh cho tung scenario. Dieu nay giup danh gia co tinh lap lai, vi moi lan chay cung mot scenario se ra cung mot bo dieu kien.

## 3.3 `ground_truth`

Day la dap an chuan cua testcase:

```json
"ground_truth": {
  "should_recommend": ["indoor_attraction", "observation_deck"],
  "should_avoid": ["coastal_beach", "urban_park"],
  "explanation": "UV high, indoor places are safer"
}
```

Y nghia:

- `should_recommend`: cac loai dia diem nen xuat hien trong top-5.
- `should_avoid`: cac loai dia diem nen tranh.
- `explanation`: giai thich vi sao ground truth duoc dat nhu vay.

Diem quan trong: ground truth danh gia theo **loai dia diem**, khong phai theo ten dia diem cu the.

Vi du, neu `should_recommend` co `indoor_attraction`, thi bat ky dia diem nao co:

```json
"type": "indoor_attraction"
```

trong top-5 deu duoc tinh la dung.

## 3.4 `expected_metrics`

Day la ky vong ve ket qua do luong:

```json
"expected_metrics": {
  "target_precision": 0.6,
  "target_recall": 0.5
}
```

Mot testcase duoc xem la pass neu:

```text
Precision@5 >= target_precision
AND
Recall@5 >= target_recall
```

Logic nay nam trong `src/recommendation/evaluation.py`:

```python
sc_pass = (
    metrics["precision_at_k"] >= target_prec
    and metrics["recall_at_k"] >= target_rec
)
```

## 4. "Expectation" nghia la gi?

Trong project nay, "expectation" co 2 lop nghia.

## 4.1 Ground-truth expectation

Day la ky vong ve hanh vi cua he thong:

- He thong nen goi y loai dia diem nao.
- He thong nen tranh loai dia diem nao.
- Ly do khoa hoc hoac logic an toan la gi.

Vi du:

- UV = 11.
- Loai da = 2.
- Hoat dong = 60 phut.

Trong tinh huong nay, expectation hop ly la:

- Nen goi y dia diem trong nha, co mai che, co kinh, it phoi nang.
- Nen tranh bai bien, cong vien ngoai troi, diem thien nhien khong co bong ram.

## 4.2 Metric expectation

Day la nguong so hoc ma he thong phai dat.

Vi du:

```json
"target_precision": 0.6
```

Nghia la trong top-5 goi y, it nhat khoang 3/5 dia diem phai thuoc nhom `should_recommend`.

```json
"target_recall": 0.5
```

Nghia la he thong phai bao phu it nhat 50% cac loai dia diem tot trong ground truth.

Co the tra loi voi hoi dong:

> Expectation khong co nghia la em ep he thong phai tra ve dung mot dia diem cu the. Expectation la tieu chi danh gia duoc dinh nghia truoc: loai dia diem nao nen xuat hien, loai nao nen tranh, va Precision/Recall toi thieu bao nhieu thi testcase duoc xem la dat.

## 5. He thong tao top-5 recommendation trong Assessment nhu the nao?

Trong assessment, he thong khong goi API live. No dung scenario co san va catalog dia diem de tinh diem offline.

Logic nam trong ham:

```text
_make_fake_recommendations_from_scenario()
```

trong file:

```text
src/recommendation/evaluation.py
```

Quy trinh:

1. Doc `skin_type`, `activity_duration_minutes`, `uv_forecast`, `temperature`, `is_raining`.
2. Duyet tung dia diem trong catalog.
3. Tinh UV hieu dung cua dia diem.
4. Tinh thoi gian an toan theo loai da.
5. Tinh `safe_ratio`.
6. Ap dung penalty neu nong hoac mua.
7. Sap xep dia diem theo `score`.
8. Lay top-5.

Cong thuc diem chinh:

```text
score = safe_ratio * thermal_modifier * rain_modifier
```

Trong do:

- `safe_ratio`: muc do thoi gian an toan co dap ung thoi gian hoat dong khong.
- `thermal_modifier`: he so phat do nhiet do cao.
- `rain_modifier`: he so phat do mua.

Neu dia diem la indoor:

```python
effective_uv = uv * 0.05
```

Nghia la he thong gia dinh trong nha hoac qua kinh lam giam manh UV.

Neu dia diem la outdoor:

```python
transmission = (1.0 - shade_ratio) + (shade_ratio * 0.3)
effective_uv = uv * transmission
```

Nghia la bong ram lam giam UV hieu dung, nhung khong triet tieu hoan toan.

Neu nhiet do ngoai troi cao:

```python
if temperature >= 35.0:
    thermal_modifier = 0.5
```

Neu troi mua va dia diem outdoor:

```python
if has_rain:
    rain_modifier = 0.3
```

## 6. Cac metric trong tab Assessment

## 6.1 Precision@5

Precision@5 tra loi cau hoi:

> Trong 5 dia diem he thong goi y, co bao nhieu dia diem thuoc nhom dung?

Cong thuc:

```text
Precision@5 = so dia diem dung trong top-5 / 5
```

Vi du:

- Top-5 co 4 dia diem thuoc `should_recommend`.

```text
Precision@5 = 4 / 5 = 0.8
```

Y nghia:

- Precision cao: it goi y sai.
- Precision thap: top-5 co nhieu dia diem khong phu hop.

## 6.2 Recall@5

Recall@5 tra loi cau hoi:

> Trong tat ca loai dia diem nen goi y, he thong tim duoc bao nhieu loai?

Cong thuc:

```text
Recall@5 = so loai dia diem dung da tim thay / tong so loai trong should_recommend
```

Vi du:

- Ground truth yeu cau 4 loai dia diem.
- Top-5 chi bao phu duoc 2 loai.

```text
Recall@5 = 2 / 4 = 0.5
```

Y nghia:

- Recall cao: he thong bao phu duoc nhieu loai dia diem phu hop.
- Recall thap: he thong co the chi lap lai mot loai dia diem, thieu da dang.

## 6.3 NDCG@5

NDCG@5 danh gia chat luong thu tu xep hang.

No khong chi hoi "co dung hay khong", ma hoi:

> Dia diem dung co nam o vi tri cao trong top-5 khong?

Neu dia diem dung nam o vi tri 1 hoac 2, diem cao hon so voi nam o vi tri 4 hoac 5.

Trong code, gain duoc tinh nhu sau:

- Neu `type` nam trong `should_recommend`: gain = 2.
- Neu `type` khong nam trong `should_avoid`: gain = 1.
- Neu `type` nam trong `should_avoid`: gain = 0.

Y nghia:

- NDCG cao: he thong khong chi chon dung ma con xep dung len dau.
- NDCG thap: ket qua dung co the bi day xuong duoi.

## 6.4 MRR

MRR la Mean Reciprocal Rank. No tra loi cau hoi:

> Ket qua dung dau tien xuat hien o vi tri nao?

Neu ket qua dau tien da dung:

```text
MRR = 1 / 1 = 1.0
```

Neu ket qua dung dau tien o vi tri 2:

```text
MRR = 1 / 2 = 0.5
```

Neu o vi tri 5:

```text
MRR = 1 / 5 = 0.2
```

Y nghia:

- MRR cao: nguoi dung thay ket qua dung rat som.
- MRR thap: nguoi dung phai xem nhieu ket qua moi gap goi y phu hop.

## 6.5 Coverage

Coverage do do bao phu catalog:

> Bao nhieu phan tram dia diem trong catalog tung duoc he thong goi y it nhat mot lan?

Y nghia:

- Coverage cao: he thong khong chi goi y lap lai vai dia diem.
- Coverage thap: he thong bi lech ve mot nhom dia diem nho.

## 6.6 Diversity

Diversity do do da dang loai dia diem bang entropy.

Y nghia:

- Diversity cao: top recommendations co nhieu loai dia diem khac nhau.
- Diversity thap: he thong goi y qua nhieu dia diem cung mot loai, vi du toan indoor.

## 7. Baseline Comparison la gi?

Tab Assessment so sanh he thong hien tai voi 3 baseline:

1. `random`: chon dia diem ngau nhien.
2. `popular`: chon dia diem pho bien nhat.
3. `distance_only`: chon dia diem gan nhat.

Muc dich cua baseline la chung minh:

> He thong UV-aware tot hon cac chien luoc don gian vi no hieu rui ro UV, mua, nhiet do, indoor/outdoor va bong ram.

Vi du:

- Neu UV = 11 luc 12 gio trua, mot baseline theo khoang cach co the chon bai bien gan nhat.
- He thong UV-aware se phat diem bai bien vi phoi nang cao, va uu tien dia diem trong nha hoac co mai che.

Co the giai thich voi hoi dong:

> Baseline dung de chung minh he thong UV-aware tot hon cac chien luoc don gian. Neu chi chon gan nhat, he thong co the dua nguoi dung ra bai bien luc UV 11. Con he thong cua em tinh rui ro UV, bong ram, indoor, mua, nhiet do nen uu tien dia diem an toan hon.

## 8. Cac phan hien thi trong tab Streamlit

Trong `app/components/evaluation.py`, tab Assessment co 5 phan nho:

## 8.1 Tong quan chi so

Hien thi:

- Precision@5.
- Recall@5.
- NDCG@5.
- Coverage.
- Diversity Score.
- Pass rate.
- MRR.

Day la phan tom tat suc khoe tong quat cua he thong.

## 8.2 So sanh Baseline

Hien thi bieu do cot so sanh:

- He thong hien tai.
- Random.
- Popular.
- Distance-only.

Theo cac metric:

- Precision@5.
- Recall@5.
- NDCG@5.

## 8.3 Da dang goi y

Hien thi phan phoi cac loai dia diem trong ket qua goi y.

Muc dich:

- Kiem tra he thong co bi thien lech vao mot loai dia diem khong.
- Xem he thong co de xuat da dang indoor, outdoor, park, beach, temple, restaurant hay khong.

## 8.4 Ket qua kich ban

Hien thi tung scenario:

- ID scenario.
- Mo ta.
- Precision@5.
- Recall@5.
- Pass hay fail.
- Input gom user profile va context.
- Top-5 dia diem goi y.
- Ground truth explanation.
- Cac metric chi tiet.

Day la phan rat quan trong khi bao ve, vi co the mo tung scenario de giai thich vi sao he thong pass/fail.

## 8.5 Chi tiet diem so

Cho phep chon mot scenario va xem cach diem cua tung dia diem duoc cau thanh boi:

- UV Safety Ratio.
- Shade Bonus.
- Indoor Bonus.
- Rain Penalty.
- Heat Penalty.
- Score.

Luu y: phan score breakdown trong UI la phan giai thich truc quan, con logic scoring offline chinh trong `src/recommendation/evaluation.py` dung cong thuc:

```text
score = safe_ratio * thermal_modifier * rain_modifier
```

## 9. Cach tra loi cac cau hoi cua hoi dong

## Cau hoi: "System Assessment la gi?"

Tra loi:

> System Assessment la module danh gia offline cho he thong goi y. Em dinh nghia cac scenario gom ho so nguoi dung, ngu canh moi truong, ground truth va expected metrics. He thong sinh top-5 recommendation cho tung scenario, sau do tinh Precision@5, Recall@5, NDCG@5, MRR, Coverage va Diversity de danh gia chat luong goi y.

## Cau hoi: "Testcase cua em duoc dinh nghia nhu the nao?"

Tra loi:

> Moi testcase la mot scenario trong file `test_scenarios.json`. No gom 4 phan: `user_profile`, `context`, `ground_truth` va `expected_metrics`. `user_profile` mo ta nguoi dung, `context` mo ta UV va thoi tiet, `ground_truth` mo ta loai dia diem nen goi y hoac nen tranh, con `expected_metrics` dat nguong Precision va Recall de quyet dinh pass/fail.

## Cau hoi: "Ground truth co phai la mot dia diem cu the khong?"

Tra loi:

> Khong. Ground truth cua em danh gia theo loai dia diem, khong theo ten dia diem cu the. Vi du trong dieu kien UV cao, loai `indoor_attraction` duoc xem la phu hop. Bat ky dia diem nao thuoc loai nay trong top-5 deu duoc tinh la recommendation dung.

## Cau hoi: "Expected metrics la gi?"

Tra loi:

> Expected metrics la nguong toi thieu ma he thong phai dat trong tung scenario. Vi du `target_precision = 0.6` nghia la top-5 can co it nhat khoang 3 ket qua dung. `target_recall = 0.5` nghia la he thong can bao phu it nhat 50% cac loai dia diem phu hop trong ground truth.

## Cau hoi: "Tai sao dung Precision va Recall?"

Tra loi:

> Precision cho biet trong top-5 co bao nhieu goi y dung, con Recall cho biet he thong bao phu duoc bao nhieu loai dia diem phu hop. Hai chi so nay bo sung cho nhau: Precision cao giup tranh goi y sai, Recall cao giup ket qua khong qua hep hoac lap lai.

## Cau hoi: "Tai sao can NDCG?"

Tra loi:

> Precision chi biet co dung trong top-5 hay khong, nhung khong danh gia thu tu. NDCG danh gia ranking quality: ket qua dung nam cang cao thi diem cang cao. Dieu nay phu hop voi recommender system vi nguoi dung thuong xem cac ket qua dau tien truoc.

## Cau hoi: "Tai sao can baseline?"

Tra loi:

> Baseline giup chung minh he thong cua em tot hon cac cach lam don gian. Random chon ngau nhien, Popular chi chon noi noi tieng, Distance-only chi chon noi gan nhat. Cac baseline nay khong hieu rui ro UV. He thong cua em co them thong tin UV, loai da, bong ram, indoor, mua va nhiet do, nen co the uu tien an toan hon.

## Cau hoi: "25 hay 33 testcase co du khong?"

Tra loi:

> Voi do an tot nghiep va proof-of-concept, so testcase nay du de minh hoa va kiem dinh cac tinh huong quan trong nhu UV cuc cao, UV thap, mua, nhiet do cao, cac loai da khac nhau. Tuy nhien, neu dua len muc production hoac nghien cuu khoa hoc nghiem ngat hon, em se can mo rong len 50-100 testcase bang phuong phap combinatorial testing de phu nhieu to hop dieu kien hon.

## Cau hoi: "He thong co han che gi?"

Tra loi:

> Co. Assessment hien tai chu yeu danh gia theo loai dia diem va an toan UV. Mot so scenario co the cho thay he thong thien ve dia diem indoor vi indoor lam giam UV manh. Dieu nay tot trong dieu kien nguy hiem, nhung o UV thap hoac ban dem, he thong co the nen ton trong preference outdoor hon. Day la huong cai tien tiep theo: can can bang giua safety, user intent, khoang cach va da dang.

## 10. Tom tat mot doan de noi nhanh

Co the dung doan nay khi bao ve:

> Tab System Assessment la phan danh gia offline cua recommender system. Em tao cac testcase trong `test_scenarios.json`, moi testcase gom user profile, context, ground truth va expected metrics. He thong chay tung testcase, sinh top-5 dia diem, sau do tinh Precision@5, Recall@5, NDCG@5, MRR, Coverage va Diversity. Testcase pass khi Precision va Recall deu vuot nguong trong `expected_metrics`. Ground truth cua em khong ep dung ten dia diem cu the, ma danh gia theo loai dia diem an toan. Vi du UV cao thi indoor va co mai che la phu hop, beach hoac outdoor khong bong ram la nen tranh. Ngoai ra, em so sanh voi baseline random, popularity va distance-only de chung minh he thong UV-aware co gia tri hon cac cach goi y don gian.

