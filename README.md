#  Titanic - Machine Learning Project

##  Mục tiêu
Dự án nhằm **dự đoán khả năng sống sót của hành khách trên tàu Titanic** dựa trên dữ liệu nhân khẩu học và thông tin vé.  
Tập trung vào việc **xử lý dữ liệu thiếu (missing values)**, **kỹ thuật feature engineering nâng cao**, và **xây dựng mô hình học máy ổn định, tránh data leakage**.

---

##  Tổng quan quy trình

1. **Nạp dữ liệu gốc (`train.csv`, `test.csv`)**
2. **Phân tích sơ bộ (EDA)**: quan sát phân phối của `Survived`, `Sex`, `Pclass`, `Embarked`, v.v.
3. **Tạo ba chiến lược xử lý missing values**:
   - Median imputation
   - Group median (theo Sex + Pclass)
   - Model-based imputation (RandomForest)
4. **Feature Engineering nâng cao** (FamilySize, Title, Deck, FarePerPerson, AgeClass, …)
5. **Tách dữ liệu sớm để chống data leakage** (train/validation)
6. **Huấn luyện & tuning mô hình** (Random Forest, XGBoost, Logistic Regression, Ensemble)
7. **Đánh giá từng biến thể & tạo VotingClassifier**
8. **Áp dụng pipeline lên tập test và sinh `submission.csv`**

---

##  Chi tiết các bước

### 1️ Chia dữ liệu & chống data leakage
Ngay sau khi đọc dữ liệu, dataset `train` được chia thành:
- `raw_train` (train set)
- `raw_val` (validation set)

 Mọi bước xử lý missing, fit imputer/model đều được huấn luyện **chỉ trên `raw_train`**  
 Sau đó mới **apply lên `raw_val`** để tránh rò rỉ thông tin (data leakage).

---

### 2️ Xử lý Missing Values

#### **A. Median Imputation**
- `Age`, `Fare` → thay bằng median của train.  
- `Embarked` → thay bằng mode.  
- Tạo cờ `HasCabin` (1 nếu có Cabin, 0 nếu không).  

 Nhanh, đơn giản, baseline ổn định.

#### **B. Group Median Imputation (theo Sex + Pclass)**
- `Age` → median theo nhóm `(Sex, Pclass)`.  
- `Fare` → median theo `Pclass`.  
- `Embarked` → mode.  

 Tận dụng cấu trúc nhóm để cải thiện độ chính xác.

#### **C. Model-based Imputation**
- Train **RandomForestRegressor** dự đoán `Age`.
- Train **RandomForestClassifier** dự đoán `Embarked`.
- Dùng các biến như `Pclass`, `Sex`, `SibSp`, `Parch`, `Fare`, `HasCabin` để dự đoán.  

 Có thể chính xác hơn, nhưng cần kiểm soát overfitting.  
 Models được fit **chỉ trên raw_train**, apply lên validation/test để tránh leakage.

---

### 3️ Feature Engineering (tạo đặc trưng mới)

| Nhóm | Biến mới | Giải thích |
|------|-----------|------------|
|  Gia đình | `FamilySize`, `IsAlone`, `FamilyCategory` | Cô đơn hay đi cùng người thân ảnh hưởng đến khả năng sống sót |
|  Title | `Title`, `TitleAgeType` | Trích xuất từ tên (Mr, Mrs, Miss, Master, Rare...) — proxy cho giới tính & tuổi |
|  Vé & giá | `FarePerPerson`, `FareBin`, `FareClass` | Chuẩn hóa giá vé theo số người và nhóm giá |
|  Cabin | `Deck`, `HasCabin` | Sàn tàu và có cabin hay không |
|  Vé | `TicketPrefix`, `TicketLen` | Mã vé cho thấy hạng/điểm khởi hành |
|  Tuổi | `AgeBin`, `AgeClass` | Phân nhóm tuổi & tương tác với Pclass |
|  Tương tác | `ClassSex`, `ClassEmbarked`, `TitleSex`, `WomenChild`, `HighClassFemale` | Domain knowledge (ví dụ: "Women & Children first") |

---

### 4️ Tiền xử lý (Preprocessing)
Dùng `ColumnTransformer`:
- **Numeric** → `StandardScaler`
- **Categorical** → `OneHotEncoder(drop='first', handle_unknown='ignore')`

---

### 5️ Huấn luyện & Tuning mô hình
####  Mô hình được thử nghiệm:
- Logistic Regression
- Random Forest (tuning qua GridSearchCV)
- XGBoost (tuning qua GridSearchCV)
- Voting Ensemble (soft voting giữa 3 model)

####  Đánh giá:
- Accuracy (chính xác)
- Có thể mở rộng thêm: Precision, Recall, F1, ROC AUC, Confusion Matrix

####  GridSearchCV:
- RandomForest:
  - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- XGBoost:
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`

---

### 6️ Ensemble Model (VotingClassifier)
- Kết hợp 3 mô hình:
  - Logistic Regression
  - Random Forest (best params)
  - XGBoost (best params)
- `voting='soft'` (dựa trên xác suất, không phải nhãn cứng)
- Pipeline cuối cùng:
  ```python
  pipe_voting = Pipeline([
      ('pre', preprocessor),
      ('model', voting_classifier)
  ])
