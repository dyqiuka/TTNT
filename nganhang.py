#----------------------- Mạng Neural Nhân Tạo để phân loại khách hàng --------------------#
# Nhập các thư viện cần thiết
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#----------------------- Bước 1: Trực quan hóa dữ liệu (EDA) ----------------------#
# Đọc dữ liệu từ tệp CSV
bank_data = pd.read_csv("Artificial_Neural_Network_Case_Study_data.csv")

# Hiển thị thông tin tổng quan về dữ liệu
print("Thông tin tổng quan về dữ liệu:")
print(bank_data.info())

# Hiển thị thống kê mô tả
print("\nThống kê mô tả:")
print(bank_data.describe())

# Lọc ra những cột số quan trọng để vẽ pairplot
selected_columns = ['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Exited']

# Vẽ biểu đồ pairplot với các cột đã chọn và cải tiến hiển thị
sns.pairplot(bank_data[selected_columns], hue='Exited', diag_kind='hist', 
             plot_kws={'s': 30, 'alpha': 0.8, 'edgecolor': 'black'},  # Tăng kích thước và viền
             palette="coolwarm")
plt.show()

# Mã hóa các biến phân loại thành số để tính ma trận tương quan
bank_data_encoded = bank_data.copy()
for col in bank_data_encoded.select_dtypes(include=['object']).columns:
    bank_data_encoded[col] = LabelEncoder().fit_transform(bank_data_encoded[col])

# Vẽ ma trận tương quan sau khi mã hóa dữ liệu
plt.figure(figsize=(10,6))
sns.heatmap(bank_data_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Ma trận tương quan sau khi mã hóa dữ liệu")
plt.show()

#----------------------- Bước 2: Tiền xử lý dữ liệu ----------------------#
# Kiểm tra giá trị bị thiếu
print("\nSố lượng giá trị bị thiếu trong mỗi cột:")
print(bank_data.isnull().sum())

# Chọn các cột đặc trưng và biến mục tiêu
X = bank_data.iloc[:, 3:-1].values  # Đặc trưng (bỏ cột ID, Số tài khoản)
y = bank_data.iloc[:, -1].values    # Biến mục tiêu (Khách hàng có rời đi không)

# Mã hóa biến giới tính (Label Encoding)
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])  # 0: Nữ, 1: Nam

# Mã hóa biến quốc gia (One-Hot Encoding)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Hiển thị 5 dòng đầu sau khi mã hóa
print("Dữ liệu sau khi mã hóa biến phân loại (5 dòng đầu):")
print(pd.DataFrame(X).head())

#----------------------- Bước 3: Chia tập dữ liệu ----------------------#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Chuẩn hóa dữ liệu (Scale về cùng khoảng giá trị)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#----------------------- Bước 4: Xây dựng mô hình ANN -----------------------#
# Khởi tạo mô hình ANN
ann = tf.keras.models.Sequential()

# Thêm lớp ẩn thứ nhất (6 neuron, hàm kích hoạt ReLU)
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Thêm lớp ẩn thứ hai (6 neuron, hàm kích hoạt ReLU)
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Thêm lớp đầu ra (1 neuron, hàm kích hoạt Sigmoid)
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#----------------------- Bước 5: Huấn luyện mô hình -----------------------#
# Biên dịch mô hình với thuật toán tối ưu Adam, hàm mất mát Binary Crossentropy
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình trong 100 epochs
history = ann.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

#----------------------- Đánh giá mô hình ---------------------#
# Vẽ biểu đồ quá trình huấn luyện
plt.figure(figsize=(14, 6))

# Biểu đồ mất mát
plt.subplot(1, 2, 1)
sns.lineplot(x=range(1, len(history.history['loss']) + 1), y=history.history['loss'], label='Train Loss', marker='o')
sns.lineplot(x=range(1, len(history.history['val_loss']) + 1), y=history.history['val_loss'], label='Test Loss', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Mất mát trong quá trình huấn luyện')
plt.legend()
plt.grid(True)

# Biểu đồ độ chính xác
plt.subplot(1, 2, 2)
sns.lineplot(x=range(1, len(history.history['accuracy']) + 1), y=history.history['accuracy'], label='Train Accuracy', marker='o')
sns.lineplot(x=range(1, len(history.history['val_accuracy']) + 1), y=history.history['val_accuracy'], label='Test Accuracy', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Độ chính xác trong quá trình huấn luyện')
plt.legend()
plt.grid(True)

plt.show()

# Hiển thị ma trận nhầm lẫn
y_pred = (ann.predict(X_test) > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black',
            xticklabels=['Không rời', 'Rời'], yticklabels=['Không rời', 'Rời'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.show()