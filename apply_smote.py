# SMOTE uygulayarak sınıf dengesini sağlama
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os

print("Veri seti yükleniyor...")
df = pd.read_csv('application_train_with_features.csv')

# Hedef değişkeni ve özellikleri ayırma
print("\nHedef değişkeni ve özellikleri ayırıyorum...")
y = df['TARGET']
X = df.drop(columns=['TARGET'])

# ID sütunlarını kaldırma
print("ID sütunlarını kaldırıyorum...")
id_columns = ['Unnamed: 0', 'SK_ID_CURR', 'ID']
X = X.drop(columns=[col for col in id_columns if col in X.columns])

# Kategorik değişkenleri işleme
print("Kategorik değişkenleri işliyorum...")
cat_features = X.select_dtypes(include=['object']).columns.tolist()
for col in cat_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Eksik değerleri doldurma
print("Eksik değerleri dolduruyorum...")
for col in X.columns:
    if X[col].dtype in ['int64', 'float64']:
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(X[col].mode()[0])

# SMOTE öncesi sınıf dağılımını gösterme
print("\nSMOTE öncesi sınıf dağılımı:")
print(Counter(y))

# SMOTE uygulama
print("\nSMOTE uyguluyorum...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# SMOTE sonrası sınıf dağılımını gösterme
print("\nSMOTE sonrası sınıf dağılımı:")
print(Counter(y_resampled))

# Dengelenmiş veri setini kaydetme
print("\nDengelenmiş veri setini kaydediyorum...")
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['TARGET'] = y_resampled

# Dengelenmiş veri setini kaydet
output_file = 'application_train_balanced.csv'
df_resampled.to_csv(output_file, index=False)

print(f"\nDengelenmiş veri seti '{output_file}' dosyasına kaydedildi.")
print(f"Orijinal veri seti boyutu: {df.shape}")
print(f"Dengelenmiş veri seti boyutu: {df_resampled.shape}")

# Dengelenmiş veri setinin ilk 5 satırını göster
print("\nDengelenmiş veri setinin ilk 5 satırı:")
print(df_resampled.head())

# Dengelenmiş veri setinin sınıf dağılımını göster
print("\nDengelenmiş veri setinin sınıf dağılımı:")
print(df_resampled['TARGET'].value_counts())
print("\nYüzdelik dağılım:")
print(df_resampled['TARGET'].value_counts(normalize=True) * 100)
