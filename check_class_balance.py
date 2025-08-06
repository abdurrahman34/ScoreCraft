import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
print("Veri seti yükleniyor...")
df = pd.read_csv('application_train_with_features.csv')

# TARGET değişkeninin dağılımını kontrol et
print("\nTARGET değişkeni sınıf dağılımı:")
target_counts = df['TARGET'].value_counts()
print(target_counts)

print("\nYüzdelik dağılım:")
target_percentage = df['TARGET'].value_counts(normalize=True) * 100
print(target_percentage)

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.countplot(x='TARGET', data=df)
plt.title('TARGET Değişkeni Sınıf Dağılımı')
plt.xlabel('TARGET (0: Ödeme Yapabilir, 1: Ödeme Yapamaz)')
plt.ylabel('Sayı')

# Sayıları ve yüzdeleri ekle
for i, count in enumerate(target_counts):
    percentage = target_percentage.iloc[i]
    plt.text(i, count + 500, f"{count} ({percentage:.2f}%)", ha='center')

plt.savefig('class_balance.png')
print("\nGörselleştirme 'class_balance.png' dosyasına kaydedildi.")

# Dengesizlik oranını hesapla
imbalance_ratio = target_counts.iloc[0] / target_counts.iloc[1]
print(f"\nDengesizlik oranı (çoğunluk sınıf / azınlık sınıf): {imbalance_ratio:.2f}")

if imbalance_ratio > 3:
    print("\nSınıf dağılımı oldukça dengesiz. SMOTE uygulanması önerilir.")
else:
    print("\nSınıf dağılımı kabul edilebilir düzeyde dengesiz. SMOTE gerekli olabilir veya olmayabilir.")
