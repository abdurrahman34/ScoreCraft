import pandas as pd
import os

# Define file paths
input_file = os.path.join('home-credit-default-risk', 'application_train.csv')
output_file = os.path.join('home-credit-default-risk', 'application_train_23cols.csv')

df = pd.read_csv(input_file)

# Get column names
all_columns = df.columns.tolist()

# Specify the columns to select
base_columns = all_columns[:19]  # İlk 19 sütun

additional_indices = [41, 34, 22, 23, 29, 30, 28,31,33]
# İndeksleri kontrol et ve geçerli olanları kullan
additional_columns = []
for i in additional_indices:
    if i <= len(all_columns):
        additional_columns.append(all_columns[i-1])
    else:
        print(f"Uyarı: {i} indeksi geçersiz. Toplam sütun sayısı: {len(all_columns)}")


# Tüm seçilen sütunları birleştir
selected_columns = base_columns + additional_columns
print(f"Selected columns: {selected_columns}")
print(f"Total selected columns: {len(selected_columns)}")

# Create a new dataframe with selected columns
df_selected = df[selected_columns]


# YENİ ÖZELLİKLER OLUŞTURMA
print("\nYeni özellikler oluşturuluyor...")

# 1. Oran Özellikleri
df_selected['CREDIT_INCOME_RATIO'] = (df_selected['AMT_CREDIT'] / df_selected['AMT_INCOME_TOTAL']).round(2)
df_selected['CREDIT_YEARS'] = (df_selected['AMT_CREDIT'] / df_selected['AMT_ANNUITY']).round(1)

# 2. Yaş Özellikleri
df_selected['AGE_YEARS'] = (abs(df_selected['DAYS_BIRTH']) / 365.25).round(1)
df_selected['AGE_GROUP'] = pd.cut(df_selected['AGE_YEARS'], 
                                bins=[0, 25, 35, 45, 55, 65, 100], 
                                labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])

# 3. İş Deneyimi Özellikleri
df_selected['EMPLOYMENT_YEARS'] = (abs(df_selected['DAYS_EMPLOYED']) / 365.25).round(1)



# 4. Çocuk ve Aile Özellikleri
if 'CNT_FAM_MEMBERS' in df_selected.columns:
    df_selected['INCOME_PER_PERSON_IN_FAMILY'] = (df_selected['AMT_INCOME_TOTAL'] / df_selected['CNT_FAM_MEMBERS']).round(1)

# 5. Başvuru Zamanı Özellikleri
if 'WEEKDAY_APPR_PROCESS_START' in df_selected.columns:
    df_selected['IS_PROCESS_WEEKEND'] = df_selected['WEEKDAY_APPR_PROCESS_START'].isin(['SATURDAY', 'SUNDAY']).astype(int)


# Yeni özelliklerin sayısını göster
new_features = [col for col in df_selected.columns if col not in selected_columns]
print(f"Oluşturulan yeni özellikler: {new_features}")
print(f"Toplam yeni özellik sayısı: {len(new_features)}")

# Yeni özellikleri içeren veri setini kaydet
output_file_with_features = os.path.join('home-credit-default-risk', 'application_train_with_features.csv')
df_selected.to_csv(output_file_with_features, index=True, index_label='ID')

# İndeksi 1'den başlat
df_final = pd.read_csv(output_file_with_features)
df_final.index = df_final.index + 1  # İndeksi 1'den başlat
df_final.to_csv(output_file_with_features)

print(f"\nYeni özelliklerle birlikte veri seti kaydedildi: {output_file_with_features}")
print(f"Son veri seti boyutu: {df_selected.shape[0]} satır, {df_selected.shape[1]} sütun")

# Dosyayı tekrar okuyup indeksi 1'den başlat
df_final = pd.read_csv(output_file)
df_final.index = df_final.index + 1  # İndeksi 1'den başlat
df_final.to_csv(output_file)

print(f"New file created: {output_file}")
print(f"The new file contains {df_selected.shape[0]} rows and {df_selected.shape[1]} columns.")
