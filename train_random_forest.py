import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Veri setini yükle (SMOTE ile dengelenmiş veri seti)
print("Dengelenmiş veri setini yüklüyorum...")
balanced_data = pd.read_csv('application_train_balanced.csv')

print(f"Veri seti boyutu: {balanced_data.shape}")
print(f"Sınıf dağılımı:\n{balanced_data['TARGET'].value_counts()}")
print(f"Sınıf dağılımı (%):\n{balanced_data['TARGET'].value_counts(normalize=True) * 100}")

# Özellikleri ve hedef değişkeni ayır
X = balanced_data.drop('TARGET', axis=1)
y = balanced_data['TARGET']

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Eğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# Random Forest modeli oluştur
print("\nRandom Forest modeli oluşturuluyor...")
model = RandomForestClassifier(random_state=42)

# Daha küçük bir veri örneği üzerinde çalışalım (eğitim süresini azaltmak için)
print("Veri örnekleme yapılıyor...")
n_samples = min(50000, len(X_train))  # En fazla 50,000 örnek kullanılacak
indices = np.random.choice(len(X_train), n_samples, replace=False)
X_train_sample = X_train.iloc[indices]
y_train_sample = y_train.iloc[indices]
print(f"Örneklenmiş eğitim seti boyutu: {X_train_sample.shape}")

# En uygun parametre kombinasyonu - tek kombinasyon kullanılacak
best_params = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# Modeli en iyi parametrelerle oluştur
best_model = RandomForestClassifier(**best_params, random_state=42)

# 8-fold cross-validation ile model performansını değerlendir
print("8-fold cross-validation başlatılıyor...")

# İlerleme takibi için callback sınıfı
class ProgressCallback:
    def __init__(self, total_iterations):
        self.total_iterations = total_iterations
        self.current_iteration = 0
    
    def __call__(self, *args, **kwargs):
        self.current_iteration += 1
        progress = (self.current_iteration / self.total_iterations) * 100
        print(f"\rEğitim ilerlemesi: %{progress:.1f} ({self.current_iteration}/{self.total_iterations})", end="")

try:
    # 8-fold cross-validation
    from sklearn.model_selection import KFold
    
    cv_scores = []
    total_iterations = 8  # 8-fold cross-validation
    progress_callback = ProgressCallback(total_iterations)
    
    for train_idx, val_idx in KFold(n_splits=8, shuffle=True, random_state=42).split(X_train_sample):
        X_train_fold, X_val_fold = X_train_sample.iloc[train_idx], X_train_sample.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_sample.iloc[train_idx], y_train_sample.iloc[val_idx]
        
        # Modeli eğit
        best_model.fit(X_train_fold, y_train_fold)
        
        # ROC AUC hesapla
        y_val_proba = best_model.predict_proba(X_val_fold)[:, 1]
        score = roc_auc_score(y_val_fold, y_val_proba)
        cv_scores.append(score)
        
        # İlerlemeyi güncelle
        progress_callback()
    
    # Ortalama cross-validation skoru
    mean_cv_score = np.mean(cv_scores)
    print(f"\n8-fold cross-validation ortalama ROC AUC skoru: {mean_cv_score:.4f}")
    
    # En iyi modeli tüm eğitim seti ile eğit
    print("\nModel tüm eğitim seti ile eğitiliyor...")
    best_model.fit(X_train_sample, y_train_sample)
    
except KeyboardInterrupt:
    print("\nKullanıcı tarafından durduruldu. Şu ana kadar eğitilen model kullanılacak.")
    # Simule edilmiş ilerleme
    if not hasattr(best_model, 'classes_'):
        print("Model henüz eğitilmedi. Varsayılan parametrelerle hızlıca eğitiliyor.")
        best_model.fit(X_train_sample, y_train_sample)
    
    if not hasattr(best_model, 'classes_'):
        print("\nBasit model eğitiliyor...")
        best_model.fit(X_train_sample, y_train_sample)

# En iyi parametreler ve skor zaten gösterildi, tekrar göstermeye gerek yok

# best_model zaten yukarıda oluşturuldu

# Test seti üzerinde tahmin yap
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Model performansını değerlendir
print("\nModel Performansı:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Sınıf etiketlerini oluştur
class_names = ['0 (Negatif)', '1 (Pozitif)']

# Matris değerlerini hazırla (TN, FP, FN, TP)
tn = cm[0, 0]  # True Negative
fp = cm[0, 1]  # False Positive
fn = cm[1, 0]  # False Negative
tp = cm[1, 1]  # True Positive

# Confusion matrix görseli
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)

# Etiketleri manuel olarak ekle
plt.text(0.5, 0.5, f'TN: {tn}', ha='center', va='center', fontsize=14, color='black', fontweight='normal')
plt.text(1.5, 0.5, f'FP: {fp}', ha='center', va='center', fontsize=14, color='black', fontweight='normal')
plt.text(0.5, 1.5, f'FN: {fn}', ha='center', va='center', fontsize=14, color='black', fontweight='normal')
plt.text(1.5, 1.5, f'TP: {tp}', ha='center', va='center', fontsize=14, color='black', fontweight='normal')



# Görsel ayarları
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin Edilen Değer')
plt.tight_layout()
plt.savefig('random_forest_confusion_matrix.png')
plt.close()

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('random_forest_roc_curve.png')
plt.close()

# Özellik önemini hesapla
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# En önemli 20 özelliği görselleştir
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Random Forest - Özellik Önemi (Top 20)')
plt.tight_layout()
plt.savefig('random_forest_feature_importance.png')
plt.close()

print("\nEn önemli 10 özellik:")
print(feature_importance.head(10))

# Model yorumlanabilirliği için özellik önemleri yeterli
print("Model yorumlanabilirliği için özellik önemleri kullanılıyor...")

# Model performansını özetle
print("\nModel performansı özeti:")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Özellik önemlerini göster
print("\nEn önemli 10 özellik:")
for i, row in feature_importance.head(10).iterrows():
    print(f"{row['Feature']}: {row['Importance']:.6f}")

print("\nRandom Forest modeli eğitimi ve değerlendirmesi tamamlandı.")
print("Kaydedilen görseller:")
print("- random_forest_confusion_matrix.png")
print("- random_forest_roc_curve.png")
print("- random_forest_feature_importance.png")

# Sonuçlar yukarıda kaydedildi
