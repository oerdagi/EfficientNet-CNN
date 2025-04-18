import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, accuracy_score
from data_preparation import load_data_from_excel, BoneCancerDataset
from model_implementation import EfficientNetModel


def predict_all_images(model, image_paths, device, batch_size=32):
    """
    Tüm görüntüler için prediction yap

    Args:
        model: Eğitilmiş model
        image_paths: Görüntü yollarının listesi
        device: Değerlendirme yapılacak device (cuda veya cpu)
        batch_size: İşleme için batch size

    Returns:
        predictions: Tahmin edilen class indekslerinin listesi
        probabilities: Her class için prediction olasılıklarının listesi
    """
    # Transform tanımla
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Model'i evaluation moduna ayarla
    model.eval()

    # Prediction ve probability'ler için listeleri başlat
    predictions = []
    probabilities = []

    # Görüntüleri batch'ler halinde işle
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        # Görüntüleri yükle ve preprocess et
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                batch_images.append(img_tensor)
            except Exception as e:
                print(f"Görüntü yüklenirken hata oluştu {img_path}: {e}")
                # Başarısız görüntüler için placeholder ekle
                batch_images.append(torch.zeros((3, 224, 224)))

        # Tensor'ları stack yap
        batch_tensor = torch.stack(batch_images).to(device)

        # Prediction'ları yap
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

        # Batch sonuçlarını ekle
        predictions.extend(preds.cpu().numpy())
        probabilities.extend(probs.cpu().numpy())

    return np.array(predictions), np.array(probabilities)


def compare_doctor_vs_cnn(doctor_labels, cnn_predictions, label_mapping):
    """
    Doktor tanılarını CNN prediction'ları ile karşılaştır

    Args:
        doctor_labels: Doktorun label'larının listesi (integer indeksleri)
        cnn_predictions: CNN'in prediction'larının listesi (integer indeksleri)
        label_mapping: String label'lardan integer'lara eşleme

    Returns:
        comparison_df: Karşılaştırma sonuçlarıyla DataFrame
        metrics: Karşılaştırma metric'lerinin dictionary'si
    """
    # Girdilerin numpy array olduğundan emin ol
    doctor_labels = np.array(doctor_labels)
    cnn_predictions = np.array(cnn_predictions)

    # Görüntüleme için label mapping'i tersine çevir
    label_names = {v: k for k, v in label_mapping.items()}

    # Karşılaştırma DataFrame'ini oluştur
    comparison_df = pd.DataFrame({
        'Doctor': [label_names.get(label, 'Unknown') for label in doctor_labels],
        'CNN': [label_names.get(pred, 'Unknown') for pred in cnn_predictions],
        'Agreement': doctor_labels == cnn_predictions
    })

    # Metric'leri hesapla
    overall_accuracy = accuracy_score(doctor_labels, cnn_predictions)
    kappa = cohen_kappa_score(doctor_labels, cnn_predictions)

    # Class'a özgü agreement hesapla
    class_agreement = {}

    # Doktor label'larındaki unique class'ları al
    unique_classes = np.unique(doctor_labels)

    for class_idx in unique_classes:
        class_name = label_names.get(class_idx, f"Class {class_idx}")

        # Doktorun label'ı mevcut class olan index'leri al (güvenli bir şekilde)
        class_mask = (doctor_labels == class_idx)
        class_indices = np.where(class_mask)[0]

        if len(class_indices) > 0:
            # Bu class için agreement hesapla
            class_acc = accuracy_score(
                doctor_labels[class_indices],
                cnn_predictions[class_indices]
            )
            class_agreement[class_name] = {
                'count': len(class_indices),
                'agreement_rate': class_acc,
                'correct_count': int(class_acc * len(class_indices))
            }
        else:
            print(f"Uyarı: {class_name} class'ı için örnek bulunamadı")

    # Label mapping'de bulunan ancak verilerde olmayan class'ların durumunu ele al
    for class_idx, class_name in label_names.items():
        if class_name not in class_agreement and class_idx in label_mapping.values():
            class_agreement[class_name] = {
                'count': 0,
                'agreement_rate': 0.0,
                'correct_count': 0
            }

    # Metric'leri birleştir
    metrics = {
        'overall_accuracy': overall_accuracy,
        'kappa': kappa,
        'class_agreement': class_agreement
    }

    return comparison_df, metrics


def plot_comparison_results(comparison_df, metrics, label_mapping):
    """
    Karşılaştırma sonuçlarını plot et

    Args:
        comparison_df: Karşılaştırma sonuçlarıyla DataFrame
        metrics: Karşılaştırma metric'lerinin dictionary'si
        label_mapping: String label'lardan integer'lara eşleme
    """
    # Görüntüleme için label mapping'i tersine çevir
    label_names = {v: k for k, v in label_mapping.items()}

    # Figure oluştur
    plt.figure(figsize=(15, 10))

    # Class bazında agreement'ı plot et
    plt.subplot(2, 2, 1)
    agreement_by_class = comparison_df.groupby('Doctor')['Agreement'].mean() * 100
    agreement_by_class.plot(kind='bar', color='skyblue')
    plt.title('Class Bazında Agreement Oranı (%)')
    plt.ylabel('Agreement Oranı (%)')
    plt.xlabel('Doktor Tanısı')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)

    # Doktor ve CNN arasındaki confusion matrix'i plot et
    plt.subplot(2, 2, 2)
    conf_matrix = pd.crosstab(
        comparison_df['Doctor'],
        comparison_df['CNN'],
        rownames=['Doctor'],
        colnames=['CNN']
    )
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title('Doktor vs CNN Confusion Matrix')

    # Agreement sayılarını plot et
    plt.subplot(2, 2, 3)
    class_counts = []
    class_correct = []
    class_names = []

    for class_name, data in metrics['class_agreement'].items():
        class_names.append(class_name)
        class_counts.append(data['count'])
        class_correct.append(data['correct_count'])

    x = np.arange(len(class_names))
    width = 0.35

    plt.bar(x - width / 2, class_counts, width, label='Toplam Vaka')
    plt.bar(x + width / 2, class_correct, width, label='Agreement Vakası')
    plt.xticks(x, class_names, rotation=45)
    plt.title('Class Bazında Vaka Sayıları ve Agreement\'lar')
    plt.xlabel('Class')
    plt.ylabel('Sayı')
    plt.legend()

    # Genel metric'leri plot et
    plt.subplot(2, 2, 4)
    overall_metrics = {
        'Genel Accuracy': metrics['overall_accuracy'] * 100,
        'Cohen\'s Kappa': metrics['kappa'] * 100
    }

    plt.bar(overall_metrics.keys(), overall_metrics.values(), color=['green', 'purple'])
    plt.title('Genel Performans Metric\'leri')
    plt.ylabel('Skor (%)')
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.show()


def save_comparison_results(comparison_df, metrics, output_file='doctor_vs_cnn_comparison.xlsx'):
    """
    Karşılaştırma sonuçlarını Excel dosyasına kaydet

    Args:
        comparison_df: Karşılaştırma sonuçlarıyla DataFrame
        metrics: Karşılaştırma metric'lerinin dictionary'si
        output_file: Çıktı Excel dosyasının path'i
    """
    # Excel writer oluştur
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Karşılaştırma DataFrame'ini kaydet
        comparison_df.to_excel(writer, sheet_name='Image Comparison', index=True)

        # Metric'ler DataFrame'i oluştur
        metrics_data = {
            'Metric': ['Genel Accuracy', 'Cohen\'s Kappa'],
            'Value': [metrics['overall_accuracy'], metrics['kappa']]
        }

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(writer, sheet_name='Overall Metrics', index=False)

        # Class agreement DataFrame'i oluştur
        class_agreement_data = []
        for class_name, data in metrics['class_agreement'].items():
            class_agreement_data.append({
                'Class': class_name,
                'Total Cases': data['count'],
                'Agreement Rate': data['agreement_rate'],
                'Correct Count': data['correct_count']
            })

        class_agreement_df = pd.DataFrame(class_agreement_data)
        class_agreement_df.to_excel(writer, sheet_name='Class Agreement', index=False)

    print(f"Karşılaştırma sonuçları {output_file} dosyasına kaydedildi")