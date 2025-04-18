import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter


class BoneCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_data_from_excel(excel_path, base_dir):
    """
    Birden fazla sheet içeren Excel dosyasından verileri yükle (her hasta için bir sheet)

    Args:
        excel_path: Excel dosyasının yolu
        base_dir: Image path için temel dizin

    Returns:
        image_paths: Image path listesi
        labels: Label'ların listesi
        label_mapping: String label'lardan integer'lara eşleme
    """
    image_paths = []
    labels = []

    # Tüm sheet'leri ile Excel dosyasını yükle
    print(f"Excel dosyası açılıyor: {excel_path}")
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names
    print(f"Bulunan sheet sayısı: {len(sheet_names)}, sheet'ler: {sheet_names}")

    # Label mapping dictionary
    label_mapping = {'SAĞLIKLI': 0, 'BENIGN': 1, 'MALIGN': 2}

    # Base directory içinde hangi hasta klasörlerinin var olduğunu kontrol et
    existing_folders = []
    for i in range(1, 10):  # En fazla 9 hasta olduğunu varsay
        folder_name = f'hasta{i}_png'
        if os.path.isdir(os.path.join(base_dir, folder_name)):
            existing_folders.append((i, folder_name))

    print(f"Bulunan hasta klasörleri: {[f[1] for f in existing_folders]}")

    if not existing_folders:
        print(f"HATA: {base_dir} içinde 'hastaN_png' klasörü bulunamadı")
        return [], [], label_mapping

    # Her sheet'i (hasta) işle
    for i, sheet in enumerate(sheet_names):
        # Eğer klasör sayısından fazla sheet varsa, işlemeyi durdur
        if i >= len(existing_folders):
            print(f"UYARI: Hasta klasörlerinden daha fazla sheet var. Sheet {sheet} atlanıyor")
            continue

        # Sheet adından çıkarmak yerine klasör indeksini kullan
        patient_id, folder_name = existing_folders[i]

        print(f"Sheet işleniyor: {sheet} -> Kullanılan klasör: {folder_name}")

        # Sheet verilerini oku
        df = pd.read_excel(excel_path, sheet_name=sheet)
        print(f"Sheet sütunları: {df.columns.tolist()}")
        print(f"Sheet boyutu: {df.shape}")

        # Sheet'teki her satırı işle
        for _, row in df.iterrows():
            try:
                # Görüntü ve label sütunlarını bulmaya çalış
                image_col = None
                label_col = None

                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'image' in col_lower or 'img' in col_lower or 'file' in col_lower:
                        image_col = col
                    elif 'label' in col_lower or 'class' in col_lower or 'etik' in col_lower:
                        label_col = col

                # Sütunlar bulunamazsa, ilk sütunun görüntü, ikincinin label olduğunu varsay
                if image_col is None and len(df.columns) > 0:
                    image_col = df.columns[0]
                if label_col is None and len(df.columns) > 1:
                    label_col = df.columns[1]

                print(f"Kullanılan sütunlar: Image={image_col}, Label={label_col}")

                # Görüntü adı ve label'ı al
                image_name = str(row[image_col])
                label = str(row[label_col])

                # Görüntü adının zaten bir uzantısı var mı kontrol et, yoksa .png ekle
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    image_name = f"{image_name}.png"

                # Tam img path oluştur
                img_path = os.path.join(base_dir, folder_name, image_name)

                # Görüntünün var olup olmadığını kontrol et
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    if label in label_mapping:
                        labels.append(label_mapping[label])
                    else:
                        print(f"Uyarı: Bilinmeyen label '{label}' - atlanıyor")
                else:
                    print(f"Uyarı: Görüntü şu yolda bulunamadı: {img_path}")

                # Konsolu aşırı mesajlarla doldurmamak için sadece birkaç görüntü için uyarı yazdır
                if len(image_paths) == 0 and _ > 10:
                    print("Çok fazla eksik görüntü var, daha fazla uyarı gösterilmiyor...")
                    break

            except Exception as e:
                print(f"{sheet} sheet'inde satır işlenirken hata oluştu: {e}")

    # Class dağılımını yazdır
    if labels:
        class_counts = Counter(labels)
        print(f"Class dağılımı:")
        for label_name, label_idx in label_mapping.items():
            count = class_counts.get(label_idx, 0)
            percentage = (count / len(labels)) * 100 if len(labels) > 0 else 0
            print(f"  {label_name}: {count} ({percentage:.2f}%)")

    return image_paths, labels, label_mapping


def create_data_loaders(image_paths, labels, batch_size=32, test_size=0.2, random_state=42):
    """
    Training ve validation için DataLoader'lar oluştur

    Args:
        image_paths: Image path listesi
        labels: Label'ların listesi
        batch_size: DataLoader için batch size
        test_size: Validation için kullanılacak veri oranı
        random_state: Tekrarlanabilirlik için random seed

    Returns:
        train_loader: Training için DataLoader
        val_loader: Validation için DataLoader
        class_weights: Her class için weight'ler (class imbalance için)
    """
    # Transformation'ları tanımla
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Veriyi training ve validation set'lerine böl
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Dataset'leri oluştur
    train_dataset = BoneCancerDataset(X_train, y_train, transform=train_transform)
    val_dataset = BoneCancerDataset(X_val, y_val, transform=val_transform)

    # Class imbalance için class weight'lerini hesapla
    class_counts = Counter(y_train)
    num_samples = len(y_train)
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = num_samples / (len(class_counts) * count)

    # Tüm class'ların temsil edildiğinden emin ol
    weights = torch.FloatTensor([class_weights.get(i, 1.0) for i in range(len(class_counts))])

    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Training set boyutu: {len(train_dataset)}")
    print(f"Validation set boyutu: {len(val_dataset)}")
    print(f"Class weight'leri: {class_weights}")

    return train_loader, val_loader, weights


def visualize_sample_images(image_paths, labels, label_mapping, num_samples=5):
    """
    Her class'tan örnek görüntüleri görselleştir

    Args:
        image_paths: Imgage path listesi
        labels: Integer label'ların listesi
        label_mapping: String label'lardan integer'lara eşleme
        num_samples: Her class'tan görselleştirilecek örnek sayısı
    """
    # Görüntüleme için label mapping'i tersine çevir
    label_names = {v: k for k, v in label_mapping.items()}

    # Image path'leri class'a göre grupla
    class_images = {}
    for path, label in zip(image_paths, labels):
        if label not in class_images:
            class_images[label] = []
        class_images[label].append(path)

    # Her class'tan örnek görüntüleri plot et
    plt.figure(figsize=(15, 5 * len(class_images)))

    for i, (label, paths) in enumerate(class_images.items()):
        samples = paths[:num_samples]
        for j, img_path in enumerate(samples):
            plt.subplot(len(class_images), num_samples, i * num_samples + j + 1)
            img = Image.open(img_path)
            plt.imshow(img, cmap='gray')
            plt.title(f"{label_names[label]}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()