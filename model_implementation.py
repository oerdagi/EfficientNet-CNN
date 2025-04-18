import torch
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=3, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # Son katmandaki feature sayısını al
        if hasattr(self.model, 'classifier'):
            if hasattr(self.model.classifier, 'in_features'):
                n_features = self.model.classifier.in_features
            else:
                # Classifier'ın Sequential veya başka bir modül olduğu modeller için
                n_features = self.model.classifier[0].in_features if isinstance(self.model.classifier,
                                                                                nn.Sequential) else 512

            # Classifier'ı değiştir
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(n_features, num_classes)
            )
        else:
            # Classifier yerine fc kullanan modeller için
            if hasattr(self.model, 'fc'):
                n_features = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(n_features, num_classes)
                )
            else:
                # Diğer mimariler için fallback
                print("Uyarı: classifier veya fc katmanı bulunamadı. Varsayılan head kullanılıyor.")
                self.model = nn.Sequential(
                    self.model,
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(0.2),
                    nn.Linear(512, num_classes)  # Varsayılan olarak 512 kullanılıyor
                )

    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, val_loader, device, class_weights=None,
                num_epochs=20, learning_rate=0.001, save_path='best_model.pth'):
    """
    EfficientNet model'ini eğit

    Args:
        model: Eğitilecek model
        train_loader: Training verisi için DataLoader
        val_loader: Validation verisi için DataLoader
        device: Eğitimin yapılacağı device (cuda veya cpu)
        class_weights: Her class için weight'ler (class imbalance için)
        num_epochs: Eğitim için epoch sayısı
        learning_rate: Optimizer için learning rate
        save_path: En iyi model'in kaydedileceği path

    Returns:
        model: Eğitilmiş model
        history: Eğitim history'si
    """
    # Model'i device'a taşı
    model = model.to(device)

    # Eğer class weight'leri sağlanmışsa loss function'ı tanımla
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer'ı tanımla
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Değişkenleri başlat
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # Eğitim döngüsü
    for epoch in range(num_epochs):
        # Training aşaması
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)

            # Gradient parametrelerini sıfırla
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass ve optimize et
            loss.backward()
            optimizer.step()

            # İstatistikleri takip et
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).type(torch.float).sum().item()

            # Progress bar'ı güncelle
            train_loop.set_postfix(loss=loss.item(), acc=train_correct / train_total)

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        # Validation aşaması
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # İstatistikleri takip et
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).type(torch.float).sum().item()

                # Progress bar'ı güncelle
                val_loop.set_postfix(loss=loss.item(), acc=val_correct / val_total)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        # Learning rate scheduler'ı güncelle
        scheduler.step(val_loss)

        # Epoch istatistiklerini yazdır
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Eğer şimdiye kadarki en iyi model ise kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Model checkpoint şuraya kaydedildi: {save_path}')

        # History'yi güncelle
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    # En iyi model'i yükle
    model.load_state_dict(torch.load(save_path))

    return model, history


def evaluate_model(model, data_loader, device, label_mapping):
    """
    Model'i değerlendir ve prediction'ları metric'lerle birlikte döndür

    Args:
        model: Eğitilmiş model
        data_loader: Değerlendirme için DataLoader
        device: Değerlendirmenin yapılacağı device (cuda veya cpu)
        label_mapping: String label'lardan integer'lara eşleme

    Returns:
        y_true: Gerçek label'lar
        y_pred: Tahmin edilen label'lar
        metrics: Değerlendirme metric'lerinin dictionary'si
    """
    # Görüntüleme için label mapping'i tersine çevir
    label_names = {v: k for k, v in label_mapping.items()}

    # Model'i evaluation moduna ayarla
    model.eval()

    # Gerçek ve tahmin edilen label'ları depolamak için listeleri başlat
    y_true = []
    y_pred = []

    # Model'i değerlendir
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Batch prediction'larını ve label'larını ekle
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Listeleri numpy array'lerine dönüştür
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metric'leri hesapla
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Class'a özgü metric'leri hesapla
    class_metrics = {}
    for class_idx in range(len(label_names)):
        class_name = label_names[class_idx]
        class_metrics[class_name] = {
            'precision': precision[class_idx],
            'recall': recall[class_idx],
            'f1': f1[class_idx]
        }

    # Genel metric'leri hesapla
    overall_metrics = {
        'accuracy': accuracy,
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1)
    }

    # Metric'leri birleştir
    metrics = {
        'class_metrics': class_metrics,
        'overall_metrics': overall_metrics,
        'confusion_matrix': conf_matrix
    }

    return y_true, y_pred, metrics


def plot_training_history(history):
    """
    Eğitim history'sini plot et

    Args:
        history: Eğitim history dictionary'si
    """
    # Figure oluştur
    plt.figure(figsize=(12, 5))

    # Loss'u plot et
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy'yi plot et
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(conf_matrix, label_mapping):
    """
    Confusion matrix'i plot et

    Args:
        conf_matrix: Confusion matrix
        label_mapping: String label'lardan integer'lara eşleme
    """
    # Görüntüleme için label mapping'i tersine çevir
    label_names = {v: k for k, v in label_mapping.items()}
    class_names = [label_names[i] for i in range(len(label_names))]

    # Figure oluştur
    plt.figure(figsize=(10, 8))

    # Confusion matrix'i plot et
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.show()


def print_metrics(metrics, label_mapping):
    """
    Değerlendirme metric'lerini yazdır

    Args:
        metrics: Değerlendirme metric'lerinin dictionary'si
        label_mapping: String label'lardan integer'lara eşleme
    """
    # Genel metric'leri yazdır
    print("Genel Metric'ler:")
    for metric, value in metrics['overall_metrics'].items():
        print(f"  {metric}: {value:.4f}")

    # Class'a özgü metric'leri yazdır
    print("\nClass'a Özgü Metric'ler:")
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"  {class_name}:")
        for metric, value in class_metrics.items():
            print(f"    {metric}: {value:.4f}")