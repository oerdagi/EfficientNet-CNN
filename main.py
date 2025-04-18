import os
import argparse
import torch
import random
import numpy as np

# Modüllerden fonksiyonları import et
from data_preparation import (
    load_data_from_excel, load_data_from_excel, create_data_loaders,
    visualize_sample_images
)
from model_implementation import (
    EfficientNetModel, train_model, evaluate_model,
    plot_training_history, plot_confusion_matrix, print_metrics
)
from doctor_comparison import (
    predict_all_images, compare_doctor_vs_cnn,
    plot_comparison_results, save_comparison_results
)


def set_seed(seed):
    """
    Tekrarlanabilirlik için seed değerini ayarla

    Args:
        seed: Seed değeri
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # Komut satırı argümanlarını parse et
    parser = argparse.ArgumentParser(description='Kemik Kanseri Tespiti')
    parser.add_argument('--data_dir', type=str, default='H1_Data', help='Veri dizininin yolu')
    parser.add_argument('--excel_file', type=str, default='etiketler.xlsx', help='Excel dosyasının adı')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Model kaydetme/yükleme yolu')
    parser.add_argument('--batch_size', type=int, default=16, help='Eğitim için batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Eğitim için epoch sayısı')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Rastgele seed')
    parser.add_argument('--train', action='store_true', help='Modeli eğit')
    parser.add_argument('--evaluate', action='store_true', help='Modeli değerlendir')
    parser.add_argument('--compare', action='store_true', help='Doktor kararları ile karşılaştır')
    parser.add_argument('--visualize', action='store_true', help='Örnek görüntüleri görselleştir')
    parser.add_argument('--output_file', type=str, default='doctor_vs_cnn_comparison.xlsx',
                        help='Karşılaştırma sonuçları için çıkış dosyası')

    args = parser.parse_args()

    # Tekrarlanabilirlik için seed değerini ayarla
    set_seed(args.seed)

    # Device ayarla
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Kullanılan device: {device}")

    # Excel dosyasından verileri yükle
    print("Veriler yükleniyor...")
    excel_path = os.path.join(args.data_dir, args.excel_file)

    if os.path.exists(excel_path):
        print(f"Excel dosyasından veriler yükleniyor: {excel_path}")
        try:
            image_paths, labels, label_mapping = load_data_from_excel(excel_path, args.data_dir)
            print(f"Excel'den 6 hasta sayfası ile {len(image_paths)} görüntü başarıyla yüklendi")
            print(f"Etiket eşleştirmesi: {label_mapping}")
        except Exception as e:
            print(f"Excel dosyası yüklenirken hata oluştu: {e}")
            raise ValueError(f"Excel dosyasından veri yüklenemedi: {excel_path}")
    else:
        print(f"Excel dosyası bulunamadı: {excel_path}")
        raise FileNotFoundError(f"Excel dosyası bulunamadı: {excel_path}")

    print(f"Etiketlerle birlikte {len(image_paths)} görüntü yüklendi")
    print(f"Etiket eşleştirmesi: {label_mapping}")

    # İstenirse örnek görüntüleri görselleştir
    if args.visualize:
        print("Örnek görüntüler görselleştiriliyor...")
        visualize_sample_images(image_paths, labels, label_mapping)

    # İstenirse modeli eğit
    if args.train:
        print("Data loader'lar oluşturuluyor...")
        train_loader, val_loader, class_weights = create_data_loaders(
            image_paths, labels, batch_size=args.batch_size
        )

        print("Model oluşturuluyor...")
        model = EfficientNetModel(num_classes=len(label_mapping), model_name='efficientnet_b0', pretrained=True)

        print("Model eğitiliyor...")
        trained_model, history = train_model(
            model, train_loader, val_loader, device,
            class_weights=class_weights, num_epochs=args.num_epochs,
            learning_rate=args.learning_rate, save_path=args.model_path
        )

        # Eğitim geçmişini grafikleştir
        plot_training_history(history)

    # Değerlendirme veya karşılaştırma için eğitilmiş modeli yükle
    if args.evaluate or args.compare:
        print("Model yükleniyor...")
        model = EfficientNetModel(num_classes=len(label_mapping), model_name='efficientnet_b0', pretrained=False)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)

    # İstenirse modeli değerlendir
    if args.evaluate:
        print("Değerlendirme için data loader'lar oluşturuluyor...")
        _, val_loader, _ = create_data_loaders(
            image_paths, labels, batch_size=args.batch_size
        )

        # Model tanımlandığından emin ol
        if 'model' not in locals() or model is None:
            print("Değerlendirme için model yükleniyor...")
            model = EfficientNetModel(num_classes=len(label_mapping), model_name='efficientnet_b0', pretrained=False)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model = model.to(device)

        print("Model değerlendiriliyor...")
        y_true, y_pred, metrics = evaluate_model(model, val_loader, device, label_mapping)

        # Metrikleri yazdır
        print_metrics(metrics, label_mapping)

        # Confusion matrix'i çiz
        plot_confusion_matrix(metrics['confusion_matrix'], label_mapping)

    # İstenirse doktor kararları ile karşılaştır
    if args.compare:
        print("Tüm görüntüler için tahminler yapılıyor...")
        predictions, probabilities = predict_all_images(model, image_paths, device, batch_size=args.batch_size)

        print("Doktor ve CNN karşılaştırılıyor...")
        comparison_df, metrics = compare_doctor_vs_cnn(labels, predictions, label_mapping)

        # Metrikleri yazdır
        print("\nGenel Metrikler:")
        print(f"  Doğruluk: {metrics['overall_accuracy']:.4f}")
        print(f"  Cohen's Kappa: {metrics['kappa']:.4f}")

        print("\nSınıfa Özgü Uyum:")
        for class_name, data in metrics['class_agreement'].items():
            print(f"  {class_name}:")
            print(f"    Toplam vaka: {data['count']}")
            print(f"    Uyum oranı: {data['agreement_rate']:.4f}")
            print(f"    Doğru sayı: {data['correct_count']}")

        # Karşılaştırma sonuçlarını çiz
        plot_comparison_results(comparison_df, metrics, label_mapping)

        # Karşılaştırma sonuçlarını kaydet
        save_comparison_results(comparison_df, metrics, args.output_file)


if __name__ == "__main__":
    main()