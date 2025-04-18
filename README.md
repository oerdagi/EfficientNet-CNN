# EfficientNet-CNN

H1_Data/hasta1_png
... H1_Data/hasta2_png
...

# Sample Görselleştirme
python3 main.py --data_dir H1_Data --excel_file etiketler.xlsx --visualize

# Train kodu
python3 main.py --data_dir H1_Data --excel_file etiketler.xlsx --train --num_epochs 15 --batch_size 16

# Modeli doğrulama
python3 main.py --data_dir H1_Data --excel_file etiketler.xlsx --evaluate

# Doktor kararları ile karşılaştırma
python3 main.py --data_dir H1_Data --excel_file etiketler.xlsx --compare --output_file doctor_vs_cnn_comparison.xlsx

# Hepsi
python3 main.py --data_dir H1_Data --excel_file etiketler.xlsx --train --evaluate --compare
