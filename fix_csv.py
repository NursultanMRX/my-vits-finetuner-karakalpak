# import csv

# input_filename = 'metadata.csv'
# output_filename = 'metadata_fixed.csv'

# with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
#      open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
    
#     # '|' belgisi bilan ajratilgan ma'lumotni o'qish
#     reader = csv.reader(infile, delimiter='|')
    
#     # ',' belgisi bilan ajratib, barcha ustunlarni qo'shtirnoqqa olib yozish
#     writer = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_ALL)
    
#     # Har bir qatorni o'qib, yangi formatda yozish
#     for row in reader:
#         writer.writerow(row)

# # Eski faylni yangisi bilan almashtirish
# import os
# os.rename(output_filename, input_filename)

# print(f"'{input_filename}' fayli muvaffaqiyatli tarzda to'g'ri CSV formatiga o'tkazildi!")





import pandas as pd
import os

# Dataset papkasi va CSV fayl manzili
dataset_folder = "./my_local_dataset"
csv_path = os.path.join(dataset_folder, "metadata.csv")

# Fayl borligini tekshiramiz
if not os.path.exists(csv_path):
    print(f"Xatolik: {csv_path} fayli topilmadi!")
    exit()

# CSV ni o'qiymiz
print("CSV fayl o'qilmoqda...")
df = pd.read_csv(csv_path)

# Datasetning to'liq (absolute) manzilini olamiz
abs_folder_path = os.path.abspath(dataset_folder)

# Funksiya: Agar yo'l to'liq bo'lmasa, uni to'liq qilamiz
def make_absolute(path):
    # Agar yo'l allaqachon to'liq bo'lsa, tegmaymiz
    if path.startswith("/"):
        return path
    # Aks holda, oldiga papka manzilini qo'shamiz
    return os.path.join(abs_folder_path, path)

# 'file_name' ustunidagi barcha yo'llarni o'zgartiramiz
# Eslatma: Agar sizda ustun nomi 'audio' bo'lsa, pastdagi 'file_name' ni 'audio' ga o'zgartiring
column_name = 'file_name' 

if column_name in df.columns:
    df[column_name] = df[column_name].apply(make_absolute)
    # O'zgarishlarni saqlaymiz
    df.to_csv(csv_path, index=False)
    print(f"Muvaffaqiyatli! Barcha yo'llar to'liq manzilga o'zgartirildi: {abs_folder_path}/wavs/...")
    print(df.head())
else:
    print(f"Xatolik: CSV ichida '{column_name}' ustuni topilmadi. Ustun nomlarini tekshiring.")