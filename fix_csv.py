import csv

input_filename = 'metadata.csv'
output_filename = 'metadata_fixed.csv'

with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
     open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
    
    # '|' belgisi bilan ajratilgan ma'lumotni o'qish
    reader = csv.reader(infile, delimiter='|')
    
    # ',' belgisi bilan ajratib, barcha ustunlarni qo'shtirnoqqa olib yozish
    writer = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_ALL)
    
    # Har bir qatorni o'qib, yangi formatda yozish
    for row in reader:
        writer.writerow(row)

# Eski faylni yangisi bilan almashtirish
import os
os.rename(output_filename, input_filename)

print(f"'{input_filename}' fayli muvaffaqiyatli tarzda to'g'ri CSV formatiga o'tkazildi!")