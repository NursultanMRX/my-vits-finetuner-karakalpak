# MMS-TTS Finetuning for Karakalpak Language (Qoraqalpoq tili)

Ushbu repozitoriy Meta kompaniyasining **MMS (Massively Multilingual Speech)** modelini **Qoraqalpoq tili (Karakalpak)** uchun finetune (qayta o'qitish) qilishga mo'ljallangan. Loyiha VITS arxitekturasiga asoslangan bo'lib, o'qitish jarayonini barqarorlashtirish uchun barcha zarur tuzatishlar kiritilgan.

## 🛠 Talablar (Requirements)

Ushbu kod xatosiz ishlashi uchun quyidagi tizim va Python kutubxonalari aniq versiyalarda o'rnatilishi shart.

### 1. Tizim kutubxonalari (System Dependencies)
Audio fayllarni qayta ishlash (`ffmpeg`) va katta hajmli datasetlarni yuklash (`git-lfs`) uchun quyidagilarni o'rnating:

```bash
sudo apt update
sudo apt install ffmpeg git-lfs -y
git lfs install
```

### 2. Python kutubxonalari
Tavsiya etilgan Python versiyasi: **3.10**.
Juda muhim: `numpy` va `datasets` versiyalari quyidagicha bo'lishi kerak, aks holda audio formatlar bilan ziddiyat kelib chiqadi.

```bash
# Asosiy kutubxonalar
pip install -r requirements.txt

# Versiya ziddiyatlarini oldini olish uchun maxsus o'rnatish:
pip install "numpy<2.0" "datasets==2.19.1" soundfile librosa pandas accelerate transformers
```

---

## 🚀 Ishga tushirish bosqichlari

O'qitishni boshlash uchun quyidagi qadamlarni ketma-ket bajaring.

### 1-qadam: Asosiy modelni tayyorlash

Bizga Meta-ning asl modeli (checkpoint) kerak. Uni yuklab olib, VITS finetuning uchun moslashtirishimiz lozim.

1.  **Model vaznlarini yuklab oling:**
    ```bash
    wget https://huggingface.co/facebook/mms-tts-kaa/resolve/main/pytorch_model.bin
    ```

2.  **Modelni konvertatsiya qiling:**
    (Bu jarayon modelga discriminator qo'shadi va uni o'qitishga tayyorlaydi).
    ```bash
    python3 convert_original_discriminator_checkpoint.py \
        --checkpoint_path ./pytorch_model.bin \
        --generator_checkpoint_path facebook/mms-tts-kaa \
        --pytorch_dump_folder_path ./mms-tts-kaa-with-discriminator
    ```

### 2-qadam: Datasetni tayyorlash (Lokal usul)

HuggingFace'dan to'g'ridan-to'g'ri o'qishda yo'llar (path) bilan bog'liq xatoliklar chiqmasligi uchun, datasetni lokal kompyuterga yuklab olamiz.

1.  **Datasetni loyiha ichiga klon qiling:**
    ```bash
    git clone https://huggingface.co/datasets/nickoo004/karakalpak-tts-speaker1 ./my_local_dataset
    ```

2.  **CSV faylini to'g'irlash:**
    Dataset ichidagi `metadata.csv` faylida audio yo'llari nisbiy bo'ladi. Dastur ularni topishi uchun to'liq (absolute) manzilga o'tkazish kerak. Buning uchun `fix_csv.py` skriptini ishga tushiring:

    ```bash
    python3 fix_csv.py
    ```
    *(Agar bu fayl sizda bo'lmasa, uni quyidagi "Yordamchi fayllar" bo'limidan nusxalab oling).*

### 3-qadam: Konfiguratsiya (JSON)

`finetune_karakalpak.json` faylidagi sozlamalar to'g'ri ekanligini tekshiring. Eng muhim qismlari:

```json
{
    "model_name_or_path": "./mms-tts-kaa-with-discriminator",
    "dataset_name": "./my_local_dataset",
    "audio_column_name": "file_name",
    "text_column_name": "text",
    "speaker_id_column_name": "speaker_name",
    "filter_on_speaker_id": "Speaker_1",
    "min_duration_in_seconds": 0.5,
    "max_duration_in_seconds": 20.0,
    ...
}
```

### 4-qadam: O'qitishni boshlash (Training)

Barcha tayyorgarliklar tugagach, finetuning jarayonini quyidagi buyruq bilan boshlang:

```bash
accelerate launch run_vits_finetuning.py ./finetune_karakalpak.json
```

*Eslatma: Agar jarayon boshida `WandB` ro'yxatdan o'tishni so'rasa va sizga grafiklar shart bo'lmasa, **3** (Don't visualize) opsiyasini tanlang.*

---

## 📂 Yordamchi fayllar

### `fix_csv.py` kodi
Bu skript dataset ichidagi `metadata.csv` faylida audio fayl manzillarini to'liq (absolute path) manzilga aylantirib beradi.

```python
import pandas as pd
import os

# Dataset papkasi manzili
dataset_folder = "./my_local_dataset"
csv_path = os.path.join(dataset_folder, "metadata.csv")

if os.path.exists(csv_path):
    print("CSV fayl o'qilmoqda...")
    df = pd.read_csv(csv_path)
    abs_folder_path = os.path.abspath(dataset_folder)

    def make_absolute(path):
        # Agar yo'l allaqachon to'liq bo'lsa, tegmaymiz
        if path.startswith("/"): return path
        return os.path.join(abs_folder_path, path)

    # Ustun nomini tekshirish (file_name yoki audio)
    if 'file_name' in df.columns:
        df['file_name'] = df['file_name'].apply(make_absolute)
        df.to_csv(csv_path, index=False)
        print(f"Muvaffaqiyatli! CSV yangilandi. Manzil: {abs_folder_path}")
    else:
        print("Xatolik: 'file_name' ustuni topilmadi. Metadata.csv ustunlarini tekshiring.")
else:
    print(f"Xatolik: {csv_path} topilmadi.")
```

---

## ❓ Muammolar va Yechimlar (Troubleshooting)

Ushbu loyihani ishga tushirishda duch kelinishi mumkin bo'lgan eng keng tarqalgan xatolar:

| Xatolik | Sababi | Yechimi |
| :--- | :--- | :--- |
| **`ImportError: ... torchcodec`** | FFmpeg o'rnatilmagan yoki `datasets` yangi versiyasi `torchcodec`ni talab qilmoqda (lekin u ishlamayapti). | `pip uninstall torchcodec` <br> `sudo apt install ffmpeg` <br> `pip install "datasets==2.19.1"` |
| **`ValueError: num_samples=0`** | Dastur audio fayllarni topa olmayapti yoki barcha fayllar filtrdan o'ta olmadi. | `dataset_name` to'g'riligini tekshiring. <br> `min_duration_in_seconds`ni 0.5 ga tushiring. <br> `fix_csv.py` ni ishlatganingizga ishonch hosil qiling. |
| **`ValueError: Unable to avoid copy...`** | `numpy` versiyasi 2.0 dan yuqori bo'lib ketgan. | `pip install "numpy<2.0"` buyrug'ini bering. |
| **`TypeError: 'NoneType' object is not subscriptable`** | Audio fayl topilmadi yoki bo'sh. | `run_vits_finetuning.py` ichida `is_audio_in_length_range` funksiyasiga `if length is None: return False` tekshiruvini qo'shing (Repo ichidagi fayl allaqachon to'g'irlangan). |

---
```