import os
import io
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# НАСТРОЙКИ
# =========================
ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"

# Подобранные параметры
K_FOR_3x3 = -0.3
K_FOR_25x25 = -0.15

RAW_DIR = "raw_images"
GRAY_DIR = "gray_bmp"
BIN_DIR = "binary_niblack"
DEMO_DIR = "demo"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(GRAY_DIR, exist_ok=True)
os.makedirs(BIN_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)


# =========================
# ПОЛУЧЕНИЕ СПИСКА ИЗОБРАЖЕНИЙ
# =========================
def get_image_urls(origin: str, sample_id: str):
    url = f"{origin}/api/samples/{sample_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    sample_data = response.json()
    return [f"{origin}/images/{page['filename']}" for page in sample_data["pages"]]


# =========================
# ЗАГРУЗКА ИЗОБРАЖЕНИЯ
# =========================
def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


# =========================
# ПЕРЕВОД В ПОЛУТОН
# Без библиотечной функции convert('L')
# Y = 0.299R + 0.587G + 0.114B
# =========================
def rgb_to_grayscale_manual(img: Image.Image) -> Image.Image:
    rgb_img = img.convert("RGB")
    arr = np.array(rgb_img, dtype=np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    return Image.fromarray(gray, mode="L")


# =========================
# АДАПТИВНАЯ БИНАРИЗАЦИЯ НИБЛЭКА
# T = m + k * s
# =========================
def niblack_binarization(gray_img: Image.Image, window_size: int, k: float) -> Image.Image:
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("window_size должен быть нечётным и >= 3")

    gray = np.array(gray_img, dtype=np.float32)
    h, w = gray.shape

    pad = window_size // 2
    padded = np.pad(gray, pad_width=pad, mode="edge")

    binary = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            window = padded[y:y + window_size, x:x + window_size]
            m = np.mean(window)
            s = np.std(window)
            t = m + k * s

            binary[y, x] = 255 if gray[y, x] > t else 0

    return Image.fromarray(binary, mode="L")


# =========================
# СОХРАНЕНИЕ СРАВНЕНИЯ 2 КАРТИНОК
# =========================
def save_comparison(original: Image.Image, processed: Image.Image,
                    title_left: str, title_right: str, out_path: str):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    if original.mode == "L":
        plt.imshow(original, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        plt.imshow(original, interpolation="nearest")
    plt.title(title_left)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title(title_right)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# СОХРАНЕНИЕ СРАВНЕНИЯ 3 КАРТИНОК
# =========================
def save_triple_comparison(img1: Image.Image, img2: Image.Image, img3: Image.Image,
                           title1: str, title2: str, title3: str, out_path: str):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title(title2)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img3, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title(title3)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# ОСНОВНАЯ ФУНКЦИЯ
# max_images=None -> обработать все изображения
# =========================
def process_images(max_images=None):
    image_urls = get_image_urls(ORIGIN, SAMPLE_ID)

    if max_images is not None:
        image_urls = image_urls[:max_images]

    print(f"Изображений для обработки: {len(image_urls)}")

    for i, url in enumerate(image_urls, start=1):
        print(f"[{i}/{len(image_urls)}] Загружается: {url}")

        try:
            img = download_image(url)
            base_name = f"image_{i:02d}"

            # Исходное изображение
            raw_path = os.path.join(RAW_DIR, f"{base_name}.png")
            img.convert("RGB").save(raw_path, format="PNG")

            # Полутоновое изображение
            gray_img = rgb_to_grayscale_manual(img)
            gray_path = os.path.join(GRAY_DIR, f"{base_name}_gray.bmp")
            gray_img.save(gray_path, format="BMP")

            demo_gray_path = os.path.join(DEMO_DIR, f"{base_name}_gray_demo.png")
            save_comparison(
                original=img.convert("RGB"),
                processed=gray_img,
                title_left="Исходное изображение",
                title_right="Полутоновое изображение",
                out_path=demo_gray_path
            )

            # Ниблэк 3x3
            binary_3 = niblack_binarization(gray_img, window_size=3, k=K_FOR_3x3)
            binary_3_path = os.path.join(BIN_DIR, f"{base_name}_niblack_3x3.png")
            binary_3.save(binary_3_path, format="PNG")

            demo_bin_3_path = os.path.join(DEMO_DIR, f"{base_name}_binary_3x3_demo.png")
            save_comparison(
                original=gray_img,
                processed=binary_3,
                title_left="Полутоновое изображение",
                title_right=f"Ниблэк 3x3, k={K_FOR_3x3}",
                out_path=demo_bin_3_path
            )

            # Ниблэк 25x25
            binary_25 = niblack_binarization(gray_img, window_size=25, k=K_FOR_25x25)
            binary_25_path = os.path.join(BIN_DIR, f"{base_name}_niblack_25x25.png")
            binary_25.save(binary_25_path, format="PNG")

            demo_bin_25_path = os.path.join(DEMO_DIR, f"{base_name}_binary_25x25_demo.png")
            save_comparison(
                original=gray_img,
                processed=binary_25,
                title_left="Полутоновое изображение",
                title_right=f"Ниблэк 25x25, k={K_FOR_25x25}",
                out_path=demo_bin_25_path
            )

            # Общее сравнение
            triple_demo_path = os.path.join(DEMO_DIR, f"{base_name}_triple_comparison.png")
            save_triple_comparison(
                gray_img,
                binary_3,
                binary_25,
                "Полутоновое изображение",
                f"Ниблэк 3x3, k={K_FOR_3x3}",
                f"Ниблэк 25x25, k={K_FOR_25x25}",
                triple_demo_path
            )

            print("  Уникальные значения:")
            print(f"    3x3   -> {np.unique(np.array(binary_3))}")
            print(f"    25x25 -> {np.unique(np.array(binary_25))}")

            print("  Сохранено:")
            print(f"    исходник:        {raw_path}")
            print(f"    полутон BMP:     {gray_path}")
            print(f"    бинарное 3x3:    {binary_3_path}")
            print(f"    бинарное 25x25:  {binary_25_path}")
            print(f"    demo gray:       {demo_gray_path}")
            print(f"    demo 3x3:        {demo_bin_3_path}")
            print(f"    demo 25x25:      {demo_bin_25_path}")
            print(f"    demo triple:     {triple_demo_path}")

        except Exception as e:
            print(f"Ошибка при обработке {url}: {e}")


if __name__ == "__main__":
    process_images(max_images=10)  
    # process_images()             # обработать все изображения
    print("\nГотово.")