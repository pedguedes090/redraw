# 🎨 Redraw - Unified Image Inpainting Tool

Kết hợp **AOT-GAN** (xóa/vẽ lại ảnh tự nhiên) và **MangaInpainting** (xóa/vẽ lại truyện tranh manga) thành một công cụ duy nhất.

## Tổng quan

| Tính năng | AOT-GAN | MangaInpainting |
|---|---|---|
| **Loại ảnh** | Ảnh tự nhiên (photos, chân dung, phong cảnh) | Truyện tranh manga/comic (đen trắng) |
| **Ứng dụng** | Xóa vật thể, phục hồi ảnh | Xóa bóng thoại, text, hiệu ứng âm thanh |
| **Độ phân giải** | 512×512 | Tùy ý |
| **Input** | Ảnh RGB + Mask | Ảnh grayscale + Structural lines + Mask |

## Cài đặt

### 1. Tạo môi trường

```bash
# Tạo conda environment
conda create -n redraw python=3.8
conda activate redraw

# Cài đặt PyTorch (chọn CUDA phù hợp)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chuẩn bị Models

#### Tự động setup (nếu đã có file zip)
```bash
python setup_models.py
```

#### Tải thủ công

**AOT-GAN:**
- [CelebA-HQ model](https://drive.google.com/drive/folders/1Zks5Hyb9WAEpupbTdBqsCafmb25yqsGJ) - Cho ảnh chân dung
- [Places2 model](https://drive.google.com/drive/folders/1bSOH-2nB3feFRyDEmiX81CEiWkghss3i) - Cho ảnh phong cảnh/tổng quát
- Đặt file `.pt` vào `checkpoints/aot_gan/`

**MangaInpainting:**
- [MangaInpainting model](https://drive.google.com/file/d/1YeVwaNfchLhy3lAA7jOLBP-W23onjy8S/view) → Giải nén vào `checkpoints/mangainpaintor/`
- [ScreenVAE model](https://drive.google.com/file/d/1QaXqR4KWl_lxntSy32QpQpXb-1-EP7_L/view) → Giải nén vào `checkpoints/ScreenVAE/`

### 3. Cấu trúc thư mục cần có

```
redraw/
├── app.py                    ← Web UI (Gradio)
├── inpaint.py                ← CLI tool
├── setup_models.py           ← Script cài đặt models
├── requirements.txt
├── checkpoints/
│   ├── aot_gan/
│   │   └── G0000000.pt       ← AOT-GAN pre-trained model
│   ├── mangainpaintor/
│   │   ├── config.yml
│   │   ├── SemanticInpaintingModel_gen.pth
│   │   └── MangaInpaintingModel_gen.pth
│   └── ScreenVAE/
│       ├── latest_net_enc.pth
│       └── latest_net_dec.pth
├── aot_gan/                  ← AOT-GAN source code
└── manga_inpainting/         ← MangaInpainting source code
```

## Sử dụng

### Cách 1: Web UI (khuyên dùng)

```bash
python app.py
```

Mở trình duyệt tại `http://localhost:7860` và sử dụng giao diện kéo thả:

- **Tab "Natural Image Inpainting"**: Upload ảnh + mask hoặc vẽ mask trực tiếp
- **Tab "Manga Inpainting"**: Upload ảnh manga + line + mask
- **Tab "Settings"**: Kiểm tra trạng thái models

### Cách 2: Command Line (CLI)

#### Xóa vật thể trong ảnh tự nhiên (AOT-GAN)
```bash
python inpaint.py --mode aot \
    --image input.jpg \
    --mask mask.png \
    --output result.png \
    --model checkpoints/aot_gan/G0000000.pt
```

#### Xóa text/bóng thoại trong manga
```bash
python inpaint.py --mode manga \
    --image manga.png \
    --mask mask.png \
    --line lines.png \
    --output result.png \
    --checkpoint checkpoints/mangainpaintor
```

#### Xử lý hàng loạt
```bash
# Batch xóa vật thể
python inpaint.py --mode aot \
    --image ./images/ \
    --mask ./masks/ \
    --output ./results/ \
    --model checkpoints/aot_gan/G0000000.pt

# Batch xóa text manga
python inpaint.py --mode manga \
    --image ./manga_pages/ \
    --mask ./text_masks/ \
    --line ./line_drawings/ \
    --output ./clean_pages/ \
    --checkpoint checkpoints/mangainpaintor
```

### Cách 3: Sử dụng AOT-GAN Demo gốc (interactive)

```bash
cd aot_gan/src
python demo.py --dir_image ../../examples/ --pre_train ../../checkpoints/aot_gan/G0000000.pt
```

- Nhấn `+`/`-` để thay đổi kích thước bút vẽ
- Nhấn `Space` để inpaint
- Nhấn `r` để reset mask
- Nhấn `s` để lưu kết quả
- Nhấn `n` để chuyển ảnh tiếp theo
- Nhấn `Esc` để thoát

### Cách 4: Sử dụng MangaInpainting gốc

```bash
cd manga_inpainting
python test.py --checkpoints ../checkpoints/mangainpaintor \
    --input examples/test/imgs/ \
    --mask examples/test/masks/ \
    --line examples/test/lines/ \
    --output examples/test/results/
```

## Training (Huấn luyện)

### AOT-GAN
```bash
cd aot_gan/src
python train.py --dir_image ../../dataset --dir_mask ../../dataset --data_train places2
```

### MangaInpainting
Xem hướng dẫn chi tiết trong `manga_inpainting/README.md`

## Credits

- **AOT-GAN**: [Zeng et al., TVCG 2023](https://arxiv.org/abs/2104.01431) - Aggregated Contextual Transformations for High-Resolution Image Inpainting
- **MangaInpainting**: [Xie et al., SIGGRAPH 2021](https://dl.acm.org/doi/10.1145/3450626.3459822) - Seamless Manga Inpainting with Semantics Awareness

## License

- AOT-GAN: Apache 2.0
- MangaInpainting: See manga_inpainting/LICENSE
