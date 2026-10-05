# Bernini Long Video + vLLM Prompt Enhancer

Custom nodes ComfyUI untuk mengedit video panjang dengan **Bernini-R** melalui pemrosesan per potongan dan overlap. Paket ini menyediakan prompt enhancer melalui vLLM serta contoh workflow V2V 30 detik pada 16 fps.

## Fitur

- Pemrosesan video per potongan dengan panjang dan overlap yang dapat diatur.
- Penyambungan frame menggunakan crossfade atau cut.
- Sampling high-noise dan low-noise melalui dukungan Bernini native ComfyUI.
- Opsi tiled VAE untuk mengatur penggunaan memori.
- Prompt enhancer berbasis teks atau cuplikan gambar melalui server vLLM.
- Contoh workflow fast dan quality.

## Instalasi

1. Clone repo ini ke `ComfyUI/custom_nodes/`:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/DrakenDrop/ComfyUI-Bernini-LongVideo.git
   ```

2. Gunakan ComfyUI yang mendukung Bernini native dan format bobot yang dipilih. Node memerlukan `comfy_extras/nodes_bernini.py`, `BerniniConditioning.execute`, dan `SamplerCustom.sample`. Kode API diperiksa terhadap ComfyUI commit `5c460d8172fe30761ff67c0df3d5643bb74e0d70`.
3. Pastikan **VideoHelperSuite** terpasang. Contoh workflow tidak memerlukan KJNodes maupun wrapper Bernini tambahan.
4. Restart ComfyUI. Cari kategori **Bernini / Long Video**.
5. Buka `workflows/bernini_30s_fast.json`. Pilih atau unggah video sumber, lalu isi instruksi edit di node Prompt Enhancer.

Node tidak mengunduh model dan tidak mengganti instalasi PyTorch. Dependensi runtime hanya PyTorch, NumPy, dan Pillow yang biasanya sudah tersedia di ComfyUI. Tidak perlu memasang vLLM di lingkungan Python ComfyUI; enhancer menghubungi server terpisah.

Contoh workflow menggunakan bobot berikut:

- `wan2.2_bernini_r_high_noise_int8_convrot.safetensors`
- `wan2.2_bernini_r_low_noise_int8_convrot.safetensors`
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- `wan_2.1_vae.safetensors`
- Workflow fast juga memerlukan LoRA LightX2V high-noise dan low-noise yang tercantum pada node loader.

Pilih nama file yang sesuai jika letak model Anda berbeda. Nama dan bobot model tersedia pada [Comfy-Org/Bernini-R](https://huggingface.co/Comfy-Org/Bernini-R/tree/main/diffusion_models).

## Cara kerja 30 detik

30 × 16 = **480 frame output**. Window default 81 frame, overlap 17, stride 64 menghasilkan delapan proses sampling berurutan. Window terakhir berisi 32 frame asli dan dipad menjadi 33; padding dibuang saat penyambungan. Window model mengikuti pola 4n+1, sedangkan file video akhir boleh tepat 480 frame.

81 merupakan default pada conditioning native, bukan hard cap API. Kode native menerima panjang yang lebih besar; itu tidak membuktikan kualitas atau memori tetap aman untuk 481 frame sekaligus. Pendekatan di sini membatasi context model per window. Ini adalah V2V: **dibutuhkan video sumber sepanjang 30 detik** untuk menghasilkan 30 detik. Input pendek tetap menghasilkan video pendek; node tidak menciptakan kelanjutan gerakan di luar sumber.

- Video loader: `force_rate=16`, `frame_load_cap=480`, `select_every_nth=1`, `format=None`.
- Long V2V: `fps=16`, `max_seconds=30`, `chunk_frames=81`, `overlap=17`.
- Video Combine: `frame_rate=16`, audio sumber tersambung.
- Parameter fps pada node Long V2V menghitung durasi; **tidak meresample input**. Bila mengubah fps, ubah loader dan saver juga.
- Output default 480×832 portrait. Conditioning native memakai center crop; ubah width/height sesuai rasio sumber jika komposisi harus utuh. Loader mengecilkan lebar menjadi 480 sambil mempertahankan rasio, sehingga tidak memuat 480 frame resolusi asli yang besar. Untuk output lebih besar, naikkan custom_width loader juga agar sumber tidak terlanjur kehilangan detail.

Mulai dengan `max_seconds=6`: ini menguji dua window dan satu sambungan. Setelah hasil benar, naikkan kembali menjadi 30. Sumber masih dimuat hingga batas loader; turunkan frame_load_cap juga jika ingin uji pendek lebih hemat RAM.

`crossfade` membaurkan frame yang waktunya sama pada overlap. `cut` memilih bagian awal dari window lama dan bagian akhir dari window baru. Keduanya mempertahankan jumlah frame, tetapi **bukan temporal attention bersama**, bukan optical flow, dan bukan penguncian latent. Crossfade bisa menimbulkan ghosting; cut bisa menampilkan lompatan detail. Seed sama tidak menjamin noise global yang selaras. Tidak ada automatic scene-cut detection; untuk video dengan pergantian shot, proses tiap shot terpisah. Referensi gambar opsional dikirim sebagai referensi native yang sama pada setiap window; efeknya bergantung pada prompt dan model.

## Dua workflow

| Workflow | Pengaturan awal | Tujuan |
|---|---|---|
| `bernini_30s_fast.json` | INT8, LightX2V 4 langkah, split 2/2, CFG 1, LoRA strength 1 / 1.35 | Preset sampling dengan LoRA percepatan |
| `bernini_30s_quality.json` | INT8, tanpa LightX2V, 20 langkah, split 10/10, CFG 4 | Pembanding untuk mengevaluasi efek LoRA percepatan |

Preset quality adalah titik awal eksperimen, **bukan preset resmi atau jaminan lebih bagus**. Periksa split high/low, CFG, sampler, dan jumlah langkah terhadap hasil edit Anda. Contoh workflow membagi langkah high/low sebesar 50/50; pembagian ini dapat disesuaikan dan tidak menggunakan pemilihan expert otomatis berdasarkan threshold. Untuk pembanding kuantisasi, ganti kedua model ke FP16 resmi lalu ukur kualitas, VRAM, dan waktu pada klip pendek yang sama.

Contoh repo menggunakan negative prompt generik; sesuaikan dengan kebutuhan edit Anda. Input negatif memakai conditioning terpisah. Pada CFG 1, prompt negatif biasanya tidak dipakai dalam perhitungan guidance.

## Pengaturan memori dan performa

Yang diterapkan:

- Satu window sampling pada satu waktu, memakai pengelolaan memori model bawaan ComfyUI.
- High-noise menambah noise sekali; low-noise meneruskan hasil high tanpa menambah noise kedua kali.
- Hasil window disalin ke buffer CPU yang dialokasikan sekali. Tidak menyimpan seluruh hasil window di VRAM atau menggabungkan ulang seluruh video setiap iterasi.
- Encode prompt berada di luar loop window dan dapat di-cache oleh graph ComfyUI.
- Tiled VAE encode/decode tersedia dan aktif pada contoh workflow.
- Tidak memaksa reload model atau membersihkan cache CUDA di setiap window.

Pada GPU dengan VRAM yang mencukupi, uji `tiled_encode=false` dan `tiled_decode=false` pada klip pendek; non-tiled bisa lebih cepat dan menghindari seam VAE, tetapi peak memory meningkat. Tiling bukan optimasi kecepatan universal. Bandingkan waktu pada resolusi, seed, prompt, dan model identik.

INT8 menghemat penyimpanan bobot, tetapi tidak otomatis lebih cepat pada semua kernel. SageAttention dan torch.compile tidak dipaksakan karena manfaat serta kompatibilitas bergantung pada PyTorch/CUDA dan GPU. Node dapat menerima MODEL yang sudah dipatch lewat workflow jika Anda telah menguji patch tersebut. Jangan mengasumsikan LoRA Wan percepatan mempertahankan seluruh kemampuan editing Bernini.

Memori RAM sistem tetap penting: output float32 480×480×832×3 sendiri sekitar **2.14 GiB**, ditambah input, temporary buffer, model offload dan encoder video. Ini bukan streaming disk; pemuatan input dan penyimpanan hasil tetap berbentuk batch IMAGE. Ukuran VRAM puncak dan durasi render belum diukur.

## Prompt enhancer vLLM / VLM

vLLM adalah server inference. Untuk memahami gambar, server harus melayani **VLM**; untuk teks saja, model instruct biasa juga bisa. Node mengirim request OpenAI-compatible `/v1/chat/completions`.

Default `enabled=false`: instruksi diteruskan langsung ke CLIP, sehingga workflow dapat dipakai sebelum server tersedia. Dengan `enabled=true`, isi:

- `base_url`: alamat server yang dapat diakses **dari proses ComfyUI**, default `http://127.0.0.1:8000/v1`.
- `model`: served model name yang tepat, contoh `bernini-enhancer`.
- `sample_frames=0`: hanya teks dan scene_description.
- `sample_frames=4`: kirim empat frame kronologis yang tersebar di batch input; gunakan model vision dengan limit minimal empat gambar. Gambar diperkecil maksimal sisi 512 px. Ini bukan analisis gerakan per frame.
- `api_key_env`: nama environment variable, default `VLLM_API_KEY`. Isi secret di environment proses ComfyUI, bukan di workflow JSON.

Saat aktif, prompt dan frame yang dipilih dikirim ke server sesuai base_url. Tidak ada request jaringan saat enhancer dimatikan. Bila server bermasalah, node menampilkan error; tidak diam-diam mengganti prompt. Anda dapat menyalin prompt hasil ke input instruksi lalu mematikan enhancer agar render berikutnya tidak membutuhkan server. Hubungkan keluaran string ke node penampil teks yang tersedia bila ingin meninjaunya dahulu.

System prompt meminta instruksi edit bahasa Inggris yang ringkas, mempertahankan intent, gerakan, timing, kamera, dan bagian sumber yang tidak diminta berubah. Enhancer tidak melatih Bernini dan tidak menggantikan semantic planner latent milik Bernini penuh.

Contoh setup **Linux/WSL2**, di environment terpisah, memakai Qwen2.5-VL-7B-Instruct yang didokumentasikan vLLM. Contoh konfigurasi single GPU:

```bash
uv venv .venv-vllm
source .venv-vllm/bin/activate
uv pip install -U vllm --torch-backend auto
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --served-model-name bernini-enhancer \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 \
  --max-num-seqs 1 --gpu-memory-utilization 0.25 \
  --limit-mm-per-prompt '{"image":4,"video":0}'
```

0.25 adalah anggaran awal server pada GPU 96 GB, bukan jaminan model akan muat bersama render. Jika server dan Bernini berebut VRAM, buat prompt terlebih dahulu, salin hasilnya, lalu hentikan server untuk render. Jangan memakai default alokasi vLLM 90% ketika ingin berbagi GPU dengan Bernini. Pada Windows/WSL/container/mesin berbeda, alamat localhost mungkin perlu disesuaikan dengan jaringan Anda.

Referensi: [resep Qwen2.5-VL vLLM](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen2.5-VL.html), [input multimodal](https://docs.vllm.ai/en/latest/features/multimodal_inputs/), [model dan panduan Bernini-R resmi](https://github.com/bytedance/Bernini/blob/main/docs/bernini_r.md), [conditioning native ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/5c460d8172fe30761ff67c0df3d5643bb74e0d70/comfy_extras/nodes_bernini.py).

## Pengujian

Jalankan pada Python yang memiliki NumPy/Pillow:

```bash
python -m unittest discover -s tests -v
```

17 tes CPU lulus untuk logika timeline, overlap, padding, penanganan respons enhancer, dan struktur workflow. Pengujian HTTP menggunakan mock.

Validasi render GPU, integrasi server vLLM nyata, dan benchmark performa belum dilakukan. Mulai dengan klip pendek untuk memeriksa hasil dan sambungan sebelum memproses video penuh.
