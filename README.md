# Bernini Long Video + Auto Model Prompt Enhancer

Custom nodes ComfyUI untuk mengedit video panjang dengan **Bernini-R** melalui pemrosesan per potongan dan overlap. Paket ini menyediakan prompt enhancer melalui model GGUF lokal atau server vLLM serta contoh workflow V2V 30 detik pada 16 fps.

## Fitur

- Pemrosesan video per potongan dengan panjang dan overlap yang dapat diatur.
- Penyambungan frame menggunakan crossfade atau cut.
- Sampling high-noise dan low-noise melalui dukungan Bernini native ComfyUI.
- Preset resolusi 360p/480p/720p/1080p dengan resize dan rasio input persis.
- Opsi tiled VAE untuk mengatur penggunaan memori.
- Prompt enhancer berbasis teks atau cuplikan gambar melalui server vLLM atau llama.cpp lokal.
- Auto-detect GGUF dari `models/LLM`, dropdown model dan `mmproj`, serta deteksi served model melalui `/v1/models`.
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

30 × 16 + 1 = **481 frame output**, mengikuti hitungan yang menyertakan frame awal dan akhir serta pola 4n+1. Window default 81 frame, overlap 17, stride 64 menghasilkan delapan proses sampling berurutan. Window terakhir berisi 33 frame. Overlap disatukan, bukan dihitung dua kali. Pada 16 FPS, rentang frame pertama sampai terakhir adalah 30 detik; durasi file 481 frame adalah 30,0625 detik.

81 merupakan default pada conditioning native, bukan hard cap API. Kode native menerima panjang yang lebih besar; itu tidak membuktikan kualitas atau memori tetap aman untuk 481 frame sekaligus. Pendekatan di sini membatasi context model per window. Ini adalah V2V: **dibutuhkan video sumber sepanjang 30 detik** untuk menghasilkan 30 detik. Input pendek tetap menghasilkan video pendek; node tidak menciptakan kelanjutan gerakan di luar sumber.

- Video loader: `force_rate=16`, `frame_load_cap=481`, `select_every_nth=1`, `format=None`.
- Long V2V: `fps=16`, `max_seconds=30`, `chunk_frames=81`, `overlap=17`.
- Video Combine: `frame_rate=16`, audio sumber tersambung.
- Parameter fps pada node Long V2V menghitung durasi; **tidak meresample input**. Bila mengubah fps, ubah loader dan saver juga.
- Output default memakai `resolution=480p`. Node menghitung ukuran dari rasio video input, melakukan resize per window, dan memakai padding sementara untuk memenuhi kelipatan 16. Padding dibuang setelah decode. Loader contoh memakai `custom_width=0`, `custom_height=0`, `format=None` agar rasio asli tidak berubah sebelum masuk ke node.

Mulai dengan `max_seconds=6`: ini menguji dua window dan satu sambungan. Setelah hasil benar, naikkan kembali menjadi 30. Sumber masih dimuat hingga batas loader; turunkan frame_load_cap juga jika ingin uji pendek lebih hemat RAM.

`crossfade` membaurkan frame yang waktunya sama pada overlap. `cut` memilih bagian awal dari window lama dan bagian akhir dari window baru. Keduanya mempertahankan jumlah frame, tetapi **bukan temporal attention bersama**, bukan optical flow, dan bukan penguncian latent. Crossfade bisa menimbulkan ghosting; cut bisa menampilkan lompatan detail. Seed sama tidak menjamin noise global yang selaras. Tidak ada automatic scene-cut detection; untuk video dengan pergantian shot, proses tiap shot terpisah. Referensi gambar opsional dikirim sebagai referensi native yang sama pada setiap window; efeknya bergantung pada prompt dan model.

## Frame count, FPS, dan video pembanding

Output **Bernini · Long V2V**: `images`, `report`, `frame_count` (INT), `fps` (FLOAT), dan `source_images` (IMAGE). Posisi output lama tetap sama. `frame_count` sama dengan jumlah frame pada `images` dan `source_images`. `fps` meneruskan nilai input node, bukan mendeteksi FPS dari tensor IMAGE atau melakukan resampling.

Target durasi = `1 + 4 × ceil(fps × max_seconds / 4)`. Pada 16 FPS: **5 detik = 81 frame, 10 detik = 161 frame, 30 detik = 481 frame**. Untuk nilai lain, target dibulatkan ke atas ke pola 4n+1 terdekat. Durasi file adalah `frame_count / fps`, sedangkan rentang waktu frame pertama sampai terakhir adalah `(frame_count - 1) / fps`. Contohnya 81 frame pada 16 FPS memiliki rentang 5 detik dan durasi file 5,0625 detik.

Sumber dipotong hingga target tersebut. Jika sumber lebih pendek, node hanya mengulang frame terakhir sebanyak 0–3 frame untuk mencapai 4n+1 terdekat; tidak memperpanjangnya sampai seluruh durasi yang diminta. Input 80 frame menjadi 81 dan frame tambahan tetap disertakan dalam hasil. `max_seconds=0` memakai seluruh sumber dengan penyelarasan akhir yang sama. `source_images` berisi sumber RGB yang dipotong/ditambah frame akhir identik, pada resolusi sumber, untuk pembanding.

`chunk_frames=81` tetap membatasi panjang tiap window. Padding tambahan di dalam window (misalnya karena overlap tidak sejajar grid temporal) hanya untuk pemrosesan window itu dan tetap dibuang saat penyambungan. Frame akhir yang menyelaraskan timeline utama dipertahankan; overlap tidak dihitung dua kali.

Untuk memeriksa hasil, lihat `input_frames`, `source_frames_used`, `tail_padding_frames`, `frames`, `fps`, `seconds` (durasi file), dan `frame_span_seconds` pada `report`. Periksa juga batas frame loader, `select_every_nth`, skip frame, node pemotong batch, dan durasi sumber. FPS Long V2V harus sama dengan FPS frame yang dimuat loader.

Untuk perbandingan menggunakan Image Concatenate:

1. Hubungkan `images` dan `source_images` dari Long V2V ke dua input Image Concatenate. Keduanya memiliki jumlah frame dan urutan waktu yang sama. Jangan memakai ImageFromBatch pada sumber mentah untuk menambah frame: node pemotong itu tidak membuat frame yang kurang.
2. Aktifkan `match_image_size=true` jika tersedia untuk menyamakan ukuran tampilan, karena `source_images` mempertahankan resolusi sumber.
3. Hubungkan `fps` Long V2V ke input `frame_rate` Video Combine pembanding dan Video Combine hasil edit. Ubah widget menjadi input jika belum ada soketnya.

Update paket, restart ComfyUI, lalu refresh halaman. Jika node lama belum menampilkan output baru, tambahkan ulang **Bernini · Long V2V** dan sambungkan kembali. Contoh workflow sudah menghubungkan output FPS ke Video Combine.

## Resolusi dan rasio video

Pilih `resolution` pada **Bernini · Long V2V**: `360p`, `480p`, `720p`, `1080p`, `source`, atau `custom`. Preset p menargetkan **sisi pendek** sehingga cocok untuk landscape, portrait, dan square. `width`/`height` hanya dipakai pada mode `custom`; pada preset p keduanya diabaikan.

Ukuran output mempertahankan rasio input **secara persis**, dengan dimensi piksel genap untuk encoder video umum. Karena jumlah piksel harus bulat, sisi pendek dipilih sedekat mungkin ke target p:

| Rasio input | Target 480p | Target 720p |
|---|---|---|
| 16:9 | 864 × 486 | 1280 × 720 |
| 9:16 | 486 × 864 | 720 × 1280 |
| 4:3 | 640 × 480 | 960 × 720 |
| 1:1 | 480 × 480 | 720 × 720 |

480p bukan selalu tepat 480 piksel: 16:9 pada tinggi 480 memerlukan lebar 853⅓ piksel, sehingga tidak bisa sekaligus memiliki dimensi bulat dan rasio persis. Node memilih 864×486. Pada rasio tidak lazim (misalnya 853:480), ukuran genap yang mempertahankan rasio persis bisa jauh dari target (1706×960). Node melaporkan ukuran aktual pada output `report`; dimensi canvas dibatasi 8192 px.

Frame di-resize menggunakan Lanczos float, lalu sisi luarnya dipad dengan piksel tepi ke kelipatan 16 untuk model. Setelah decode, hanya padding yang dibuang. Tidak ada crop pada isi gambar atau stretch rasio. Misalnya output 864×486 memakai canvas internal 864×496. Ini menjaga geometri input/output; hasil edit generatif tetap bergantung pada model.

`source` mempertahankan ukuran input, termasuk dimensi ganjil jika ada; beberapa encoder video memerlukan dimensi genap. `custom` mempertahankan perilaku lama: ukuran manual kelipatan 16 dengan center crop native. Workflow API lama yang tidak mengirim `resolution` tetap memakai mode custom. Untuk workflow UI lama, pilih preset yang diinginkan setelah update.

Agar rasio dihitung dari sumber asli, hindari crop/stretch/resize yang membulatkan rasio pada node sebelum Long V2V. Contoh workflow kini memuat resolusi sumber asli; kebutuhan RAM loader dapat meningkat, walaupun resize untuk Bernini dilakukan per window. Sebagai gambaran, 480 frame 1920×1080 RGB float32 membutuhkan sekitar 11.1 GiB hanya untuk input. Output 720p juga memakai lebih banyak memori daripada 480p.

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

Memori RAM sistem tetap penting: output 480 frame float32 pada 864×486×3 sendiri sekitar **2.25 GiB**, ditambah input, temporary buffer, model offload dan encoder video. Ini bukan streaming disk; pemuatan input dan penyimpanan hasil tetap berbentuk batch IMAGE. Ukuran VRAM puncak dan durasi render belum diukur.

## Prompt enhancer: auto-detect model lokal

Pola pemilihan model mengikuti [ComfyUI-MiniMaxH3-Prompter](https://github.com/DrakenDrop/ComfyUI-MiniMaxH3-Prompter): node memindai GGUF dalam `ComfyUI/models/LLM` dan subfoldernya. Ini adalah model **prompt enhancer**; bobot diffusion Bernini tetap dipilih pada loader ComfyUI.

1. Letakkan model instruct GGUF di `ComfyUI/models/LLM/`. Untuk vision, letakkan `mmproj` yang cocok dari distribusi model yang sama. Model harus didukung oleh build llama.cpp yang dipakai.
2. Node mencari `llama-server` secara otomatis. Jika MiniMaxH3 Prompter sudah terpasang pada ComfyUI yang sama, lokasi executable dari `config.json` dan folder `llama.cpp` miliknya ikut diperiksa. Tidak perlu konfigurasi tambahan jika ditemukan. Jika belum tersedia, pasang [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) dengan backend GPU yang sesuai. Untuk lokasi khusus, salin `bernini_config.example.json` menjadi `bernini_config.json` dan isi `llama_server_path` dengan path file atau folder, misalnya `"C:/llama.cpp"`.
3. Restart ComfyUI dan refresh halaman. Pilih file pada dropdown `llm_model`, lalu aktifkan `enabled`. Setelah menambahkan file, refresh daftar node/model melalui frontend atau reload halaman untuk mengambil daftar terbaru.
4. `mmproj=auto` mencocokkan nama keluarga model setelah menghapus penanda kuantisasi. File generik seperti `mmproj-F16.gguf` dipilih hanya bila folder itu berisi satu keluarga model dan satu projector generik. Bila pasangan tidak jelas, pilih secara manual. Untuk teks saja gunakan `none (text only)` atau `sample_frames=0`.
5. `context_size=8192` adalah nilai awal untuk teks dan gambar; sesuaikan dengan model. `unload_llm_after=true` menutup proses lokal milik node setelah prompt selesai sehingga VRAM dilepas sebelum Bernini berjalan.

Node menjalankan llama-server di localhost, dengan port pilihan **8091**. Jika port itu sedang dipakai, node otomatis memilih port kosong dan menulis port aktual ke log. Set `managed_port=0` untuk selalu memilih port otomatis, dan melepas model ComfyUI yang sedang berada di GPU sebelum loading LLM lokal. `base_url`, `model`, dan `api_key_env` hanya dipakai pada mode server. Pada mode lokal, node memakai alias internal tanpa mengirim API key server lain. Dengan `unload_llm_after=false`, model lokal tetap memakai VRAM; proses yang sama digunakan lagi jika konfigurasi sama. Error dan pembatalan saat startup tetap membersihkan proses milik node. Pembatalan saat request HTTP menunggu respons/timeout; atur `timeout` sesuai kebutuhan.

Pencarian executable berurutan: `llama_server_path` eksplisit → PATH → `config.json` MiniMaxH3 → folder `llama.cpp` pada node Bernini/MiniMaxH3 → ComfyUI dan folder induknya (portable) → home pengguna → `C:/llama.cpp`/drive sistem pada Windows. Layout root, `bin`, `build/bin`, dan `build/bin/Release` didukung. Path relatif pada konfigurasi dihitung dari folder plugin pemiliknya. Path eksplisit yang salah menampilkan error; kosongkan nilainya untuk kembali ke auto-detect.

Deteksi memakai instalasi executable yang sudah ada, tanpa mengunduh binary atau menghentikan proses MiniMaxH3. Bernini menjalankan proses sendiri pada port 8091 atau port kosong yang dipilih otomatis. Bila MiniMaxH3 masih menyimpan model di GPU, unload model itu terlebih dahulu jika diperlukan untuk menyediakan VRAM.

Konfigurasi tambahan: `extra_model_dirs` menerima daftar folder; path terdaftar ComfyUI dengan key `LLM`/`llm` juga dipindai. Model split hanya menampilkan shard pertama; seluruh shard harus tersedia. Node tidak mengunduh model atau llama.cpp. Port yang sudah dipakai dilewati; node tidak menghentikan atau mengambil alih server milik proses lain. Memilih port lain tidak membebaskan VRAM yang masih dipakai proses lama. Log startup ada di folder temporary sistem sebagai `bernini-llama-<pid>.log`.

## Prompt enhancer melalui server vLLM / VLM

vLLM adalah server inference. Untuk memahami gambar, server harus melayani **VLM**; untuk teks saja, model instruct biasa juga bisa. Node mengirim request OpenAI-compatible `/v1/chat/completions`.

Default `enabled=false`: instruksi diteruskan langsung ke CLIP, sehingga workflow dapat dipakai sebelum server tersedia. Dengan `enabled=true` dan `llm_model=server (vLLM / OpenAI-compatible)`, isi:

- `base_url`: alamat server yang dapat diakses **dari proses ComfyUI**, default `http://127.0.0.1:8000/v1`.
- `model=auto` (atau kosong): deteksi nama model melalui `GET /v1/models`. Jika server menyediakan beberapa model, node menampilkan nama yang tersedia dan meminta pilihan pada field `model`. Nama eksplisit seperti `bernini-enhancer` tetap didukung dan melewati discovery. Deteksi nama tidak membuktikan dukungan vision; gunakan model VLM untuk gambar.
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

Tes CPU mencakup timeline, overlap, padding temporal/spasial, resize dan rasio persis, discovery model, pemilihan projector, kepemilikan proses server, respons enhancer, dan struktur workflow. Lihat `validation.json` untuk hasil terakhir; pengujian proses llama-server memakai mock.

Validasi render GPU, inferensi model llama.cpp/vLLM nyata, dan benchmark performa belum dilakukan. Mulai dengan klip pendek untuk memeriksa hasil dan sambungan sebelum memproses video penuh.
