# Báo cáo tiến độ Model B (Flow Matching) — v7–v10

Cập nhật: 2026-11-27

Model B đã train xong 120k bước; kết quả test và RQ2 (số bước lấy mẫu) đã có, RQ1 (quét CFG w) đã có kết quả ở v10.

Notebook Kaggle: [anime-colorization-flow](https://www.kaggle.com/code/thanhnc241622/anime-colorization-flow) — bản chạy RQ1: [version 10](https://www.kaggle.com/code/thanhnc241622/anime-colorization-flow?scriptVersionId=374473278). Mã nguồn notebook: [notebooks/flow_kaggle.ipynb](../notebooks/flow_kaggle.ipynb).

## Trạng thái

Cả ba phần thực nghiệm của Model B đã xong: kết quả test, RQ2 và RQ1. Mọi số liệu dưới đây đo trên Tesla T4 (Kaggle), tập test 579 ảnh (cross_ref: 600), 5 seed mỗi ảnh.

| Session | Nội dung | Kết quả |
| --- | --- | --- |
| v1–v6 | Cache latent, smoke run, train đầy đủ | Xong, 120.000 bước (EMA weights) |
| v7 | Trần VAE + ảnh test 3 chế độ + đánh giá | Xong |
| v8 | RQ2: quét số bước 1–50 | Xong (168 phút) |
| v10 | RQ1: quét CFG w ∈ {1, 1.5, 2, 3, 4, 6} | Xong |

## Kết quả test

Với reference đúng (self_ref), Model B đạt FID 24,3 và LPIPS 0,114, còn cách trần VAE khá xa (FID 12,1, LPIPS 0,034). Cấu hình mặc định: 32 bước Euler, CFG w = 2,0.

| Chế độ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FD-DINOv2 ↓ | Edge F1 ↑ | Diversity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Trần VAE (encode→decode GT) | 26,46 | 0,861 | 0,034 | 12,1 | 74,1 | 0,884 | — |
| self_ref | 20,38 | 0,737 | 0,114 | 24,3 | 170,2 | 0,877 | 0,076 |
| shuffled_ref | — | — | — | 36,9 | 267,4 | 0,873 | 0,116 |
| cross_ref | — | — | — | 62,2 | 398,9 | 0,796 | 0,133 |

- Model B có dùng reference: đổi sang reference ngẫu nhiên (shuffled_ref) làm FID tăng từ 24,3 lên 36,9, trong khi nét vẽ giữ nguyên (Edge F1 0,873).
- Không có PSNR/SSIM/LPIPS cho shuffled_ref và cross_ref vì không có ground truth tương ứng.
- cross_ref dùng split khác (test_cross, 600 ảnh) nên FID không so trực tiếp được với hai dòng trên. Riêng cross_ref có thêm color_dist 0,557 và clip_sim 0,842; cần số của Model A để đánh giá hai số này.

## RQ2: số bước lấy mẫu

Chất lượng bão hòa từ 4–8 bước: 8 bước cho LPIPS và FD-DINOv2 tốt nhất, FID chỉ kém 0,9 điểm so với 32 bước, và nhanh hơn 2,3 lần (0,155 so với 0,352 giây/ảnh). Tất cả chạy self_ref, w = 2,0.

| Số bước | Giây/ảnh (T4) | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FD-DINOv2 ↓ | Edge F1 ↑ | Diversity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0,095 | 17,63 | 0,669 | 0,244 | 60,8 | 331,3 | 0,810 | 0,067 |
| 2 | 0,106 | 19,85 | 0,747 | 0,141 | 33,9 | 182,6 | 0,845 | 0,033 |
| 4 | 0,123 | **20,88** | **0,764** | 0,111 | 27,8 | 158,2 | 0,865 | 0,038 |
| 8 | 0,155 | 20,77 | 0,756 | **0,109** | 25,3 | **155,9** | 0,873 | 0,054 |
| 16 | 0,221 | 20,54 | 0,745 | 0,112 | 24,4 | 161,7 | 0,876 | 0,067 |
| 32 (mặc định) | 0,352 | 20,38 | 0,737 | 0,114 | **24,3** | 170,2 | **0,877** | 0,076 |
| 50 | 0,499 | 20,33 | 0,734 | 0,115 | 24,4 | 174,4 | **0,877** | **0,079** |

- Sau 4 bước, PSNR/SSIM giảm nhẹ trong khi diversity tăng: nhiều bước hơn cho ảnh đa dạng hơn nhưng lệch khỏi ground truth hơn một chút.
- Kết quả 32 bước trùng khớp tuyệt đối với lần chạy self_ref ở v7, nên việc lấy mẫu tái lập được.

## RQ1: đa dạng và độ trung thực (quét CFG w)

w = 1,5 cho kết quả tốt nhất: trung thực nhất (LPIPS 0,113, FID 23,9) mà vẫn đa dạng hơn mặc định w = 2 khoảng 12%. Tăng w quá 2 làm kém đi cả độ đa dạng lẫn độ trung thực. Chạy ở v10, self_ref, 32 bước.

| w | Diversity ↑ | LPIPS ↓ | FID ↓ | FD-DINOv2 ↓ | PSNR ↑ | Edge F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (không CFG) | **0,101** | 0,118 | 24,9 | 174,3 | 20,11 | **0,883** |
| 1,5 | 0,085 | **0,113** | **23,9** | **169,1** | **20,43** | 0,879 |
| 2 (mặc định) | 0,076 | 0,114 | 24,3 | 170,2 | 20,38 | 0,877 |
| 3 | 0,066 | 0,121 | 27,2 | 178,9 | 20,13 | 0,875 |
| 4 | 0,060 | 0,130 | 31,7 | 190,6 | 19,86 | 0,876 |
| 6 | 0,053 | 0,148 | 41,8 | 214,7 | 19,38 | 0,879 |

- Độ đa dạng giảm đều khi w tăng, từ 0,101 (w = 1) xuống 0,053 (w = 6), tức giảm 48%. Phần này đúng với giả thuyết.
- Độ trung thực không tăng theo w như giả thuyết dự đoán: LPIPS tốt nhất ở w = 1,5 rồi xấu dần, lên 0,148 ở w = 6. Vì vậy chỉ có đánh đổi thật sự trong khoảng w = 1–1,5; từ w > 2 cả hai đều kém đi.
- CFG có ích nhưng ít: so với không dùng guidance (w = 1), w = 1,5 giảm LPIPS từ 0,118 xuống 0,113 và FID từ 24,9 xuống 23,9.
- Edge F1 giữ ổn định 0,875–0,883 ở mọi w, nên độ đa dạng không đến từ việc tô lem ra ngoài nét sketch.
- Kết quả w = 2 trùng khớp tuyệt đối với lần chạy self_ref ở v7, nên việc lấy mẫu tái lập được.

## Việc tiếp theo

- [ ] Vẽ biểu đồ RQ2 bằng `plot_quality_vs_steps.py` và RQ1 bằng `plot_diversity_fidelity.py`
- [ ] Gói `outputs/` thành một file `outputs.tar` trong notebook, để tải output Kaggle không bị lỗi 429 do quá nhiều file
- [ ] Lấy `flow/log.csv` từ output v6 để có đường loss cho báo cáo cuối
- [ ] Cân nhắc dùng w = 1,5 làm mặc định (bảng kết quả test và RQ2 hiện chạy ở w = 2)
- [ ] Ghép số liệu với Model A (pix2pix) và Model C (ControlNet) trên cùng bộ metric
