| model                        |   psnr |     ssim |   ms_ssim |     lpips |    fid |         kid |   fd_dinov2 |   diversity |   edge_f1 |
|:-----------------------------|-------:|---------:|----------:|----------:|-------:|------------:|------------:|------------:|----------:|
| SCFT (CVPR 2020) †           |  23.89 | nan      |    0.922  | nan       | nan    | nan         |      nan    |   nan       |  nan      |
| Li et al. 2022 †             |  18.63 | nan      |    0.821  | nan       | nan    | nan         |      nan    |   nan       |  nan      |
| Cao et al. 2022 †            |  19.77 | nan      |    0.839  | nan       | nan    | nan         |      nan    |   nan       |  nan      |
| AnimeDiffusion (TVCG 2024) † |  25.47 | nan      |    0.96   | nan       | nan    | nan         |      nan    |   nan       |  nan      |
| vae_ceiling                  |  26.46 |   0.8607 |    0.9688 |   0.03428 |  12.09 |   0.0002933 |       74.11 |   nan       |    0.8842 |
| flow                         |  20.38 |   0.7368 |    0.8813 |   0.1141  |  24.32 |   0.002673  |      170.2  |     0.07567 |    0.877  |
| controlnet                   |  11.71 |   0.5397 |    0.5669 |   0.2991  |  53.07 |   0.02674   |      221.4  |     0.252   |    0.883  |
| controlnet_lora              |  16.05 |   0.7045 |    0.7549 |   0.2014  |  51.49 |   0.029     |      168.5  |     0.1686  |    0.859  |

† reported in AnimeDiffusion (Cao et al., TVCG 2024), Table 1.
