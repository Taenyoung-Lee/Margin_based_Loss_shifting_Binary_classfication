source("loss_shifting.r")
#source("twomoon_data_parallel_test.r")
#source("spiral_data_parallel_assymetric_flip_test.r")
source("twomoon_data_parallel_assymetric_flip_test.r")

elapsed <- system.time(
  res <- repeat_benchmark(
    n_repeat = 50,
    n_sample = 200,
    test_frac = 0.3,
    flip_pos = 0.30,   # +1 클래스 30 %
    flip_neg = 0.10,   # −1 클래스 15 %
    ls_style       = c("none","soft","hard"),
    ls_kernel      = c("gaussian"),
    ls_svm_dual    = c(TRUE)
  )
)
print(elapsed["elapsed"])  
write.csv(res$raw, "0714_twomoon_seed123456_unsym.csv")


source("twomoon_data_parallel_assymetric_flip-contam_test.r")   # ← 새 파일

## 2) 반복 벤치마크 실행 (예: outlier 구조)
elapsed <- system.time(
  res <- repeat_benchmark(
    n_repeat = 2,
    n_sample = 100,
    test_frac = 0.30,
    flip_pos  = 0,   #0
    flip_neg  = 0,   #0
    data_mode = "outlier",   # 'clean' | 'outlier' | 'contam_pos'
    ls_style  = c("none","soft","hard"),
    ls_kernel = c("gaussian"),
    ls_svm_dual = c(TRUE)
  )
)
print(elapsed["elapsed"])
write.csv(res$raw, "0716_twomoon_robust_test_errorcheck.csv", row.names = FALSE)


source("twomoon_data_parallel_assymetric_flip-contam_test.r")   # ← 새 파일
## +1 클래스 10 %, −1 클래스 0 %  → asym outlier
res_asym_outlier <- repeat_benchmark(
  n_repeat  = 30,
  n_sample  = 200,
  data_mode = "outlier",
  make_two_moon_arg = list(
    out_frac_pos = 0.20,   # +1 클래스
    out_frac_neg = 0.05,   # −1 클래스
    out_dist     = 3       # 거리
  ),
  flip_pos = 0, flip_neg = 0,
  ls_style = c("none","soft","hard"),
  ls_kernel = "gaussian",
  ls_svm_dual = TRUE
)
write.csv(res_asym_outlier$raw, "0716_twomoon_outlier_contam.csv", row.names = FALSE)



res_asym_pos_contam <- repeat_benchmark(
  n_repeat  = 30,
  n_sample  = 200,
  data_mode = "contam_pos",
  make_two_moon_arg = list(
    contam_frac  = 0.3  # 1클래스 오염비율
  ),
  flip_pos = 0, flip_neg = 0,
  ls_style = c("none","soft","hard"),
  ls_kernel = "gaussian",
  ls_svm_dual = TRUE
)

