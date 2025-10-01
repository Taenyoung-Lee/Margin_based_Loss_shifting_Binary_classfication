## ==================================================================
## Standalone Script: e1071 SVM on XOR — decision boundaries & 100-rep table
## (gamma & lambda tuning aligned with pred_e1071_aligned)
## ==================================================================

## -------- Dependencies --------
suppressPackageStartupMessages({
  library(e1071)
  library(ggplot2)
  library(dplyr)
  library(future)
  library(future.apply)
  library(progressr)
  library(knitr)
  library(kableExtra)
})

## -------- Global Config --------
N_TRAIN <- 200
N_TEST  <- 400
RATES_POS <- c(0.00, 0.05, 0.10, 0.15, 0.20)
RATE_NEG  <- 0.00
KERNEL    <- "radial"         # e1071::svm kernel ("radial"에서만 γ–λ 튜닝 수행)
K_REPS_TABLE <- 100           # 반복 횟수(테이블용)
SEED_BASE <- 20250916         # 전체 시드 베이스

## 멀티프로세싱 계획 (사용환경에 따라 실패 시 순차로 폴백)
ok <- TRUE
tryCatch({
  plan(multisession)
}, error = function(e) {
  message("multisession 사용 불가. sequential로 전환합니다.")
  plan(sequential); ok <<- FALSE
})

## -------- Utils --------
acc <- function(y_true, y_pred) mean(y_true == y_pred)

make_grid <- function(X, n = 300, pad = 0.8) {
  x1r <- range(X[,1]); x2r <- range(X[,2])
  x1s <- seq(x1r[1]-pad, x1r[2]+pad, length.out = n)
  x2s <- seq(x2r[1]-pad, x2r[2]+pad, length.out = n)
  expand.grid(X1 = x1s, X2 = x2s)
}

## XOR data generator (p=2)
gen_xor_2d <- function(n, p = 2, mu = 1, sd = 0.5, seed = NULL) {
  stopifnot(p == 2)
  if (!is.null(seed)) set.seed(seed)
  n_per_quad <- n %/% 4; n_rem <- n %% 4
  ns <- rep(n_per_quad, 4) + c(rep(1, n_rem), rep(0, 4 - n_rem))
  X_p1 <- cbind(rnorm(ns[1], mean=mu, sd=sd),  rnorm(ns[1], mean=mu, sd=sd))
  X_p2 <- cbind(rnorm(ns[2], mean=-mu, sd=sd), rnorm(ns[2], mean=-mu, sd=sd))
  X_n1 <- cbind(rnorm(ns[3], mean=mu, sd=sd),  rnorm(ns[3], mean=-mu, sd=sd))
  X_n2 <- cbind(rnorm(ns[4], mean=-mu, sd=sd), rnorm(ns[4], mean=mu, sd=sd))
  X <- rbind(X_p1, X_p2, X_n1, X_n2)
  y <- c(rep(1, ns[1]+ns[2]), rep(-1, ns[3]+ns[4]))
  idx <- sample(n)
  list(X = X[idx, ], y = y[idx])
}

## Asymmetric label flip
flip_labels_asym <- function(y, rate_pos = 0, rate_neg = 0, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  idx_p <- which(y==1); kp <- floor(length(idx_p)*rate_pos)
  idx_n <- which(y==-1); kn <- floor(length(idx_n)*rate_neg)
  flip_idx <- c(if(kp>0) sample(idx_p,kp) else integer(0),
                if(kn>0) sample(idx_n,kn) else integer(0))
  y2 <- y; if(length(flip_idx)) y2[flip_idx] <- -y2[flip_idx]
  list(y=y2, flipped_idx=flip_idx)
}

## -------- Helper: sigma grid from median distance (like pred_e1071_aligned) --------
.sigma_grid_from_median_scaled <- function(X, sigma_mult = c(0.5, 1, 2), seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  d <- as.vector(stats::dist(X, method = "euclidean"))
  d <- d[d > 0]
  if (length(d) == 0L) {
    med <- 1.0
  } else {
    med <- stats::median(d, na.rm = TRUE)
    if (!is.finite(med) || med <= 0) med <- 1.0
  }
  sort(med * sigma_mult)
}

## -------- Helper: fit SVM with gamma–lambda aligned tuning --------
fit_e1071_aligned <- function(Xtr, ytr, kernel = "radial", seed = NULL, cross = 3L) {
  y_fac <- factor(ifelse(ytr > 0, "pos", "neg"))
  
  if (kernel != "radial") {
    fit <- e1071::svm(x = Xtr, y = y_fac, kernel = "linear",
                      type = "C-classification", cost = 1.0, scale = FALSE)
    return(list(fit = fit, best = data.frame(kernel = "linear", cost = 1.0, gamma = NA_real_)))
  }
  
  n <- nrow(Xtr)
  if (!is.null(seed)) set.seed(seed)
  ##################################################################################################여기다여기
  ##################################################################################################여기다여기
  ##################################################################################################여기다여기
  ##################################################################################################여기다여기
  ##################################################################################################여기다여기
  
  sigma_grid  <- .sigma_grid_from_median_scaled(Xtr, sigma_mult = c(0.5, 1, 2), seed = seed)
  gamma_grid  <- 1 / (2 * sigma_grid^2)
  lambda_grid <- 10^seq(-12, -4, length.out = 4)
  cost_grid   <- 1 / (2 * n * lambda_grid + 1e-10)
  
  tuned <- e1071::tune(
    svm, train.x = Xtr, train.y = y_fac, kernel = "radial",
    type = "C-classification", scale = FALSE,
    ranges = list(gamma = gamma_grid, cost = cost_grid),
    tunecontrol = tune.control(cross = cross)
  )
  
  perf_df <- tuned$performances
  best_error <- min(perf_df$error, na.rm = TRUE)
  ties <- perf_df[perf_df$error == best_error, ]
  sorted_ties <- ties[order(ties$cost, ties$gamma), ]
  best_params <- sorted_ties[1, ]
  
  fit <- e1071::svm(
    x = Xtr, y = y_fac, kernel = "radial", type = "C-classification",
    cost = best_params$cost, gamma = best_params$gamma, scale = FALSE, decision.values = TRUE
  )
  
  list(fit = fit, best = best_params)
}

## -------- (1) rate_pos별 대표 결정경계 플롯 --------
plot_e1071_boundary_once <- function(n_train, n_test, rate_pos, rate_neg = 0.0,
                                     kernel = "radial",
                                     seed_base = 2025,
                                     save_png = TRUE, out_dir = "plots",
                                     cross = 3L) {
  set.seed(seed_base)
  Tr <- gen_xor_2d(n_train, seed = seed_base + 1L)
  Te <- gen_xor_2d(n_test,  seed = seed_base + 2L)
  
  ytr_noisy <- flip_labels_asym(Tr$y, rate_pos = rate_pos, rate_neg = rate_neg,
                                seed = seed_base + 3L)$y
  
  fit_obj <- fit_e1071_aligned(Tr$X, ytr_noisy, kernel = kernel,
                               seed = seed_base + 4L, cross = cross)
  fit <- fit_obj$fit
  best <- fit_obj$best
  
  grd <- make_grid(Tr$X, n = 300, pad = 0.8)
  pr  <- predict(fit, as.matrix(grd), decision.values = TRUE)
  dec <- as.numeric(attr(pr, "decision.values")[,1])
  df_grid <- cbind(grd, dec = dec)
  
  df_tr <- data.frame(X1 = Tr$X[,1], X2 = Tr$X[,2],
                      y_noisy = factor(ifelse(ytr_noisy > 0, "pos", "neg")))
  
  title_main <- if (kernel == "radial") {
    sprintf("e1071 SVM (radial, tuned C=%.4g, gamma=%.4g) — rate_pos = %.2f",
            best$cost, best$gamma, rate_pos)
  } else {
    sprintf("e1071 SVM (linear, C=1) — rate_pos = %.2f", rate_pos)
  }
  
  p <- ggplot() +
    geom_point(data = df_tr, aes(X1, X2, color = y_noisy), size = 1.6, alpha = 0.85) +
    geom_contour(data = df_grid, aes(X1, X2, z = dec), breaks = 0, linewidth = 1.1) +
    labs(
      title = title_main,
      subtitle = sprintf("rate_neg = %.2f, n_train = %d, n_test = %d (3-fold CV)", rate_neg, n_train, n_test),
      color = "Train label (noisy)"
    ) +
    theme_minimal(base_size = 12)
  
  print(p)
  
  if (save_png) {
    if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
    fn <- file.path(out_dir, sprintf("e1071_cv_boundary_ratepos_%0.2f.png", rate_pos))
    ggsave(filename = fn, plot = p, width = 6.8, height = 5.4, dpi = 150)
    message("Saved: ", fn)
  }
}

message("\n--- (1) Plotting decision boundaries per rate_pos ---")
for (i in seq_along(RATES_POS)) {
  rp <- RATES_POS[i]
  seed_here <- SEED_BASE + i*100  # rate마다 대표 샘플이 달라지도록
  plot_e1071_boundary_once(
    n_train = N_TRAIN, n_test = N_TEST,
    rate_pos = rp, rate_neg = RATE_NEG,
    kernel = KERNEL,
    seed_base = seed_here,
    save_png = TRUE, out_dir = "plots",
    cross = 3L
  )
}

## -------- (2) rate_pos별 100회 반복 정확도 평균/표준편차 --------
message("\n--- (2) 100-rep accuracy summary per rate_pos (new data each rep) ---")

with_progress({
  tg <- expand.grid(rate_pos = RATES_POS,
                    rep_i = seq_len(K_REPS_TABLE),
                    stringsAsFactors = FALSE)
  pbar <- progressor(steps = nrow(tg))
  
  acc_list <- future_lapply(seq_len(nrow(tg)), function(j) {
    rp  <- tg$rate_pos[j]
    rid <- tg$rep_i[j]
    cur_seed <- SEED_BASE + j  # 작업별 고유 시드
    
    Tr <- gen_xor_2d(N_TRAIN, seed = cur_seed + 1L)
    Te <- gen_xor_2d(N_TEST,  seed = cur_seed + 2L)
    ytr_noisy <- flip_labels_asym(Tr$y, rate_pos = rp, rate_neg = RATE_NEG,
                                  seed = cur_seed + 3L)$y
    
    ## --- 핵심: γ–λ 정렬 튜닝으로 학습 ---
    fit_obj <- fit_e1071_aligned(Tr$X, ytr_noisy, kernel = KERNEL,
                                 seed = cur_seed + 4L, cross = 3L)
    fit <- fit_obj$fit
    
    pr  <- predict(fit, Te$X)
    y_hat <- ifelse(pr == "pos", 1, -1)
    
    pbar(sprintf("rp=%.2f rep=%d", rp, rid))
    data.frame(rate_pos = rp, acc = mean(Te$y == y_hat))
  }, future.seed = TRUE)
})

acc_df <- dplyr::bind_rows(acc_list) %>%
  dplyr::mutate(rate_pos = factor(rate_pos, levels = RATES_POS, ordered = TRUE))

e1071_summary <- acc_df %>%
  dplyr::group_by(rate_pos) %>%
  dplyr::summarise(acc_mean = mean(acc),
                   acc_sd   = sd(acc),
                   .groups = "drop")

## 콘솔표 + kable(렌더러 있을 때 예쁘게) + CSV 저장
print(e1071_summary)
try({
  kable(e1071_summary, digits = 5, align = "c",
        col.names = c("rate_pos", "acc_mean", "acc_sd")) %>%
    kable_styling(full_width = FALSE) %>%
    print()
}, silent = TRUE)

# out_csv <- "e1071_ratepos_100reps_summary.csv"
# write.csv(e1071_summary, out_csv, row.names = FALSE)
# message("Saved CSV: ", out_csv)

# message("\nAll done.")
