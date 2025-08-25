suppressPackageStartupMessages({
  library(e1071)
  library(LiblineaR)
  library(caret)
  library(glmnet)
  library(foreach)
  library(doParallel)
  library(dplyr)
  library(tibble)
})

## 1) Loss-Shifting 원본 로드 --------------------------------------------
source("loss_shifting.r")   # IRLS 안정화·lambda_grid 1e-2 반영본

## 2) 확장판 두-반달(two-moon) 데이터 ------------------------------------
make_two_moon <- function(n,
                          noise        = 0.15,
                          mode         = c("clean", "outlier", "contam_pos"),
                          out_frac_pos = 0.2,   # +1 클래스 outlier 비율
                          out_frac_neg = 0.05,  # −1 클래스 outlier 비율
                          out_dist     = 3,     # outlier 거리지표
                          contam_frac  = 0.25) {# contam_pos용 −1 오염 비율
  mode <- match.arg(mode)
  
  ## 기본 두-moon
  n1  <- n %/% 2
  th1 <- runif(n1,       0, pi)
  th2 <- runif(n - n1,   0, pi)
  x1 <- cbind(cos(th1),  sin(th1)) +
    matrix(rnorm(2 * n1, 0, noise), n1)
  x2 <- cbind(1 - cos(th2),
              1 - sin(th2) - 0.5) +
    matrix(rnorm(2 * (n - n1), 0, noise), n - n1)
  X <- rbind(x1, x2)
  y <- c(rep(1, n1), rep(-1, n - n1))
  
  if (mode == "outlier") {
    move_points <- function(idxs, frac) {
      k <- ceiling(length(idxs) * frac)
      if (k == 0) return()
      sel <- sample(idxs, k)
      X[sel, ] <<- X[sel, ] + matrix(rnorm(2 * k, 0, 1), nrow = k) * out_dist
    }
    move_points(which(y ==  1), out_frac_pos)
    move_points(which(y == -1), out_frac_neg)
    
  } else if (mode == "contam_pos") {
    neg_idx <- which(y == -1)
    k <- ceiling(length(neg_idx) * contam_frac)
    if (k > 0) {
      donor <- x1[sample.int(n1, k, replace = TRUE), ]
      X[neg_idx[seq_len(k)], ] <<- donor +
        matrix(rnorm(2 * k, 0, noise), nrow = k)
    }
  }
  data.frame(X1 = X[, 1], X2 = X[, 2], y = y)
}

## 3) 단일 벤치마크 --------------------------------------------------------
run_benchmark <- function(
    n         = 2000,
    test_frac = 0.3,
    flip_pos  = 0.30,
    flip_neg  = 0.30,
    cv_folds  = 3,
    n_cores   = max(1, parallel::detectCores() - 1),
    seed      = 123456,
    data_mode         = "clean",   # "clean", "outlier", "contam_pos"
    make_two_moon_arg = list(),    # list(noise, out_frac_pos, out_frac_neg, out_dist, contam_frac)
    
    # ── Loss-Shifting 옵션 --------------------------------------------------
    ls_base_loss   = c("hinge","sqhinge","logistic","exp"),
    ls_style       = c("none","soft","hard"),
    ls_kernel      = c("linear","gaussian"),
    ls_svm_dual    = c(FALSE, TRUE),
    ls_lambda_grid = 10^seq(-2, 1, len = 5),
    ls_sigma_grid  = c(0.01, 0.1, 1, 10),
    ls_alpha_grid  = c(0.05, 0.5, 1, 2),
    ls_eta_grid    = c(0.01, 0.1, 0.5, 1),
    ls_restarts    = 3,
    ls_n_folds     = cv_folds
) {
  set.seed(seed)
  
  ## 3-1 데이터 생성 (항상 clean)
  raw <- do.call(make_two_moon,
                 c(list(n = n, mode = "clean"), make_two_moon_arg))
  idx   <- sample(seq_len(nrow(raw)))
  n_te  <- round(test_frac * n)
  test  <- raw[idx[seq_len(n_te)], ]
  train <- raw[idx[-seq_len(n_te)], ]
  
  ## 3-1.5 train에만 outlier/contam_pos 적용 -------------------------------
  get_arg <- function(name, default) {
    if (!is.null(make_two_moon_arg[[name]]))
      make_two_moon_arg[[name]] else default
  }
  
  if (data_mode == "outlier") {
    ofp <- get_arg("out_frac_pos", 0.2)
    ofn <- get_arg("out_frac_neg", 0.05)
    odd <- get_arg("out_dist",     3)
    move_points <- function(X_mat, y_vec, frac_pos, frac_neg, dist) {
      # +1 클래스
      i1 <- which(y_vec ==  1)
      k1 <- ceiling(length(i1) * frac_pos)
      if (k1 > 0) {
        sel1 <- sample(i1, k1)
        X_mat[sel1, ] <- X_mat[sel1, ] +
          matrix(rnorm(2 * k1, 0, 1), nrow = k1) * dist
      }
      # −1 클래스
      i2 <- which(y_vec == -1)
      k2 <- ceiling(length(i2) * frac_neg)
      if (k2 > 0) {
        sel2 <- sample(i2, k2)
        X_mat[sel2, ] <- X_mat[sel2, ] +
          matrix(rnorm(2 * k2, 0, 1), nrow = k2) * dist
      }
      X_mat
    }
    train[, 1:2] <- move_points(train[, 1:2], train$y, ofp, ofn, odd)
    
  } else if (data_mode == "contam_pos") {
    cf    <- get_arg("contam_frac", 0.25)
    noise<- get_arg("noise",       0.15)
    n1   <- n %/% 2
    # clean x1 재생성
    th1 <- runif(n1, 0, pi)
    x1  <- cbind(cos(th1), sin(th1)) +
      matrix(rnorm(2 * n1, 0, noise), n1)
    neg_idx <- which(train$y == -1)
    k <- ceiling(length(neg_idx) * cf)
    if (k > 0) {
      donor <- x1[sample.int(n1, k, replace = TRUE), ]
      train[neg_idx[seq_len(k)], 1:2] <-
        donor + matrix(rnorm(2 * k, 0, noise), nrow = k)
    }
  }
  
  ## 3-2 train 라벨 flip --------------------------------------------------
  pos_idx <- which(train$y ==  1)
  neg_idx <- which(train$y == -1)
  flip_pos_idx <- if (flip_pos > 0)
    sample(pos_idx, ceiling(length(pos_idx) * flip_pos)) else integer(0)
  flip_neg_idx <- if (flip_neg > 0)
    sample(neg_idx, ceiling(length(neg_idx) * flip_neg)) else integer(0)
  flip_idx <- c(flip_pos_idx, flip_neg_idx)
  train$y[flip_idx] <- -train$y[flip_idx]
  
  ## 3-3 표준화 -----------------------------------------------------------
  mu  <- colMeans(train[, 1:2])
  sdv <- apply(train[, 1:2], 2, sd)
  train[, 1:2] <- scale(train[, 1:2], mu, sdv)
  test [, 1:2] <- scale(test [, 1:2], mu, sdv)
  
  ## 3-4 병렬 클러스터 ----------------------------------------------------
  cl <- makeCluster(n_cores)
  clusterEvalQ(cl, {
    suppressPackageStartupMessages({
      library(e1071); library(LiblineaR); library(caret); library(glmnet)
      library(dplyr); library(tibble)
    })
    source("loss_shifting.r")
  })
  registerDoParallel(cl)
  on.exit({ stopCluster(cl); registerDoSEQ() })
  
  results <- list()
  errs    <- list()
  
  ########################################################################
  # A) Loss-Shifting -----------------------------------------------------
  combos <- expand.grid(ls_base_loss, ls_style,
                        ls_kernel, ls_svm_dual,
                        KEEP.OUT.ATTRS = FALSE,
                        stringsAsFactors = FALSE)
  names(combos) <- c("base_loss","style","kernel","svm_dual")
  
  ls_out <- foreach(i = seq_len(nrow(combos)),
                    .combine = bind_rows,
                    .packages = "dplyr") %dopar% {
                      r   <- combos[i, ]
                      tag <- with(r, sprintf("LossShift_%s_%s_%s_%s",
                                             base_loss, style, kernel,
                                             ifelse(svm_dual, "dual", "primal")))
                      tryCatch({
                        mod <- train_loss_shift(
                          X_train     = as.matrix(train[, 1:2]),
                          y_train     = train$y,
                          base_loss   = r$base_loss,
                          style       = r$style,
                          kernel      = r$kernel,
                          svm_dual    = r$svm_dual,
                          lambda_grid = ls_lambda_grid,
                          sigma_grid  = ls_sigma_grid,
                          alpha_grid  = ls_alpha_grid,
                          eta_grid    = ls_eta_grid,
                          restarts    = ls_restarts,
                          n_folds     = ls_n_folds,
                          verbose     = FALSE)
                        pred <- predict_loss_shift(mod$model, as.matrix(test[, 1:2]))
                        m    <- evaluate_metrics(test$y, pred)
                        tibble(model = tag,
                               accuracy  = m$accuracy,
                               precision = m$precision,
                               recall    = m$recall,
                               f1        = m$f1,
                               error     = NA_character_)
                      }, error = function(e) {
                        tibble(model = tag,
                               accuracy  = NA, precision = NA, recall = NA, f1 = NA,
                               error     = conditionMessage(e))
                      })
                    }
  results[["loss_shift"]] <- ls_out
  errs[["loss_shift"]]    <- filter(ls_out, !is.na(error))
  
  ########################################################################
  # B) LiblineaR (caret backend) ----------------------------------------
  lib_types <- list("L2SVM_L2" = 2, "L1SVM_L2" = 5)
  lib_out <- foreach(tn = names(lib_types),
                     .combine = bind_rows,
                     .packages = "LiblineaR") %dopar% {
                       tryCatch({
                         ctrl <- trainControl(method = "cv", number = cv_folds, allowParallel = FALSE)
                         fit <- caret::train(
                           x = as.matrix(train[, 1:2]),
                           y = factor(ifelse(train$y == 1, 1, 0)),
                           method    = "svmLinear3",
                           trControl = ctrl,
                           tuneGrid  = data.frame(
                             cost = 2 ^ (-5:5),
                             Loss = rep(lib_types[[tn]], 11))
                         )
                         pred <- predict(fit, as.matrix(test[, 1:2]))
                         m <- evaluate_metrics(test$y, ifelse(pred == 1, 1, -1))
                         tibble(model = paste0("LiblineaR_", tn),
                                accuracy  = m$accuracy,
                                precision = m$precision,
                                recall    = m$recall,
                                f1        = m$f1,
                                error     = NA_character_)
                       }, error = function(e) {
                         tibble(model = paste0("LiblineaR_", tn),
                                accuracy  = NA, precision = NA, recall = NA, f1 = NA,
                                error     = conditionMessage(e))
                       })
                     }
  results[["liblinear"]] <- lib_out
  errs[["liblinear"]]    <- filter(lib_out, !is.na(error))
  
  ########################################################################
  # C) e1071 SVM ---------------------------------------------------------
  svm_specs <- list(
    #"e1071_linear" = list(kernel = "linear", cost = 2 ^ (-3:3)),
    "e1071_rbf"    = list(kernel = "radial",
                          cost   = 2 ^ (-3:3),
                          gamma  = 2 ^ (-3:3))
  )
  svm_out <- foreach(tag = names(svm_specs),
                     .combine = bind_rows,
                     .packages = "e1071") %dopar% {
                       sp <- svm_specs[[tag]]
                       tryCatch({
                         tr <- tune.control(cross = cv_folds)
                         tune_res <- do.call(e1071::tune,
                                             c(list(e1071::svm,
                                                    train.x = as.matrix(train[, 1:2]),
                                                    train.y = factor(train$y, levels = c(-1,1))),
                                               sp, list(tunecontrol = tr)))
                         best <- tune_res$best.model
                         pred <- predict(best, as.matrix(test[, 1:2]))
                         m    <- evaluate_metrics(test$y, as.numeric(as.character(pred)))
                         tibble(model = tag,
                                accuracy  = m$accuracy,
                                precision = m$precision,
                                recall    = m$recall,
                                f1        = m$f1,
                                error     = NA_character_)
                       }, error = function(e) {
                         tibble(model = tag,
                                accuracy  = NA, precision = NA, recall = NA, f1 = NA,
                                error     = conditionMessage(e))
                       })
                     }
  results[["e1071"]] <- svm_out
  errs[["e1071"]]    <- filter(svm_out, !is.na(error))
  
  ########################################################################
  # D) GLMNET (선형 / RFF) -----------------------------------------------
  glmnet_out <- foreach(tag = c("GLMNet_rbf"),
                        .combine = bind_rows,
                        .packages = "glmnet") %dopar% {
                          tryCatch({
                            if (tag == "GLMNet_linear") {
                              Xtr <- as.matrix(train[, 1:2])
                              Xts <- as.matrix(test [, 1:2])
                            } else {
                              D <- 500; gamma <- 1
                              W <- matrix(rnorm(2 * D, 0, sqrt(2 * gamma)), ncol = 2)
                              b <- runif(D, 0, 2 * pi)
                              rff <- function(X)
                                sqrt(2 / D) * cos(X %*% t(W) +
                                                    matrix(b, nrow = nrow(X), ncol = D, byrow = TRUE))
                              Xtr <- rff(as.matrix(train[, 1:2]))
                              Xts <- rff(as.matrix(test [, 1:2]))
                            }
                            ytr_bin <- ifelse(train$y == 1, 1, 0)
                            cvfit <- cv.glmnet(
                              x = Xtr, y = ytr_bin,
                              family      = "binomial",
                              alpha       = 1,
                              nfolds      = cv_folds,
                              standardize = FALSE,
                              parallel    = FALSE)
                            prob <- as.vector(predict(cvfit, Xts, s = "lambda.min", type = "response"))
                            pred <- ifelse(prob >= 0.5, 1, -1)
                            m <- evaluate_metrics(test$y, pred)
                            tibble(model = tag,
                                   accuracy  = m$accuracy,
                                   precision = m$precision,
                                   recall    = m$recall,
                                   f1        = m$f1,
                                   error     = NA_character_)
                          }, error = function(e) {
                            tibble(model = tag,
                                   accuracy  = NA, precision = NA, recall = NA, f1 = NA,
                                   error     = conditionMessage(e))
                          })
                        }
  results[["glmnet"]] <- glmnet_out
  errs[["glmnet"]]    <- filter(glmnet_out, !is.na(error))
  
  ## 결과 반환 ------------------------------------------------------------
  list(
    metrics = bind_rows(results),
    errors  = bind_rows(errs)
  )
}

## 4) 반복 실행 ------------------------------------------------------------
repeat_benchmark <- function(
    n_repeat  = 5,
    n_sample  = 3000,
    test_frac = 0.25,
    flip_pos  = 0.30,
    flip_neg  = 0.30,
    cv_folds  = 3,
    n_cores   = max(1, parallel::detectCores() - 1),
    seeds     = 1:n_repeat,
    data_mode         = "clean",
    make_two_moon_arg = list(),
    ...               # 모델 관련 옵션 그대로 전달
) {
  if (length(seeds) != n_repeat)
    stop("seeds 길이가 n_repeat 와 같아야 합니다.")
  
  outs <- foreach(s = seeds, .packages = c("dplyr","tibble")) %do% {
    message(sprintf("▶ 반복 %d / %d (seed = %d)", s, n_repeat, s))
    run_benchmark(
      n         = n_sample,
      test_frac = test_frac,
      flip_pos  = flip_pos,
      flip_neg  = flip_neg,
      cv_folds  = cv_folds,
      n_cores   = n_cores,
      seed      = 100 + s,
      data_mode = data_mode,
      make_two_moon_arg = make_two_moon_arg,
      ...
    )
  }
  
  raw_all <- bind_rows(lapply(outs, `[[`, "metrics"), .id = "rep")
  err_all <- bind_rows(lapply(outs, `[[`, "errors"), .id = "rep")
  
  summary_tbl <- raw_all %>%
    group_by(model) %>%
    summarise(
      n_ok     = sum(is.na(error)),
      acc_mean = mean(accuracy, na.rm = TRUE),
      acc_sd   = sd  (accuracy, na.rm = TRUE),
      f1_mean  = mean(f1,       na.rm = TRUE),
      f1_sd    = sd  (f1,       na.rm = TRUE),
      .groups  = "drop"
    ) %>%
    arrange(desc(f1_mean))
  
  list(summary = summary_tbl, raw = raw_all, errors = err_all)
}
