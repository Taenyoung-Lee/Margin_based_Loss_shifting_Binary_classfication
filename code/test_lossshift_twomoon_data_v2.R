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
                          out_frac_pos = 0.2,
                          out_frac_neg = 0.05,
                          out_dist     = 3,
                          contam_frac  = 0.25) {
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
      X[sel, ] <<- X[sel, ] +
        matrix(rnorm(2 * k, 0, 1), nrow = k) * out_dist
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
  
  data.frame(X1 = X[,1], X2 = X[,2], y = y)
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
    data_mode = c("clean", "outlier", "outlier_flip",
                  "contam_pos", "contam_specific", "margin_noise"),
    make_two_moon_arg = list(),
    
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
  data_mode <- match.arg(data_mode)
  
  ## 3-1  항상 clean 데이터 생성 후 split
  raw <- do.call(make_two_moon,
                 c(list(n = n, mode = "clean"), make_two_moon_arg))
  idx   <- sample(seq_len(nrow(raw)))
  n_te  <- round(test_frac * n)
  test  <- raw[idx[seq_len(n_te)], ]
  train <- raw[idx[-seq_len(n_te)], ]
  
  ## helper
  get_arg <- function(name, default) {
    if (!is.null(make_two_moon_arg[[name]]))
      make_two_moon_arg[[name]] else default
  }
  
  ## 3-1.5 train 전용 오염 / 변형 ----------------------------------------
  if (data_mode == "outlier") {
    ofp <- get_arg("out_frac_pos", 0.2)
    ofn <- get_arg("out_frac_neg", 0.05)
    odd <- get_arg("out_dist",     3)
    move_pts <- function(idxs, frac) {
      k <- ceiling(length(idxs) * frac)
      if (k > 0) {
        sel <- sample(idxs, k)
        train[sel,1:2] <<- train[sel,1:2] +
          matrix(rnorm(2*k), nrow=k) * odd
      }
    }
    move_pts(which(train$y== 1), ofp)
    move_pts(which(train$y==-1), ofn)
    
  } else if (data_mode == "outlier_flip") {
    ofp <- get_arg("out_frac_pos", 0.2)
    ofn <- get_arg("out_frac_neg", 0.05)
    odd <- get_arg("out_dist",     3)
    
    idx_pos <- which(train$y ==  1)
    idx_neg <- which(train$y == -1)
    
    sel_pos <- if (length(idx_pos)>0)
      sample(idx_pos, ceiling(length(idx_pos)*ofp)) else integer(0)
    sel_neg <- if (length(idx_neg)>0)
      sample(idx_neg, ceiling(length(idx_neg)*ofn)) else integer(0)
    
    moved_idx <- c(sel_pos, sel_neg)
    
    if (length(sel_pos)>0)
      train[sel_pos,1:2] <- train[sel_pos,1:2] +
      matrix(rnorm(2*length(sel_pos)), nrow=length(sel_pos)) * odd
    if (length(sel_neg)>0)
      train[sel_neg,1:2] <- train[sel_neg,1:2] +
      matrix(rnorm(2*length(sel_neg)), nrow=length(sel_neg)) * odd
    
    # 이상치 포인트만 레이블 flip
    train$y[moved_idx] <- -train$y[moved_idx]
    
  } else if (data_mode == "contam_pos") {
    cf    <- get_arg("contam_frac", 0.25)
    noise <- get_arg("noise",       0.15)
    n1    <- n %/% 2
    th1 <- runif(n1, 0, pi)
    x1  <- cbind(cos(th1), sin(th1)) +
      matrix(rnorm(2*n1,0,noise), n1)
    neg_idx <- which(train$y==-1)
    k <- ceiling(length(neg_idx)*cf)
    if (k > 0) {
      donor <- x1[sample.int(n1,k,replace=TRUE), ]
      train[neg_idx[seq_len(k)],1:2] <-
        donor + matrix(rnorm(2*k,0,noise), nrow=k)
    }
    
  } else if (data_mode == "contam_specific") {
    label <- get_arg("contam_label", -1)
    cf    <- get_arg("contam_frac",  0.25)
    noise <- get_arg("noise",        0.15)
    idx_lab <- which(train$y == label)
    k <- ceiling(length(idx_lab)*cf)
    if (k > 0) {
      donor_pool <- raw[raw$y == -label, c("X1","X2")]
      donors <- donor_pool[sample(nrow(donor_pool), k, replace=TRUE), ]
      train[idx_lab[seq_len(k)],1:2] <-
        donors + matrix(rnorm(2*k,0,noise), nrow=k)
    }
    
  } else if (data_mode == "margin_noise") {
    mf <- get_arg("margin_frac",  0.2)
    mn <- get_arg("margin_noise", 0.5)
    idxs <- sample(seq_len(nrow(train)), ceiling(nrow(train)*mf))
    train[idxs,1:2] <- train[idxs,1:2] +
      matrix(rnorm(2*length(idxs), 0, mn), nrow=length(idxs))
  }
  ## clean 모드는 그대로
  
  ## 3-2  레이블 flip (global 비율)
  pos_idx <- which(train$y ==  1)
  neg_idx <- which(train$y == -1)
  if (flip_pos > 0) {
    k <- ceiling(length(pos_idx)*flip_pos)
    train$y[sample(pos_idx,k)] <- -1
  }
  if (flip_neg > 0) {
    k <- ceiling(length(neg_idx)*flip_neg)
    train$y[sample(neg_idx,k)] <-  1
  }
  
  ## 3-3  표준화 ----------------------------------------------------------
  mu  <- colMeans(train[,1:2]);  sdv <- apply(train[,1:2],2,sd)
  train[,1:2] <- scale(train[,1:2], mu, sdv)
  test [,1:2] <- scale(test [,1:2], mu, sdv)
  
  ## 3-4  병렬 클러스터 ---------------------------------------------------
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
  
  results <- list(); errs <- list()
  
  ########################################################################
  
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
                          X_train     = as.matrix(train[,1:2]),
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
                        pred <- predict_loss_shift(mod$model, as.matrix(test[,1:2]))
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
                           x = as.matrix(train[,1:2]),
                           y = factor(ifelse(train$y==1,1,0)),
                           method    = "svmLinear3",
                           trControl = ctrl,
                           tuneGrid  = data.frame(cost = 2^(-5:5), Loss = rep(lib_types[[tn]],11))
                         )
                         pred <- predict(fit, as.matrix(test[,1:2]))
                         m <- evaluate_metrics(test$y, ifelse(pred==1,1,-1))
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
    "e1071_rbf" = list(kernel = "radial", cost = 2^(-3:3), gamma = 2^(-3:3))
  )
  svm_out <- foreach(tag = names(svm_specs),
                     .combine = bind_rows,
                     .packages = "e1071") %dopar% {
                       sp <- svm_specs[[tag]]
                       tryCatch({
                         tr <- tune.control(cross = cv_folds)
                         tune_res <- do.call(e1071::tune,
                                             c(list(e1071::svm,
                                                    train.x = as.matrix(train[,1:2]),
                                                    train.y = factor(train$y, levels=c(-1,1))),
                                               sp, list(tunecontrol = tr)))
                         best <- tune_res$best.model
                         pred <- predict(best, as.matrix(test[,1:2]))
                         m <- evaluate_metrics(test$y, as.numeric(as.character(pred)))
                         tibble(model     = tag,
                                accuracy  = m$accuracy,
                                precision = m$precision,
                                recall    = m$recall,
                                f1        = m$f1,
                                error     = NA_character_)
                       }, error = function(e) {
                         tibble(model     = tag,
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
                              Xtr <- as.matrix(train[,1:2])
                              Xts <- as.matrix(test[,1:2])
                            } else {
                              D <- 500; gamma <- 1
                              W <- matrix(rnorm(2*D,0,sqrt(2*gamma)), ncol=2)
                              b <- runif(D, 0, 2*pi)
                              rff <- function(X)
                                sqrt(2/D) * cos(X %*% t(W) +
                                                  matrix(b, nrow=nrow(X), ncol=D, byrow=TRUE))
                              Xtr <- rff(as.matrix(train[,1:2]))
                              Xts <- rff(as.matrix(test[,1:2]))
                            }
                            ytr_bin <- ifelse(train$y==1,1,0)
                            cvfit <- cv.glmnet(x=Xtr, y=ytr_bin, family="binomial",
                                               alpha=1, nfolds=cv_folds,
                                               standardize=FALSE, parallel=FALSE)
                            prob <- as.vector(predict(cvfit, Xts, s="lambda.min", type="response"))
                            pred <- ifelse(prob>=0.5,1,-1)
                            m <- evaluate_metrics(test$y, pred)
                            tibble(model     = tag,
                                   accuracy  = m$accuracy,
                                   precision = m$precision,
                                   recall    = m$recall,
                                   f1        = m$f1,
                                   error     = NA_character_)
                          }, error = function(e) {
                            tibble(model     = tag,
                                   accuracy  = NA, precision = NA, recall = NA, f1 = NA,
                                   error     = conditionMessage(e))
                          })
                        }
  results[["glmnet"]] <- glmnet_out
  errs[["glmnet"]]    <- filter(glmnet_out, !is.na(error)) 
  ########################################################################
  
  list(metrics = bind_rows(results),
       errors  = bind_rows(errs))
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
    ...
) {
  if (length(seeds) != n_repeat)
    stop("seeds 길이가 n_repeat 와 같아야 합니다.")
  
  outs <- foreach(s = seeds, .packages=c("dplyr","tibble")) %do% {
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
  
  raw_all <- bind_rows(lapply(outs, `[[`, "metrics"), .id="rep")
  err_all <- bind_rows(lapply(outs, `[[`, "errors"),  .id="rep")
  
  summary_tbl <- raw_all %>%
    group_by(model) %>%
    summarise(
      n_ok     = sum(is.na(error)),
      acc_mean = mean(accuracy, na.rm=TRUE),
      acc_sd   = sd  (accuracy, na.rm=TRUE),
      f1_mean  = mean(f1,       na.rm=TRUE),
      f1_sd    = sd  (f1,       na.rm=TRUE),
      .groups  = "drop"
    ) %>%
    arrange(desc(f1_mean))
  
  list(summary = summary_tbl, raw = raw_all, errors = err_all)
}