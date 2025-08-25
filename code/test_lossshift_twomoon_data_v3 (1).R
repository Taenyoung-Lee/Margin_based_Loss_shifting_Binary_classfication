suppressPackageStartupMessages({
  library(e1071)
  library(LiblineaR)
  library(caret)
  library(glmnet)
  library(foreach)
  library(doParallel)
  library(dplyr)
  library(tibble)
  library(kernlab)
})

## 1) Loss-Shifting 구현 로드 --------------------------------------------
source("loss_shifting.r")   # train_loss_shift(), predict_loss_shift(), evaluate_metrics()

## 2) 지역별 아웃라이어 클러스터 생성 (클래스별 비율) ------------------------
add_regional_outliers <- function(data,
                                  clusters = list(
                                    list(cx =  0.0, cy =  1.1, label = -1),
                                    list(cx = -1.7, cy =  0.2, label = -1),
                                    list(cx =  0.0, cy = -1.1, label =  1),
                                    list(cx =  1.7, cy =  0.2, label =  1)
                                  ),
                                  outlier_frac = 0.05,  # 각 클래스당 비율
                                  sd           = 0.05) {
  dfs <- list(data)
  for (cl in clusters) {
    n_class <- sum(data$y == cl$label)
    n_extra <- ceiling(n_class * outlier_frac)
    pts     <- matrix(rnorm(2 * n_extra, 0, sd), ncol = 2)
    dfs[[length(dfs) + 1]] <- data.frame(
      X1 = pts[,1] + cl$cx,
      X2 = pts[,2] + cl$cy,
      y  = cl$label
    )
  }
  bind_rows(dfs)
}

## 3) 두-반달(two-moon) 데이터 생성 (항상 clean) --------------------------
make_two_moon <- function(n, noise = 0.2) {
  n1  <- n %/% 2
  th1 <- runif(n1,     0, pi)
  th2 <- runif(n - n1, 0, pi)
  x1  <- cbind(cos(th1), sin(th1)) +
    matrix(rnorm(2 * n1, 0, noise), ncol = 2)
  x2  <- cbind(1 - cos(th2),
               1 - sin(th2) - 0.5) +
    matrix(rnorm(2 * (n - n1), 0, noise), ncol = 2)
  data.frame(
    X1 = c(x1[,1], x2[,1]),
    X2 = c(x1[,2], x2[,2]),
    y  = c(rep(1,  n1), rep(-1, n - n1))
  )
}

## 4) 단일 벤치마크 --------------------------------------------------------
run_benchmark <- function(
    n             = 2000,
    test_frac     = 0.3,
    cv_folds      = 3,
    n_cores       = max(1, parallel::detectCores() - 1),
    seed          = 123456,
    data_mode       = c("clean", "regional_outlier", "outlier", "contam_pos",
                        "outlier_flip", "contam_specific", "margin_noise",
                        "linear_sep"),
    clusters      = NULL,
    outlier_frac  = 0.05,
    outlier_sd    = 0.05,
    ls_base_loss   = c("hinge","sqhinge","logistic","exp"),
    ls_style       = c("none","soft","hard"),
    ls_kernel      = c("linear"),
    ls_svm_dual    = c(FALSE),
    ls_lambda_grid = 2^c(-5, -4, -3, -2, -1,0,1,2, 3),
    ls_sigma_grid  = c(0.001, 0.01, 0.5, 1 ,2),
    ls_alpha_grid  = c(0.5, 1, 2),
    ls_eta_grid    = c(0.5,1,2),
    ls_restarts    = 3,
    ls_n_folds     = cv_folds,
    flip_train_frac = list(pos = 0.0, neg = 0.0),  # ② train 레이블 뒤집기 비율
    flip_test_frac  = list(pos = 0.0, neg = 0.0)  # ② test 레이블 뒤집기 비율
    
) {
  set.seed(seed)
  data_mode <- match.arg(data_mode)
  
  ## 4-1) 데이터 생성 -----------------------------------------------------
  if (data_mode == "linear_sep") {
    #  (a) 클래스별로 가우시안 클러스터 생성
    n_pos <- n %/% 2
    n_neg <- n - n_pos
    pos  <- matrix(rnorm(2 * n_pos, mean =  1, sd = 0.9), ncol = 2)
    neg  <- matrix(rnorm(2 * n_neg, mean = -1, sd = 0.9), ncol = 2)
    raw  <- data.frame(
      X1 = c(pos[,1], neg[,1]),
      X2 = c(pos[,2], neg[,2]),
      y  = c(rep(1, n_pos), rep(-1, n_neg))
    )
  } else {
    # 기존 two-moon
    raw <- make_two_moon(n = n, noise = 0.15)
  }
  
  ## 4-2) Train/Test 분리 --------------------------------------------------
  idx  <- sample(nrow(raw))
  n_te <- round(test_frac * nrow(raw))
  test <- raw[idx[1:n_te], ]
  train<- raw[idx[-(1:n_te)], ]
  
  ## 4-3) linear_sep 모드일 때 레이블 뒤집기 -------------------------------
  if (data_mode == "linear_sep") {
    flip_labels <- function(df, frac) {
      for (lab in c(1, -1)) {
        ix     <- which(df$y == lab)
        n_flip <- ceiling(length(ix) * frac[[ if (lab==1) "pos" else "neg" ]])
        toflip <- sample(ix, n_flip)
        df$y[toflip] <- -lab
      }
      df
    }
    train <- flip_labels(train, flip_train_frac)
    test  <- flip_labels(test,  flip_test_frac)
  }
  
  ## 4-4) regional_outlier 등 기존 모드 처리 -------------------------------
  if (data_mode == "regional_outlier") {
    train <- add_regional_outliers(
      data        = train,
      clusters    = clusters %||% list(
        list(cx= 0.0, cy= 1.4, label=-1),
        list(cx=-1.2, cy= 0.2, label=-1),
        list(cx= 0.8, cy=-0.7, label= 1),
        list(cx= 2.1, cy= 0.2, label= 1)
      ),
      outlier_frac = outlier_frac,
      sd           = outlier_sd
    )
  }
  # 추가 모드(outlier, contam_pos 등)는 여기에 동일 패턴으로 구현 가능
  
  # 4-4) 표준화
  mu   <- colMeans(train[,1:2])
  sdv  <- apply(train[,1:2], 2, sd)
  train[,1:2] <- scale(train[,1:2], center = mu, scale = sdv)
  test [,1:2] <- scale(test [,1:2], center = mu, scale = sdv)
  
  # 4-5) 병렬 클러스터 설정
  cl <- makeCluster(n_cores)
  clusterEvalQ(cl, {
    suppressPackageStartupMessages({
      library(e1071); library(LiblineaR); library(caret)
      library(glmnet); library(dplyr); library(tibble); library(kernlab)
    })
    source("loss_shifting.r")
  })
  registerDoParallel(cl)
  
  results <- list(); errs <- list()
  
  ## A) Loss-Shifting -----------------------------------------------------
  combos <- expand.grid(ls_base_loss, ls_style,
                        ls_kernel, ls_svm_dual,
                        KEEP.OUT.ATTRS = FALSE,
                        stringsAsFactors = FALSE)
  names(combos) <- c("base_loss","style","kernel","svm_dual")
  ls_out <- foreach(i = seq_len(nrow(combos)),
                    .combine = bind_rows,
                    .packages = "dplyr") %dopar% {
                      r   <- combos[i, ]
                      tag <- sprintf("LossShift_%s_%s_%s_%s",
                                     r$base_loss, r$style, r$kernel,
                                     ifelse(r$svm_dual, "dual", "primal"))
                      tryCatch({
                        mod  <- train_loss_shift(
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
                          verbose     = FALSE
                        )
                        pred <- predict_loss_shift(mod$model, as.matrix(test[,1:2]))
                        m    <- evaluate_metrics(test$y, pred)
                        tibble(model     = tag,
                               accuracy  = m$accuracy,
                               precision = m$precision,
                               recall    = m$recall,
                               f1        = m$f1,
                               error     = NA_character_)
                      }, error = function(e) {
                        tibble(model     = tag,
                               accuracy  = NA, precision = NA,
                               recall    = NA, f1        = NA,
                               error     = conditionMessage(e))
                      })
                    }
  results[["loss_shift"]] <- ls_out
  errs   [["loss_shift"]] <- filter(ls_out, !is.na(error))
  
  ## B) LiblineaR (caret backend) ----------------------------------------
  lib_types <- list("L2SVM_L2"=2, "L1SVM_L2"=5)
  lib_out <- foreach(tn = names(lib_types),
                     .combine = bind_rows,
                     .packages = c("LiblineaR","caret","dplyr")) %dopar% {
                       tryCatch({
                         ctrl <- trainControl(method="cv", number=cv_folds, allowParallel=FALSE)
                         fit  <- caret::train(
                           x         = as.matrix(train[,1:2]),
                           y         = factor(ifelse(train$y==1,1,0)),
                           method    = "svmLinear",#svmLinear3
                           trControl = ctrl,
                           tuneGrid  = data.frame(C = 2^(-1:1))
                                                  #,Loss = rep(lib_types[[tn]], length=5))
                         )
                         pred <- predict(fit, as.matrix(test[,1:2]))
                         m    <- evaluate_metrics(test$y, ifelse(pred==1,1,-1))
                         tibble(model     = paste0("LiblineaR_", tn),
                                accuracy  = m$accuracy,
                                precision = m$precision,
                                recall    = m$recall,
                                f1        = m$f1,
                                error     = NA_character_)
                       }, error = function(e) {
                         tibble(model     = paste0("LiblineaR_", tn),
                                accuracy  = NA, precision = NA,
                                recall    = NA, f1        = NA,
                                error     = conditionMessage(e))
                       })
                     }
  results[["liblinear"]] <- lib_out
  errs   [["liblinear"]] <- filter(lib_out, !is.na(error))
  
  ## C) e1071 SVM ---------------------------------------------------------
  #svm_specs <- list(
  #  "e1071_rbf" = list(kernel="radial",
  #                     cost = 2^(-2:2),
  #                     gamma= 2^(-2:2))
  #)
  #svm_out <- foreach(tag = names(svm_specs),
  #                   .combine = bind_rows,
  #                   .packages = "e1071") %dopar% {
  #                     sp <- svm_specs[[tag]]
  #                     tryCatch({
  #                       tune_res <- tune(
  #                         svm,
  #                         train.x     = as.matrix(train[,1:2]),
  #                         train.y     = factor(train$y, levels=c(-1,1)),
  #                         kernel      = sp$kernel,
  #                         cost        = sp$cost,
#                           gamma       = sp$gamma,
#                           tunecontrol = tune.control(cross=cv_folds)
#                         )
#                         best <- tune_res$best.model
#                         pred <- predict(best, as.matrix(test[,1:2]))
#                         m    <- evaluate_metrics(test$y, as.numeric(as.character(pred)))
##                         tibble(model     = tag,
#                                accuracy  = m$accuracy,
#                                precision = m$precision,
#                                recall    = m$recall,
#                                f1        = m$f1,
#                                error     = NA_character_)
#                       }, error = function(e) {
###                         tibble(model     = tag,
#                                accuracy  = NA, precision = NA,
#                                recall    = NA, f1        = NA,
#                                error     = conditionMessage(e))
#                       })
##                     }
#  results[["e1071"]] <- svm_out
#  errs   [["e1071"]] <- filter(svm_out, !is.na(error))
  ## C) e1071 SVM (linear kernel)
  svm_specs <- list(
    "e1071_linear" = list(
      kernel = "linear",
      cost   = 2^(-1:1)
    )
  )
  
  svm_out <- foreach(tag = names(svm_specs),
                     .combine = bind_rows,
                     .packages = "e1071") %dopar% {
                       sp <- svm_specs[[tag]]
                       tryCatch({
                         tune_res <- tune(
                           svm,
                           train.x     = as.matrix(train[,1:2]),
                           train.y     = factor(train$y, levels=c(-1,1)),
                           kernel      = sp$kernel,         # 변경: linear
                           cost        = sp$cost,           # cost 그리드
                           tunecontrol = tune.control(cross=cv_folds)
                         )
                         best <- tune_res$best.model
                         pred <- predict(best, as.matrix(test[,1:2]))
                         m    <- evaluate_metrics(test$y, as.numeric(as.character(pred)))
                         tibble(model     = tag,
                                accuracy  = m$accuracy,
                                precision = m$precision,
                                recall    = m$recall,
                                f1        = m$f1,
                                error     = NA_character_)
                       }, error = function(e) {
                         tibble(model     = tag,
                                accuracy  = NA, precision = NA,
                                recall    = NA, f1        = NA,
                                error     = conditionMessage(e))
                       })
                     }
  results[["e1071"]] <- svm_out
  errs   [["e1071"]] <- filter(svm_out, !is.na(error))  
  ## D) GLMNET (RFF 포함) -----------------------------------------------
  glmnet_out <- foreach(tag = c("GLMNet_linear"),
                        .combine = bind_rows,
                        .packages = "glmnet") %dopar% {
                          tryCatch({
                            if (tag == "GLMNet_linear") {
                              Xtr <- as.matrix(train[,1:2])
                              Xts <- as.matrix(test[,1:2])
                            } else {
                              D     <- 500; gamma_rff <- 1
                              W     <- matrix(rnorm(2*D,0,sqrt(2*gamma_rff)), ncol=2)
                              b     <- runif(D, 0, 2*pi)
                              rff   <- function(X_) sqrt(2/D) * cos(X_%*%t(W) +
                                                                      matrix(b, nrow=nrow(X_), ncol=D, byrow=TRUE))
                              Xtr <- rff(as.matrix(train[,1:2]))
                              Xts <- rff(as.matrix(test[,1:2]))
                            }
                            ytr_b <- ifelse(train$y==1,1,0)
                            cvfit <- cv.glmnet(x=Xtr, y=ytr_b, family="binomial",
                                               alpha=1, nfolds=cv_folds,
                                               standardize=FALSE)
                            prob  <- as.vector(predict(cvfit, Xts, s="lambda.min", type="response"))
                            pred  <- ifelse(prob>=0.5,1,-1)
                            m     <- evaluate_metrics(test$y, pred)
                            tibble(model     = tag,
                                   accuracy  = m$accuracy,
                                   precision = m$precision,
                                   recall    = m$recall,
                                   f1        = m$f1,
                                   error     = NA_character_)
                          }, error = function(e) {
                            tibble(model     = tag,
                                   accuracy  = NA, precision = NA,
                                   recall    = NA, f1        = NA,
                                   error     = conditionMessage(e))
                          })
                        }
  results[["glmnet"]] <- glmnet_out
  errs   [["glmnet"]] <- filter(glmnet_out, !is.na(error))
  
  ## 4-6) 클러스터 종료
  stopCluster(cl)
  registerDoSEQ()
  
  list(metrics = bind_rows(results),
       errors  = bind_rows(errs))
}

# 1) repeat_benchmark 정의 수정
repeat_benchmark <- function(
    n_repeat        = 5,
    n_sample        = 3000,                              # run_benchmark의 n
    test_frac       = 0.25,
    cv_folds        = 3,
    n_cores         = max(1, parallel::detectCores() - 1),
    seeds           = 1:n_repeat,
    data_mode       = "clean",
    clusters        = NULL,
    outlier_frac    = 0.05,
    outlier_sd      = 0.05,
    flip_train_frac = list(pos = 0, neg = 0),             # ← 추가
    flip_test_frac  = list(pos = 0, neg = 0)              # ← 추가
) {
  if (length(seeds) != n_repeat)
    stop("seeds 길이가 n_repeat와 같아야 합니다.")
  
  outs <- foreach(s = seeds, .packages = c("dplyr","tibble")) %do% {
    message(sprintf("▶ 반복 %d/%d (seed=%d)", s, n_repeat, s))
    
    # 2) run_benchmark 호출 시 flip 인자 전달
    run_benchmark(
      n               = n_sample,
      test_frac       = test_frac,
      cv_folds        = cv_folds,
      n_cores         = n_cores,
      seed            = 100 + s,
      data_mode       = data_mode,
      clusters        = clusters,
      outlier_frac    = outlier_frac,
      outlier_sd      = outlier_sd,
      flip_train_frac = flip_train_frac,                  # ← 전달
      flip_test_frac  = flip_test_frac                   # ← 전달
    )
  }
  
  # 결과 결합 및 요약
  raw_all <- bind_rows(lapply(outs, `[[`, "metrics"), .id="rep")
  err_all <- bind_rows(lapply(outs, `[[`, "errors"),  .id="rep")
  
  summary_tbl <- raw_all %>%
    group_by(model) %>%
    summarise(
      n_ok     = sum(is.na(error)),
      acc_mean = mean(accuracy, na.rm=TRUE),
      acc_sd   = sd(accuracy,   na.rm=TRUE),
      f1_mean  = mean(f1,       na.rm=TRUE),
      f1_sd    = sd(f1,         na.rm=TRUE),
      .groups  = "drop"
    ) %>%
    arrange(desc(f1_mean))
  
  list(summary = summary_tbl,
       raw     = raw_all,
       errors  = err_all)
}


