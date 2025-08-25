########################################################################
#  loss_shifting.r   (2025-06-02)          
########################################################################

#################### (0) 패키지 #######################################
suppressPackageStartupMessages({
  library(quadprog)     # QP for dual SVM
})

#################### (1) RBF 커널 #####################################
rbf_kernel <- function(X1, X2 = NULL, sigma = 1) {
  if (is.null(X2)) X2 <- X1
  dist2 <- outer(rowSums(X1^2), rowSums(X2^2), "+") - 2 * tcrossprod(X1, X2)
  exp(-dist2 / (2 * sigma^2))
}

#################### (2) Shift 함수 ###################################
shift_r_none <- function(u) rep(0, length(u))
shift_r_soft <- function(u, alpha, eta)
  ifelse(u >= -eta, 0, (1 + alpha) * (u + eta))
shift_r_hard <- function(u, alpha, eta)
  ifelse(u >= -eta, 0, (1 + alpha) * u)

#################### (3) Base loss ####################################
loss_hinge    <- function(u) pmax(0, 1 - u)
loss_sqhinge  <- function(u) pmax(0, 1 - u)^2
loss_logistic <- function(u) log1p(exp(-u))
loss_exp      <- function(u) exp(-u)

#################### (4) 목적 함수 ####################################
objective_value <- function(theta, X, y, shift_fun, loss_fun, lambda) {
  X_aug <- cbind(1, X)
  f_val <- as.vector(X_aug %*% theta)
  u     <- y * f_val
  r_vec <- shift_fun(u)
  eff_m <- u - r_vec
  mean(loss_fun(eff_m)) + lambda * sum(theta[-1]^2) / 2
}
make_penalty_mat <- function(p, n, lambda) {
  diag(c(0, rep(n * lambda / 2, p)))
}

update_f_hinge_MM <- function(theta, X, y, r, lambda,
                              max_inner = 20, tol_inner = 1e-6, verbose_inner = FALSE) {
  n <- nrow(X); p <- ncol(X)
  X_aug <- cbind(1, X); eps <- 1e-4
  for (it in seq_len(max_inner)) {
    f_val <- as.vector(X_aug %*% theta)
    u0    <- y * (f_val - r)
    w_vec <- 1 / (4 * abs(1 - u0) + eps)
    t_vec <- y * (r + 1 + abs(1 - u0))
    WX    <- sqrt(w_vec) * X_aug
    Wt    <- sqrt(w_vec) * t_vec
    A     <- crossprod(WX) + make_penalty_mat(p, n, lambda)
    b     <- crossprod(WX, Wt)
    theta_new <- solve(A, b)
    if (verbose_inner) {
      cat(sprintf(" [hinge inner %d] |Δθ|₂ = %.3e\n",
                  it, sqrt(sum((theta_new - theta)^2))))
    }
    if (sqrt(sum((theta_new - theta)^2)) < tol_inner) {
      theta <- theta_new; break
    }
    theta <- theta_new
  }
  theta
}

update_f_sqhinge_MM <- function(theta, X, y, r, lambda,
                                max_inner = 20, tol_inner = 1e-6, verbose_inner = FALSE) {
  n <- nrow(X); p <- ncol(X)
  X_aug <- cbind(1, X)
  for (it in seq_len(max_inner)) {
    f_val <- as.vector(X_aug %*% theta)
    u0    <- y * (f_val - r)
    t_vec <- y * (r + pmax(1, u0))
    A     <- crossprod(X_aug) + make_penalty_mat(p, n, lambda)
    b     <- crossprod(X_aug, t_vec)
    theta_new <- solve(A, b)
    if (verbose_inner) {
      cat(sprintf(" [sq-hinge inner %d] |Δθ|₂ = %.3e\n",
                  it, sqrt(sum((theta_new - theta)^2))))
    }
    if (sqrt(sum((theta_new - theta)^2)) < tol_inner) {
      theta <- theta_new; break
    }
    theta <- theta_new
  }
  theta
}

# update_f_logistic_IRLS <- function(theta, X, y, r, lambda,
#                                    max_inner = 20, tol_inner = 1e-6, verbose_inner = FALSE) {
#   X_aug <- cbind(1, X); n <- nrow(X); eps_H <- 1e-4
#   for (it in seq_len(max_inner)) {
#     f_val <- as.vector(X_aug %*% theta)
#     u     <- y * (f_val - r)
#     sigma <- 1 / (1 + exp(u))
#     w_vec <- sigma * (1 - sigma)
#     grad  <- c(
#       (-1 / n) * sum(y * sigma),
#       (-1 / n) * (t(X) %*% (y * sigma)) + lambda * theta[-1]
#     )
#     W <- diag(as.vector(w_vec) + eps_H)
#     H <- (1 / n) * t(X_aug) %*% W %*% X_aug
#     diag(H)[-1] <- diag(H)[-1] + lambda
#     delta     <- solve(H, grad)
#     theta_new <- theta - delta
#     if (verbose_inner) {
#       cat(sprintf(" [logistic inner %d] ‖Δθ‖₂ = %.3e\n",
#                   it, sqrt(sum(delta^2))))
#     }
#     if (sqrt(sum(delta^2)) < tol_inner) {
#       theta <- theta_new; break
#     }
#     theta <- theta_new
#   }
#   theta
# }
update_f_logistic_IRLS <- function(theta, X, y, r, lambda,
                                   max_inner = 20, tol_inner = 1e-6,
                                   verbose_inner = FALSE) {

  X_aug   <- cbind(1, X)
  n       <- nrow(X)
  p       <- ncol(X)            # feature 개수 (절편 제외)
  w_min   <- 1e-8               ###—— MOD ——###  W 의 최소값
  eps_rd  <- 1e-6               ###—— MOD ——###  헤시안 릿지

  for (it in seq_len(max_inner)) {

    f_val <- as.vector(X_aug %*% theta)
    u     <- y * (f_val - r)

    sigma <- 1 / (1 + exp(u))
    w_vec <- pmax(sigma * (1 - sigma), w_min)   ###—— MOD ——###

    ## --- gradient --------------------------------------------------
    grad <- c(
      (-1 / n) * sum(y * sigma),
      (-1 / n) * (t(X) %*% (y * sigma)) + lambda * theta[-1]
    )

    ## --- Hessian ---------------------------------------------------
    W <- diag(as.vector(w_vec))
    H <- (1 / n) * t(X_aug) %*% W %*% X_aug +
         diag(c(eps_rd, rep(lambda + eps_rd, p)))   ###—— MOD ——###

    ## --- Newton step ----------------------------------------------
    delta     <- solve(H, grad)      # 재시도 삭제
    theta_new <- theta - delta

    if (verbose_inner)
      cat(sprintf(" [logistic inner %d] ‖Δθ‖₂ = %.3e\n",
                  it, sqrt(sum(delta^2))))

    if (sqrt(sum(delta^2)) < tol_inner) {
      theta <- theta_new; break
    }
    theta <- theta_new
  }
  theta
}
# update_f_exponential_IRLS <- function(theta, X, y, r, lambda,
#                                       max_inner = 20, tol_inner = 1e-6, verbose_inner = FALSE) {
#   X_aug <- cbind(1, X); n <- nrow(X); eps_H <- 1e-4
#   for (it in seq_len(max_inner)) {
#     f_val <- as.vector(X_aug %*% theta)
#     u     <- y * (f_val - r)
#     e_neg <- exp(-u)
#     w_vec <- e_neg
#     grad  <- c(
#       (-1 / n) * sum(e_neg * y),
#       (-1 / n) * (t(X) %*% (e_neg * y)) + lambda * theta[-1]
#     )
#     W <- diag(as.vector(w_vec) + eps_H)
#     H <- (1 / n) * t(X_aug) %*% W %*% X_aug
#     diag(H)[-1] <- diag(H)[-1] + lambda
#     delta     <- solve(H, grad)
#     theta_new <- theta - delta
#     if (verbose_inner) {
#       cat(sprintf(" [exp inner %d] ‖Δθ‖₂ = %.3e\n",
#                   it, sqrt(sum(delta^2))))
#     }
#     if (sqrt(sum(delta^2)) < tol_inner) {
#       theta <- theta_new; break
#     }
#     theta <- theta_new
#   }
#   theta
# }
update_f_exponential_IRLS <- function(theta, X, y, r, lambda,
                                      max_inner = 20, tol_inner = 1e-6,
                                      verbose_inner = FALSE) {

  X_aug   <- cbind(1, X)
  n       <- nrow(X)
  p       <- ncol(X)
  w_min   <- 1e-6               ###—— MOD ——###
  eps_rd  <- 1e-5               ###—— MOD ——###

  for (it in seq_len(max_inner)) {

    f_val <- as.vector(X_aug %*% theta)
    u     <- y * (f_val - r)

    e_neg <- exp(-u)
    w_vec <- pmax(e_neg, w_min)        ###—— MOD ——###

    ## --- gradient --------------------------------------------------
    grad <- c(
      (-1 / n) * sum(e_neg * y),
      (-1 / n) * (t(X) %*% (e_neg * y)) + lambda * theta[-1]
    )

    ## --- Hessian ---------------------------------------------------
    W <- diag(as.vector(w_vec))
    H <- (1 / n) * t(X_aug) %*% W %*% X_aug +
         diag(c(eps_rd, rep(lambda + eps_rd, p)))   ###—— MOD ——###

    ## --- Newton step ----------------------------------------------
    delta     <- solve(H, grad)
    theta_new <- theta - delta

    if (verbose_inner)
      cat(sprintf(" [exp inner %d] ‖Δθ‖₂ = %.3e\n",
                  it, sqrt(sum(delta^2))))

    if (sqrt(sum(delta^2)) < tol_inner) {
      theta <- theta_new; break
    }
    theta <- theta_new
  }
  theta
}
###########################################
# (6) Dual SVM (quadprog) 함수들
###########################################
dual_svm_hinge <- function(X, y, r, lambda) {
  n     <- nrow(X)
  C_val <- 1 / (2 * n * lambda)
  eps_v <- 1e-4
  K_mat <- X %*% t(X)
  Dmat  <- (y %*% t(y)) * K_mat + diag(eps_v, n)
  dvec  <- 1 + r
  Amat  <- t(rbind(y, diag(n), -diag(n)))
  bvec  <- c(0, rep(0, n), rep(-C_val, n))
  sol   <- solve.QP(Dmat, dvec, Amat, bvec, meq = 1)
  alpha <- sol$solution
  beta  <- t(X) %*% (alpha * y)
  sv    <- which(alpha > eps_v)
  bound <- which(alpha > eps_v & alpha < (C_val - eps_v))
  if (length(bound) > 0) {
    b0_vals <- sapply(bound, function(i) {
      y[i] * (1 + r[i]) - sum(alpha[sv] * y[sv] * as.numeric(X[i,] %*% t(X[sv,])))
    })
    b0 <- mean(b0_vals)
  } else b0 <- 0
  list(theta = c(b0, beta), alpha = alpha)
}

dual_svm_sqhinge <- function(X, y, r, lambda) {
  n     <- nrow(X)
  C_val <- 1 / (n * lambda)
  eps_v <- 1e-4
  K_mat <- X %*% t(X)
  Dmat  <- (y %*% t(y)) * K_mat + diag(1 / C_val, n) + diag(eps_v, n)
  dvec  <- 1 + r
  Amat  <- t(rbind(y, diag(n)))
  bvec  <- c(0, rep(0, n))
  sol   <- solve.QP(Dmat, dvec, Amat, bvec, meq = 1)
  alpha <- sol$solution
  beta  <- t(X) %*% (alpha * y)
  sv    <- which(alpha > eps_v)
  if (length(sv) > 0) {
    b0_vals <- sapply(sv, function(i) {
      sum(sapply(sv, function(j) {
        delta_ij <- ifelse(i == j, 1, 0)
        alpha[j] * y[j] * (as.numeric(X[i,] %*% X[j,]) + delta_ij / C_val)
      }))
    })
    b0 <- mean(y[sv] * (1 + r[sv]) - b0_vals)
  } else b0 <- 0
  list(theta = c(b0, beta), alpha = alpha)
}


#################### (6) 라인 서치 ###################################
line_search_choice <- function(theta_old, theta_raw,
                               X, y,
                               shift_fun, loss_fun, lambda,
                               step_set = 2 ^ c(-2, -1, 0, 1, 2)) {
  
  ## ① 후보 θ 생성
  candidates <- lapply(step_set,
                       function(s) theta_old + s * (theta_raw - theta_old))
  
  ## ② 각 후보의 목적함수 값 평가
  objs <- vapply(
    candidates,
    FUN = function(th) objective_value(
      th,
      X         = X,
      y         = y,
      shift_fun = shift_fun,
      loss_fun  = loss_fun,
      lambda    = lambda),
    FUN.VALUE = numeric(1L)
  )
  
  ## ③ 최소 목적함수 값을 주는 θ 반환
  candidates[[which.min(objs)]]
}


#################### (8) 주 학습 함수 ################################
loss_shift_classifier_all <- function(
    X, y,
    base_loss = c("hinge", "sqhinge", "logistic", "exp"),
    style     = c("none", "soft", "hard"),
    kernel    = c("linear", "gaussian"),
    sigma = 1, alpha = 1, eta = 0.5,
    lambda = 0.1, max_iter = 100, tol = 1e-7,
    max_inner = 50, svm_dual = FALSE,
    init_theta = NULL,
    line_search = TRUE,
    step_set = 2^c(-2, -1, 0, 1),
    verbose = FALSE
) {
  ## ----------- 데이터 준비 -----------------------------------------
  X_mat <- as.matrix(X)
  y_vec <- ifelse(y > 0, 1, -1)
  
  base_loss <- match.arg(base_loss)
  style     <- match.arg(style)
  kernel    <- match.arg(kernel)
  use_dual  <- svm_dual && base_loss %in% c("hinge", "sqhinge")
  
  model_desc <- paste(base_loss, style,
                      if (use_dual) "dual" else "primal",
                      kernel, sep = "_")
  if (verbose) cat(sprintf("\n\n[START_Fitting_Model] %s\n", model_desc))
  
  ## ----------- 커널 변환 (gaussian) --------------------------------
  if (kernel == "gaussian") {
    K_train <- rbf_kernel(X_mat, NULL, sigma)
    eig     <- eigen(K_train, symmetric = TRUE)
    vals    <- pmax(eig$values, .Machine$double.eps)
    U       <- eig$vectors
    D_half  <- diag(sqrt(vals))
    X_feat  <- U %*% D_half
  } else {
    X_feat <- X_mat
  }
  
  ## ----------- 함수 핸들러 ----------------------------------------
  shift_fun <- switch(style,
                      none = shift_r_none,
                      soft = function(u) shift_r_soft(u, alpha, eta),
                      hard = function(u) shift_r_hard(u, alpha, eta)
  )
  loss_fun <- switch(base_loss,
                     hinge    = loss_hinge,
                     sqhinge  = loss_sqhinge,
                     logistic = loss_logistic,
                     exp      = loss_exp
  )
  updater <- switch(base_loss,
                    hinge    = update_f_hinge_MM,
                    sqhinge  = update_f_sqhinge_MM,
                    logistic = update_f_logistic_IRLS,
                    exp      = update_f_exponential_IRLS
  )
  
  ## ----------- 초기화 ---------------------------------------------
  n <- nrow(X_feat); p <- ncol(X_feat)
  if (!is.null(init_theta) && length(init_theta) == p + 1) {
    theta <- init_theta
  } else {
    theta <- rep(0, p + 1)
  }
  obj_old <- objective_value(theta, X_feat, y_vec,
                             shift_fun, loss_fun, lambda)
  iter_used <- NA
  
  ## ----------- 반복 ------------------------------------------------
  for (it in seq_len(max_iter)) {
    # ① shift 벡터 r 계산
    X_aug <- cbind(1, X_feat)
    f_val <- as.vector(X_aug %*% theta)
    u_vec <- y_vec * f_val
    r_vec <- shift_fun(u_vec)
    
    # ② theta_raw 업데이트 (primal or dual)
    if (use_dual) {
      sol <- if (base_loss == "hinge") dual_svm_hinge(X_feat, y_vec, r_vec, lambda)
      else                       dual_svm_sqhinge(X_feat, y_vec, r_vec, lambda)
      theta_raw <- sol$theta
    } else {
      theta_raw <- updater(theta, X_feat, y_vec, r_vec,
                           lambda, max_inner, tol)
    }
    
    # ③ Exact line-search (style ≠ "none" & alpha>0 & primal)
    if (line_search && style != "none" && !use_dual) {
      theta_new <- line_search_choice(theta, theta_raw, X_feat, y_vec,
                                      shift_fun, loss_fun, lambda, step_set)
    } else {
      theta_new <- theta_raw
    }
    
    # ④ 수렴 검사
    obj_new <- objective_value(theta_new, X_feat, y_vec,
                               shift_fun, loss_fun, lambda)
    rel <- abs(obj_new - obj_old) / (abs(obj_old) + 1e-8)
    if (verbose)
      cat(sprintf("[%s] Iter %3d/%3d  rel=%.2e\n",
                  model_desc, it, max_iter, rel))
    
    theta    <- theta_new
    obj_old  <- obj_new
    if (rel < tol) { iter_used <- it; break }
    if (it == max_iter) iter_used <- it
  }
  if (verbose) cat(sprintf("[DONE_Fitting_Model] %s in %d iters\n\n", model_desc, iter_used))
  
  ## ----------- 결과 객체 ------------------------------------------
  out <- list(theta = theta, base_loss = base_loss,
              style = style, kernel = kernel,
              sigma = sigma, alpha = alpha, eta = eta,
              lambda = lambda, iter_used = iter_used,
              step_set = step_set)
  if (kernel == "gaussian") {
    out$U            <- U
    out$D_inv_sqrt   <- 1 / sqrt(vals)
    out$X_train_orig <- X_mat
  }
  out
}

#################### (9) 예측 ########################################
predict_loss_shift <- function(model, X_new) {
  if (model$kernel == "gaussian") {
    K_new <- rbf_kernel(as.matrix(X_new),
                        model$X_train_orig, model$sigma)
    Z_new <- K_new %*% model$U %*% diag(model$D_inv_sqrt)
    X_aug <- cbind(1, Z_new)
  } else {
    X_aug <- cbind(1, as.matrix(X_new))
  }
  f_val <- as.vector(X_aug %*% model$theta)
  ifelse(f_val >= 0, 1, -1)
}

#################### (10) 평가 지표 ###################################
evaluate_metrics <- function(truth, pred) {
  cm <- table(factor(truth, levels = c(-1, 1)),
              factor(pred,  levels = c(-1, 1)))
  acc <- sum(diag(cm)) / sum(cm)
  prec <- if ((cm[2, 2] + cm[1, 2]) == 0) NA else cm[2, 2] / (cm[2, 2] + cm[1, 2])
  rec  <- if ((cm[2, 2] + cm[2, 1]) == 0) NA else cm[2, 2] / (cm[2, 2] + cm[2, 1])
  f1   <- if (is.na(prec) || is.na(rec) || (prec + rec) == 0) NA else
    2 * prec * rec / (prec + rec)
  list(accuracy = acc, precision = prec, recall = rec, f1 = f1, confusion = cm)
}

#################### (11) CV 분할 ####################################
cv_split <- function(n, n_folds, strat_y = NULL, val_frac = 0.2) {
  if (n_folds > 1) {
    if (is.null(strat_y)) strat_y <- rep(1, n)
    folds <- rep(NA_integer_, n)
    for (cl in unique(strat_y)) {
      idx <- which(strat_y == cl)
      folds[idx] <- sample(rep(seq_len(n_folds), length.out = length(idx)))
    }
    lapply(seq_len(n_folds), function(k) list(
      tr = which(folds != k), va = which(folds == k)))
  } else if (n_folds == 1) {
    va <- sample(n, floor(val_frac * n))
    list(list(tr = setdiff(seq_len(n), va), va = va))
  } else {
    list(list(tr = seq_len(n), va = integer(0)))
  }
}

#################### (12) CV + restart 정확도 #########################
cv_accuracy_restart <- function(
    X, y, folds,
    base_loss, style, kernel,
    lambda, sigma, alpha, eta,
    restarts, svm_dual, max_iter,
    line_search, step_set) {
  
  best_acc <- -Inf
  p_feat <- if (kernel == "gaussian") nrow(X) else ncol(X)
  
  for (r in seq_len(restarts)) {
    init_th <- rnorm(p_feat + 1, 0, 0.1)
    accs <- numeric(length(folds))
    for (i in seq_along(folds)) {
      tr <- folds[[i]]$tr; va <- folds[[i]]$va
      mod <- loss_shift_classifier_all(
        X[tr, ], y[tr],
        base_loss = base_loss, style = style, kernel = kernel,
        sigma = sigma, alpha = alpha, eta = eta, lambda = lambda,
        svm_dual = svm_dual, max_iter = max_iter,
        line_search = line_search, step_set = step_set,
        init_theta = init_th,
        verbose = FALSE)
      if (length(va) == 0) { accs[i] <- NA } else {
        preds <- predict_loss_shift(mod, X[va, ])
        accs[i] <- mean(preds == y[va])
      }
    }
    best_acc <- max(best_acc, mean(accs, na.rm = TRUE))
  }
  best_acc
}

#################### (13) 풀 그리드 튜너 ##############################
fullgrid_tune_loss_shift <- function(
    X, y,
    base_loss = "hinge", style = "none", kernel = "linear",
    lambda_grid, sigma_grid, alpha_grid, eta_grid,
    restarts = 5,
    n_folds = 5, val_frac = 0.2,
    svm_dual = FALSE, max_iter = 100,
    line_search = TRUE, step_set = 2^c(-2, -1, 0, 1),
    verbose = FALSE) {
  
  grid <- expand.grid(lambda = lambda_grid,
                      sigma  = if (kernel == "gaussian") sigma_grid else 1,
                      alpha  = if (style == "none") 1 else alpha_grid,
                      eta    = if (style == "none") 0 else eta_grid)
  folds <- cv_split(nrow(X), n_folds, strat_y = y, val_frac = val_frac)
  
  acc_vec <- numeric(nrow(grid))
  for (i in seq_len(nrow(grid))) {
    acc_vec[i] <- cv_accuracy_restart(
      X, y, folds,
      base_loss, style, kernel,
      lambda = grid$lambda[i],
      sigma  = grid$sigma[i],
      alpha  = grid$alpha[i],
      eta    = grid$eta[i],
      restarts = restarts,
      svm_dual = svm_dual, max_iter = max_iter,
      line_search = line_search, step_set = step_set)
    ########################################################################
    model_desc_verbose <- paste(base_loss, style,
                            if (svm_dual) "dual" else "primal",
                            kernel, sep = "_")
    if (verbose) 
    cat(sprintf("\n[START Tuning] %s\n", model_desc_verbose))
    cat(sprintf(
      "(lambda=%.3g, sigma=%.3g, alpha=%.3g, eta=%.3g)  cv-acc=%.4f  [%d/%d]\n",
      grid$lambda[i], grid$sigma[i],
      grid$alpha[i],  grid$eta[i],
      acc_vec[i], i, nrow(grid)))
    cat(sprintf("[END Tuning] %s\n", model_desc_verbose))
  }
  
  best_idx <- which.max(acc_vec)
  list(best_pars = grid[best_idx, , drop = FALSE],
       best_cv_acc = acc_vec[best_idx],
       cv_table = cbind(grid, cv_acc = acc_vec))
}

#################### (14) 전용 Fit 함수 ###############################
fit_loss_shift <- function(
    X, y,
    base_loss = "hinge", style = "none", kernel = "gaussian",
    lambda_grid = 10^seq(-3, 1, len = 3),
    sigma_grid  = c(0.1, 1, 10),
    alpha_grid  = c(0.5, 1, 2),
    eta_grid    = c(0.1, 0.5, 1),
    restarts    = 5,
    n_folds     = 5, val_frac = 0.2,
    svm_dual    = FALSE, max_iter = 100,
    line_search = TRUE, step_set = 2^c(-2, -1, 0, 1),
    verbose = FALSE) {
  
  tune <- fullgrid_tune_loss_shift(
    X, y, base_loss, style, kernel,
    lambda_grid, sigma_grid, alpha_grid, eta_grid,
    restarts, n_folds, val_frac,
    svm_dual, max_iter, line_search, step_set, verbose)
  
  bp <- tune$best_pars
  if (verbose) {
    cat("\n[fit_loss_shift] best hyper-parameters\n"); print(bp)
    cat(sprintf("CV accuracy = %.4f\n\n", tune$best_cv_acc))
  }
  loss_shift_classifier_all(
    X, y, base_loss, style, kernel,
    sigma  = bp$sigma, alpha = bp$alpha, eta = bp$eta, lambda = bp$lambda,
    max_iter = max_iter, svm_dual = svm_dual,
    line_search = line_search, step_set = step_set,
    verbose = verbose)
}

#################### (15) 통합 Wrapper ###############################
train_loss_shift <- function(
    X_train, y_train,
    X_test = NULL,                      # test 데이터가 있으면 예측까지
    base_loss = c("hinge", "sqhinge", "logistic", "exp"),
    style     = c("none", "soft", "hard"),
    kernel    = c("linear", "gaussian"),
    lambda_grid = 10^seq(-3, 1, len = 5),
    sigma_grid  = c(0.1, 1, 10),
    alpha_grid  = c(0.5, 1, 2),
    eta_grid    = c(0.1, 0.5, 1),
    restarts    = 5,
    n_folds     = 5,
    svm_dual    = FALSE,
    max_iter    = 100,
    line_search = TRUE,
    step_set    = 2^c(-2, -1, 0, 1),
    verbose     = FALSE) {
  
  ## 1) 하이퍼파라미터 튜닝
  tune_out <- fullgrid_tune_loss_shift(
    X_train, y_train,
    base_loss = match.arg(base_loss),
    style     = match.arg(style),
    kernel    = match.arg(kernel),
    lambda_grid = lambda_grid,
    sigma_grid  = sigma_grid,
    alpha_grid  = alpha_grid,
    eta_grid    = eta_grid,
    restarts    = restarts,
    n_folds     = n_folds,
    svm_dual    = svm_dual,
    max_iter    = max_iter,
    line_search = line_search,
    step_set    = step_set,
    verbose     = verbose)
  
  best <- tune_out$best_pars
  
  ## 2) 최적 하이퍼파라미터로 전체 train 재학습
  final_model <- loss_shift_classifier_all(
    X_train, y_train,
    base_loss = match.arg(base_loss),
    style     = match.arg(style),
    kernel    = match.arg(kernel),
    sigma  = best$sigma,
    alpha  = best$alpha,
    eta    = best$eta,
    lambda = best$lambda,
    max_iter   = max_iter,
    svm_dual   = svm_dual,
    line_search = line_search,
    step_set    = step_set,
    verbose     = verbose)
  
  ## 3) (선택) test 예측
  test_pred <- if (!is.null(X_test))
    predict_loss_shift(final_model, X_test) else NULL
  
  ## 4) 결과 반환
  list(
    model       = final_model,
    best_param  = best,
    cv_table    = tune_out$cv_table,
    cv_best_acc = tune_out$best_cv_acc,
    test_pred   = test_pred)
}