########################################################################
#  loss_shifting_v3.R   (2025-08-14)
#  - 'warm-start' 용어를 'pre_trained_initial'로 변경
#  - linear 커널에서도 자동 표준화 적용 (범용 'standardize' 옵션)
#  - sigma grid: median heuristic on z-scored X
#  - reproducibility, Gaussian fold-dim init, final restarts kept
########################################################################

suppressPackageStartupMessages({ library(quadprog) })

#################### (1) RBF 커널 #####################################
rbf_kernel <- function(X1, X2 = NULL, sigma = 1) {
  if (is.null(X2)) X2 <- X1
  dist2 <- outer(rowSums(X1^2), rowSums(X2^2), "+") - 2 * tcrossprod(X1, X2)
  exp(-dist2 / (2 * sigma^2))
}

#################### (2) Shift 함수 ###################################
shift_r_none <- function(u) rep(0, length(u))
shift_r_soft <- function(u, alpha, eta) ifelse(u >= -eta, 0, (1 + alpha) * (u + eta))
shift_r_hard <- function(u, alpha, eta) ifelse(u >= -eta, 0, (1 + alpha) * u)

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
make_penalty_mat <- function(p, n, lambda) diag(c(0, rep(n * lambda / 2, p)))

update_f_hinge_MM <- function(theta, X, y, r, lambda,
                              max_inner = 20, tol_inner = 1e-6, verbose_inner = FALSE) {
  n <- nrow(X); p <- ncol(X)
  X_aug <- cbind(1, X); eps <- 1e-4
  for (it in seq_len(max_inner)) {
    f_val <- as.vector(X_aug %*% theta)
    #u0    <- y * (f_val - r) ############################################################ 에러 의심
    u0 <- y * f_val - r
    w_vec <- 1 / (4 * abs(1 - u0) + eps)
    t_vec <- y * (r + 1 + abs(1 - u0))
    WX    <- sqrt(w_vec) * X_aug
    Wt    <- sqrt(w_vec) * t_vec
    A     <- crossprod(WX) + make_penalty_mat(p, n, lambda)
    b     <- crossprod(WX, Wt)
    theta_new <- solve(A, b)
    if (verbose_inner) cat(sprintf(" [hinge inner %d] |Δθ|₂ = %.3e\n", it, sqrt(sum((theta_new - theta)^2))))
    if (sqrt(sum((theta_new - theta)^2)) < tol_inner) { theta <- theta_new; break }
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
    #u0    <- y * (f_val - r) ########################################################에러 의심
    u0 <- y * f_val - r
    t_vec <- y * (r + pmax(1, u0))
    A     <- crossprod(X_aug) + make_penalty_mat(p, n, lambda)
    b     <- crossprod(X_aug, t_vec)
    theta_new <- solve(A, b)
    if (verbose_inner) cat(sprintf(" [sq-hinge inner %d] |Δθ|₂ = %.3e\n", it, sqrt(sum((theta_new - theta)^2))))
    if (sqrt(sum((theta_new - theta)^2)) < tol_inner) { theta <- theta_new; break }
    theta <- theta_new
  }
  theta
}

update_f_logistic_IRLS <- function(theta, X, y, r, lambda,
                                   max_inner = 20, tol_inner = 1e-6, verbose_inner = FALSE) {
  X_aug <- cbind(1, X); n <- nrow(X); p <- ncol(X)
  w_min <- 1e-5; eps_rd <- 1e-4 # 안정성을 위해 1e-5 -> 1e-4
  for (it in seq_len(max_inner)) {
    f_val <- as.vector(X_aug %*% theta)
    #u     <- y * (f_val - r) #################################################에러의심
    #sigma <- 1 / (1 + exp(u))#################################################에러의심
    u_eff <- y * f_val - r
    sigma <- 1 / (1 + exp(u_eff))
    
    w_vec <- pmax(sigma * (1 - sigma), w_min)
    grad  <- c((-1 / n) * sum(y * sigma),
               (-1 / n) * (t(X) %*% (y * sigma)) + lambda * theta[-1])
    W <- diag(as.vector(w_vec))
    H <- (1 / n) * t(X_aug) %*% W %*% X_aug + diag(c(eps_rd, rep(lambda + eps_rd, p)))
    delta <- solve(H, grad); theta_new <- theta - delta
    if (verbose_inner) cat(sprintf(" [logistic inner %d] ‖Δθ‖₂ = %.3e\n", it, sqrt(sum(delta^2))))
    if (sqrt(sum(delta^2)) < tol_inner) { theta <- theta_new; break }
    theta <- theta_new
  }
  theta
}

update_f_exponential_IRLS <- function(theta, X, y, r, lambda,
                                      max_inner = 20, tol_inner = 1e-6, verbose_inner = FALSE) {
  X_aug <- cbind(1, X); n <- nrow(X); p <- ncol(X)
  w_min <- 1e-5; eps_rd <- 1e-4 # 안정성을 위해 1e-5 -> 1e-4
  for (it in seq_len(max_inner)) {
    f_val <- as.vector(X_aug %*% theta)
    #u     <- y * (f_val - r) #################################################에러의심
    #e_neg <- exp(-u) #################################################에러의심
    u_eff <- y * f_val - r
    e_neg <- exp(-u_eff)
    #u_eff_capped <- pmax(u_eff, -700) 
    #e_neg <- exp(-u_eff_capped) ###############################필요할까?오버플로우 막기
    w_vec <- pmax(e_neg, w_min)
    grad  <- c((-1 / n) * sum(e_neg * y),
               (-1 / n) * (t(X) %*% (e_neg * y)) + lambda * theta[-1])
    W <- diag(as.vector(w_vec))
    H <- (1 / n) * t(X_aug) %*% W %*% X_aug + diag(c(eps_rd, rep(lambda + eps_rd, p)))
    delta <- solve(H, grad); theta_new <- theta - delta
    if (verbose_inner) cat(sprintf(" [exp inner %d] ‖Δθ‖₂ = %.3e\n", it, sqrt(sum(delta^2))))
    if (sqrt(sum(delta^2)) < tol_inner) { theta <- theta_new; break }
    theta <- theta_new
  }
  theta
}

###########################################
# (6) Dual SVM (quadprog)
###########################################
dual_svm_hinge <- function(X, y, r, lambda) {
  n <- nrow(X); C_val <- 1 / (2 * n * lambda); eps_v <- 1e-4
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
  n <- nrow(X); C_val <- 1 / (n * lambda); eps_v <- 1e-4
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

#################### (7) 라인 서치 ###################################
line_search_choice <- function(theta_old, theta_raw, X, y, shift_fun, loss_fun, lambda,
                               step_set = 2 ^ c(-2, -1, 0, 1, 2)) {
  candidates <- lapply(step_set, function(s) theta_old + s * (theta_raw - theta_old))
  objs <- vapply(candidates, function(th)
    objective_value(th, X, y, shift_fun, loss_fun, lambda), numeric(1L))
  candidates[[which.min(objs)]]
}

#################### (helper) z-score #################################
.standardize_fit <- function(X) {
  mu <- colMeans(X)
  sd <- apply(X, 2, sd); sd[sd == 0] <- 1
  Xs <- sweep(sweep(X, 2, mu, "-"), 2, sd, "/")
  list(Xs = Xs, mu = mu, sd = sd)
}
.standardize_apply <- function(X, mu, sd) {
  sd[sd == 0] <- 1
  sweep(sweep(X, 2, mu, "-"), 2, sd, "/")
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
    verbose = FALSE,
    standardize = TRUE
) {
  X_mat <- as.matrix(X)
  y_vec <- ifelse(y > 0, 1, -1)
  base_loss <- match.arg(base_loss)
  style     <- match.arg(style)
  kernel    <- match.arg(kernel)
  use_dual  <- svm_dual && base_loss %in% c("hinge", "sqhinge")
  
  # 먼저 표준화를 수행 (범용)
  if (standardize) {
    st <- .standardize_fit(X_mat)
    X_proc <- st$Xs
  } else {
    st <- NULL
    X_proc <- X_mat
  }
  
  # 커널에 따라 피처 공간(X_feat)을 결정
  if (kernel == "gaussian") {
    K_train <- rbf_kernel(X_proc, NULL, sigma) # 표준화된 X_proc 사용
    eig     <- eigen(K_train, symmetric = TRUE)
    vals    <- pmax(eig$values, .Machine$double.eps)
    U       <- eig$vectors
    D_half  <- diag(sqrt(vals))
    X_feat  <- U %*% D_half
  } else { # 리니어 커널
    X_feat <- X_proc # 표준화된 X_proc를 그대로 사용
  }
  
  shift_fun <- switch(style,
                      none = shift_r_none,
                      soft = function(u) shift_r_soft(u, alpha, eta),
                      hard = function(u) shift_r_hard(u, alpha, eta))
  loss_fun <- switch(base_loss,
                     hinge = loss_hinge, sqhinge = loss_sqhinge,
                     logistic = loss_logistic, exp = loss_exp)
  updater <- switch(base_loss,
                    hinge = update_f_hinge_MM, sqhinge = update_f_sqhinge_MM,
                    logistic = update_f_logistic_IRLS, exp = update_f_exponential_IRLS)
  
  n <- nrow(X_feat); p <- ncol(X_feat)
  
  theta <- if (!is.null(init_theta) && length(init_theta) == p + 1) init_theta else rep(0, p + 1)
  
  obj_old <- objective_value(theta, X_feat, y_vec, shift_fun, loss_fun, lambda)
  iter_used <- NA
  for (it in seq_len(max_iter)) {
    X_aug <- cbind(1, X_feat)
    f_val <- as.vector(X_aug %*% theta)
    u_vec <- y_vec * f_val
    r_vec <- shift_fun(u_vec)
    
    theta_raw <- if (use_dual) {
      sol <- if (base_loss == "hinge") dual_svm_hinge(X_feat, y_vec, r_vec, lambda)
      else                       dual_svm_sqhinge(X_feat, y_vec, r_vec, lambda)
      sol$theta
    } else {
      updater(theta, X_feat, y_vec, r_vec, lambda, max_inner, tol)
    }
    
    theta_new <- if (line_search && !use_dual)
      line_search_choice(theta, theta_raw, X_feat, y_vec, shift_fun, loss_fun, lambda, step_set)
    else theta_raw
    
    obj_new <- objective_value(theta_new, X_feat, y_vec, shift_fun, loss_fun, lambda)
    rel <- abs(obj_new - obj_old) / (abs(obj_old) + 1e-8)
    if (verbose) cat(sprintf("[Iter %3d/%3d] rel=%.2e\n", it, max_iter, rel))
    theta <- theta_new; obj_old <- obj_new
    if (rel < tol) { iter_used <- it; break }
    if (it == max_iter) iter_used <- it
  }
  
  out <- list(theta = theta, base_loss = base_loss, style = style, kernel = kernel,
              sigma = sigma, alpha = alpha, eta = eta, lambda = lambda,
              iter_used = iter_used, step_set = step_set,
              standardize = standardize)
  
  # 표준화 정보를 저장 (두 커널 모두에 해당)
  if (isTRUE(standardize)) {
    out$center <- st$mu
    out$scale <- st$sd
  }
  
  if (kernel == "gaussian") {
    out$U <- U; out$D_inv_sqrt <- 1 / sqrt(vals); out$X_train_orig <- X_mat
  }
  out
}

#################### (9) 예측 ########################################
predict_loss_shift <- function(model, X_new, type = "class") {
  X_new_mat <- as.matrix(X_new)
  
  # Step 1: 모델이 표준화를 사용했다면 새로운 데이터도 동일하게 표준화
  if (isTRUE(model$standardize)) {
    X_proc <- .standardize_apply(X_new_mat, model$center, model$scale)
  } else {
    X_proc <- X_new_mat
  }
  
  # Step 2: 커널에 따라 예측 피처 공간을 구성
  if (model$kernel == "gaussian") {
    X_tr <- model$X_train_orig
    
    # 가우시안 커널 계산을 위해 학습 데이터도 표준화
    if (isTRUE(model$standardize)) {
      X_tr_proc <- .standardize_apply(X_tr, model$center, model$scale)
    } else {
      X_tr_proc <- X_tr
    }
    
    K_new <- rbf_kernel(X_proc, X_tr_proc, model$sigma)
    Z_new <- K_new %*% model$U %*% diag(model$D_inv_sqrt)
    X_aug <- cbind(1, Z_new)
  } else { # 리니어 커널
    X_aug <- cbind(1, X_proc)
  }
  
  # Step 3: 최종 점수 및 클래스 계산
  f_val <- as.vector(X_aug %*% model$theta)
  
  if (type == "score") {
    return(f_val)
  }
  ifelse(f_val >= 0, 1, -1)
}


#################### (10) 평가 지표 ###################################
evaluate_metrics <- function(truth, pred) {
  cm <- table(factor(truth, levels = c(-1, 1)),
              factor(pred,  levels = c(-1, 1)))
  acc <- sum(diag(cm)) / sum(cm)
  prec <- if ((cm[2, 2] + cm[1, 2]) == 0) NA else cm[2, 2] / (cm[2, 2] + cm[1, 2])
  rec  <- if ((cm[2, 2] + cm[2, 1]) == 0) NA else cm[2, 2] / (cm[2, 2] + cm[2, 1])
  f1   <- if (is.na(prec) || is.na(rec) || (prec + rec) == 0) NA else 2 * prec * rec / (prec + rec)
  list(accuracy = acc, precision = prec, recall = rec, f1 = f1, confusion = cm)
}

#################### (11) CV 분할 (재현성) ###########################
cv_split <- function(n, n_folds, strat_y = NULL, val_frac = 0.2, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  if (n_folds > 1) {
    if (is.null(strat_y)) strat_y <- rep(1, n)
    folds <- rep(NA_integer_, n)
    for (cl in unique(strat_y)) {
      idx <- which(strat_y == cl)
      folds[idx] <- sample(rep(seq_len(n_folds), length.out = length(idx)))
    }
    lapply(seq_len(n_folds), function(k) list(tr = which(folds != k), va = which(folds == k)))
  } else if (n_folds == 1) {
    va <- sample(n, floor(val_frac * n))
    list(list(tr = setdiff(seq_len(n), va), va = va))
  } else list(list(tr = seq_len(n), va = integer(0)))
}

#################### (11.5) sigma grid (on z-scored X) ################
.sigma_grid_from_median_scaled <- function(X, sigma_mult = c(0.5,1,2), m_max = 400, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  Xs <- scale(as.matrix(X))
  n  <- nrow(Xs)
  m  <- min(n, m_max)
  idx <- if (n > m) sample(n, m) else seq_len(n)
  D  <- as.matrix(dist(Xs[idx, , drop = FALSE]))
  med2 <- median(D[upper.tri(D)]^2)
  sigma_med <- sqrt(pmax(med2, .Machine$double.eps) / 2)
  sort(unique(sigma_med * sigma_mult))
}

#################### (12) CV + restart 정확도 #########################
cv_accuracy_restart <- function(
    X, y, folds,
    base_loss, style, kernel,
    lambda, sigma, alpha, eta,
    restarts, svm_dual, max_iter,
    line_search, step_set,
    base_seed = NULL) {
  
  best_acc <- -Inf
  for (r in seq_len(restarts)) {
    accs <- numeric(length(folds))
    for (i in seq_along(folds)) {
      tr <- folds[[i]]$tr; va <- folds[[i]]$va
      p_tr <- if (kernel == "gaussian") length(tr) else ncol(X)
      if (!is.null(base_seed)) set.seed(base_seed + r * 10000 + i)
      init_th <- rnorm(p_tr + 1, 0, 0.1)
      mod <- loss_shift_classifier_all(
        X[tr, ], y[tr],
        base_loss = base_loss, style = style, kernel = kernel,
        sigma = sigma, alpha = alpha, eta = eta, lambda = lambda,
        svm_dual = svm_dual, max_iter = max_iter,
        line_search = line_search, step_set = step_set,
        init_theta = init_th, verbose = FALSE,
        standardize = TRUE)
      accs[i] <- if (length(va) == 0) NA else mean(predict_loss_shift(mod, X[va, ]) == y[va])
    }
    best_acc <- max(best_acc, mean(accs, na.rm = TRUE))
  }
  best_acc
}

#################### (13) 풀 그리드 튜너 ################################
fullgrid_tune_loss_shift <- function(
    X, y,
    base_loss = "hinge", style = "none", kernel = "linear",
    lambda_grid, sigma_grid = NULL, alpha_grid, eta_grid,
    restarts = 5,
    n_folds = 5, val_frac = 0.2,
    svm_dual = FALSE, max_iter = 100,
    line_search = TRUE, step_set = 2^c(-2, -1, 0, 1),
    verbose = FALSE,
    seed = NULL,
    sigma_mult = c(0.5,1,2)) {
  
  # sigma grid from z-scored X
  sigma_grid_eff <- if (kernel == "gaussian")
    .sigma_grid_from_median_scaled(X, sigma_mult = sigma_mult, seed = if (is.null(seed)) NULL else seed + 777)
  else 1
  
  grid <- expand.grid(lambda = lambda_grid,
                      sigma  = sigma_grid_eff,
                      alpha  = if (style == "none") 1 else alpha_grid,
                      eta    = if (style == "none") 0 else eta_grid)
  
  folds <- cv_split(nrow(X), n_folds, strat_y = y, val_frac = val_frac, seed = seed)
  
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
      line_search = line_search, step_set = step_set,
      base_seed = if (is.null(seed)) NULL else seed + i * 1000)
    if (verbose) {
      model_desc_verbose <- paste(base_loss, style, if (svm_dual) "dual" else "primal", kernel, sep = "_")
      cat(sprintf("\n[START Tuning] %s\n", model_desc_verbose))
      cat(sprintf("(lambda=%.3g, sigma=%.3g, alpha=%.3g, eta=%.3g)  cv-acc=%.4f  [%d/%d]\n",
                  grid$lambda[i], grid$sigma[i], grid$alpha[i], grid$eta[i], acc_vec[i], i, nrow(grid)))
      cat(sprintf("[END Tuning] %s\n", model_desc_verbose))
    } else {
      cat(sprintf("(lambda=%.3g, sigma=%.3g, alpha=%.3g, eta=%.3g)  cv-acc=%.4f  [%d/%d]\n",
                  grid$lambda[i], grid$sigma[i], grid$alpha[i], grid$eta[i], acc_vec[i], i, nrow(grid)))
    }
  }
  
  best_idx <- which.max(acc_vec)
  list(best_pars = grid[best_idx, , drop = FALSE],
       best_cv_acc = acc_vec[best_idx],
       cv_table = cbind(grid, cv_acc = acc_vec))
}

#################### (13.5) 목적함수 유틸 ################################
.compute_obj_for_model <- function(model, X, y) {
  y_vec <- ifelse(y > 0, 1, -1)
  shift_fun <- switch(model$style,
                      none = shift_r_none,
                      soft = function(u) shift_r_soft(u, model$alpha, model$eta),
                      hard = function(u) shift_r_hard(u, model$alpha, model$eta))
  loss_fun <- switch(model$base_loss,
                     hinge = loss_hinge, sqhinge = loss_sqhinge,
                     logistic = loss_logistic, exp = loss_exp)
  if (model$kernel == "gaussian") {
    D_half <- diag(1 / model$D_inv_sqrt)   # = diag(sqrt(vals))
    X_feat <- model$U %*% D_half
  } else {
    if (isTRUE(model$standardize)) {
      X_feat <- .standardize_apply(as.matrix(X), model$center, model$scale)
    } else {
      X_feat <- as.matrix(X)
    }
  }
  objective_value(model$theta, X_feat, y_vec, shift_fun, loss_fun, model$lambda)
}

#################### (14) 전용 Fit (최종 restart 적용) ##################
fit_loss_shift <- function(
    X, y,
    base_loss = "hinge", style = "none", kernel = "gaussian",
    lambda_grid = 2^seq(-12, 8, 1),
    sigma_grid  = NULL,              # ignored for gaussian
    alpha_grid  = c(0.1, 0.25, 0.5, 1, 2, 4),
    eta_grid    = c(0.25, 0.5, 0.75, 1, 1.5, 2, 3),
    restarts    = 10,
    n_folds     = 5, val_frac = 0.2,
    svm_dual    = FALSE, max_iter = 100,
    line_search = TRUE, step_set = 2^c(-3, -2, -1, 0, 1, 2),
    verbose     = FALSE,
    seed        = NULL,
    sigma_mult  = c(0.125,0.25,0.5,1,2,),
    init_theta  = NULL
) {
  
  if (!is.null(seed)) set.seed(seed)
  
  tune <- fullgrid_tune_loss_shift(
    X, y, base_loss, style, kernel,
    lambda_grid, sigma_grid = NULL, alpha_grid, eta_grid,
    restarts, n_folds, val_frac,
    svm_dual, max_iter, line_search, step_set, verbose,
    seed = if (is.null(seed)) NULL else seed + 123,
    sigma_mult = sigma_mult)
  
  bp <- tune$best_pars
  if (verbose) { cat("\n[fit_loss_shift] best hyper-parameters\n"); print(bp)
    cat(sprintf("CV accuracy = %.4f\n\n", tune$best_cv_acc)) }
  
  best_mod <- NULL; best_obj <- Inf
  p_full <- if (kernel == "gaussian") nrow(X) else ncol(X)
  for (r in seq_len(restarts)) {
    if (r == 1 && !is.null(init_theta) && length(init_theta) == p_full + 1) {
      init_th <- init_theta
    } else {
      if (!is.null(seed)) set.seed(seed + 100000 + r)
      init_th <- rnorm(p_full + 1, 0, 0.1)
    }
    m <- loss_shift_classifier_all(
      X, y, base_loss, style, kernel,
      sigma = bp$sigma, alpha = bp$alpha, eta = bp$eta, lambda = bp$lambda,
      max_iter = max_iter, svm_dual = svm_dual,
      line_search = line_search, step_set = step_set,
      init_theta = init_th, verbose = verbose,
      standardize = TRUE)
    obj <- .compute_obj_for_model(m, X, y)
    if (obj < best_obj) { best_obj <- obj; best_mod <- m }
  }
  best_mod
}

#################### (15) 통합 Wrapper (최종 restart 적용) #############
train_loss_shift <- function(
    X_train, y_train,
    X_test = NULL,
    base_loss = c("hinge", "sqhinge", "logistic", "exp"),
    style     = c("none", "soft", "hard"),
    kernel    = c("linear", "gaussian"),
    lambda_grid = 2^seq(-5, 2, len = 6),
    sigma_grid  = NULL,              # ignored for gaussian
    alpha_grid  = c(0.25, 0.5, 1, 2),
    eta_grid    = c(0.1, 0.25, 0.5, 1),
    restarts    = 5,
    n_folds     = 5,
    svm_dual    = FALSE,
    max_iter    = 100,
    line_search = TRUE,
    step_set    = 2^c(-2, -1, 0, 1),
    verbose     = FALSE,
    seed        = NULL,
    sigma_mult  = c(0.5,1,2),
    pre_trained_initial_params = NULL # <--- 'warm_start_params'에서 변경
) {
  
  base_loss <- match.arg(base_loss)
  style     <- match.arg(style)
  kernel    <- match.arg(kernel)
  if (!is.null(seed)) set.seed(seed)
  
  tune_out <- fullgrid_tune_loss_shift(
    X_train, y_train,
    base_loss = base_loss, style = style, kernel = kernel,
    lambda_grid = lambda_grid, sigma_grid = NULL, alpha_grid = alpha_grid, eta_grid = eta_grid,
    restarts = restarts, n_folds = n_folds, svm_dual = svm_dual,
    max_iter = max_iter, line_search = line_search, step_set = step_set,
    verbose = verbose,
    seed = if (is.null(seed)) NULL else seed + 1234,
    sigma_mult = sigma_mult)
  
  best <- tune_out$best_pars
  
  p_full <- if (kernel == "gaussian") nrow(X_train) else ncol(X_train)
  best_model <- NULL; best_obj <- Inf
  for (r in seq_len(restarts)) {
    init_th <- NULL
    # 첫 번째 restart이고, 사전 학습된 초기값이 제공된 경우에만 사용
    if (r == 1 && !is.null(pre_trained_initial_params)) {
      if (verbose) message("--> [Restart 1] Applying pre-trained initial parameters.")
      # w(가중치)와 b(편향)를 합쳐서 theta 생성
      init_th <- c(pre_trained_initial_params$b, as.vector(pre_trained_initial_params$w))
      
      # 차원이 맞지 않으면 경고 후 랜덤 초기값 사용
      if (length(init_th) != p_full + 1) {
        warning("Pre-trained initial parameters dimension mismatch. Reverting to random init.", call. = FALSE)
        init_th <- NULL
      }
    }
    
    # 사전 학습된 초기값을 사용하지 않거나, 두 번째 이후의 restart인 경우
    if (is.null(init_th)) {
      if (!is.null(seed)) set.seed(seed + 200000 + r)
      init_th <- rnorm(p_full + 1, 0, 0.1)
    }
    m <- loss_shift_classifier_all(
      X_train, y_train,
      base_loss = base_loss, style = style, kernel = kernel,
      sigma = best$sigma, alpha = best$alpha, eta = best$eta, lambda = best$lambda,
      max_iter = max_iter, svm_dual = svm_dual,
      line_search = line_search, step_set = step_set,
      init_theta = init_th, verbose = verbose,
      standardize = TRUE)
    obj <- .compute_obj_for_model(m, X_train, y_train)
    if (obj < best_obj) { best_obj <- obj; best_model <- m }
  }
  
  test_pred <- if (!is.null(X_test)) predict_loss_shift(best_model, X_test) else NULL
  list(model = best_model, best_param = best,
       cv_table = tune_out$cv_table, cv_best_acc = tune_out$best_cv_acc,
       test_pred = test_pred)
}