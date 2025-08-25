########################################################################
#  loss_shifting_v3.R   (2025-08-14, verbose & tiebreak revamp 2025-08-17)
#  - 'warm-start' -> 'pre_trained_initial'
#  - linear 커널에도 자동 표준화 (standardize 옵션, default=TRUE)
#  - sigma grid: z-scored X의 median heuristic
#  - reproducibility, Gaussian fold-dim init, final restarts kept
#  - [NEW] 일관 로깅: START / STEP / END
#  - [NEW] acc 동점 시: 가장 단순한 하이퍼파라미터 우선
#         우선순위: 큰 lambda ▶ (gaussian이면) 큰 sigma ▶ 작은 alpha ▶ 큰 eta
########################################################################

suppressPackageStartupMessages({ library(quadprog) })

#################### (0) Logging Utils ################################
.ts_now <- function() format(Sys.time(), "%H:%M:%S")
.v_on  <- function(verbose) isTRUE(verbose)
.vopen <- function(tag, msg = "", verbose = FALSE) { if (.v_on(verbose)) cat(sprintf("\n┏[%s %s] %s\n", tag, .ts_now(), msg)) }
.vstep <- function(tag, msg, verbose = FALSE)      { if (.v_on(verbose)) cat(sprintf("┃ %s\n", msg)) }
.vok   <- function(tag, msg = "done", verbose = FALSE) { if (.v_on(verbose)) cat(sprintf("┗[%s %s] %s\n", tag, .ts_now(), msg)) }
.vkv <- function(k, v) sprintf("%s=%s", k, v)

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

#################### (5) Primal Updaters (MM/IRLS) ####################
.update_iter_msg <- function(name, it, delta, verbose) {
  if (.v_on(verbose)) cat(sprintf("┃ [%s][iter=%02d] ||dtheta||2=%.3e\n", name, it, delta))
}

update_f_hinge_MM <- function(theta, X, y, r, lambda,
                              max_inner = 20, tol_inner = 1e-6, verbose_inner = FALSE) {
  tag <- "HINGE-MM"
  n <- nrow(X); p <- ncol(X)
  X_aug <- cbind(1, X); eps <- 1e-4
  .vopen(tag, "inner optimization START", verbose_inner)
  for (it in seq_len(max_inner)) {
    f_val <- as.vector(X_aug %*% theta)
    u0 <- y * f_val - r
    w_vec <- 1 / (4 * abs(1 - u0) + eps)
    t_vec <- y * (r + 1 + abs(1 - u0))
    WX    <- sqrt(w_vec) * X_aug
    Wt    <- sqrt(w_vec) * t_vec
    A     <- crossprod(WX) + make_penalty_mat(p, n, lambda)
    b     <- crossprod(WX, Wt)
    theta_new <- solve(A, b)
    d <- sqrt(sum((theta_new - theta)^2))
    .update_iter_msg("hinge-MM", it, d, verbose_inner)
    theta <- theta_new
    if (d < tol_inner) break
  }
  .vok(tag, "inner optimization END", verbose_inner)
  theta
}

update_f_sqhinge_MM <- function(theta, X, y, r, lambda,
                                max_inner = 20, tol_inner = 1e-6, verbose_inner = FALSE) {
  tag <- "SQHINGE-MM"
  n <- nrow(X); p <- ncol(X)
  X_aug <- cbind(1, X)
  .vopen(tag, "inner optimization START", verbose_inner)
  for (it in seq_len(max_inner)) {
    f_val <- as.vector(X_aug %*% theta)
    u0 <- y * f_val - r
    t_vec <- y * (r + pmax(1, u0))
    A     <- crossprod(X_aug) + make_penalty_mat(p, n, lambda)
    b     <- crossprod(X_aug, t_vec)
    theta_new <- solve(A, b)
    d <- sqrt(sum((theta_new - theta)^2))
    .update_iter_msg("sqhinge-MM", it, d, verbose_inner)
    theta <- theta_new
    if (d < tol_inner) break
  }
  .vok(tag, "inner optimization END", verbose_inner)
  theta
}

update_f_logistic_IRLS <- function(theta, X, y, r, lambda,
                                   max_inner = 20, tol_inner = 1e-6, verbose_inner = FALSE) {
  tag <- "LOGISTIC-IRLS"
  X_aug <- cbind(1, X); n <- nrow(X); p <- ncol(X)
  w_min <- 1e-5; eps_rd <- 1e-4
  .vopen(tag, "inner optimization START", verbose_inner)
  for (it in seq_len(max_inner)) {
    f_val <- as.vector(X_aug %*% theta)
    u_eff <- y * f_val - r
    sigma <- 1 / (1 + exp(u_eff))
    w_vec <- pmax(sigma * (1 - sigma), w_min)
    grad  <- c((-1 / n) * sum(y * sigma),
               (-1 / n) * (t(X) %*% (y * sigma)) + lambda * theta[-1])
    W <- diag(as.vector(w_vec))
    H <- (1 / n) * t(X_aug) %*% W %*% X_aug + diag(c(eps_rd, rep(lambda + eps_rd, p)))
    delta <- solve(H, grad); theta_new <- theta - delta
    d <- sqrt(sum(delta^2))
    .update_iter_msg("logistic-IRLS", it, d, verbose_inner)
    theta <- theta_new
    if (d < tol_inner) break
  }
  .vok(tag, "inner optimization END", verbose_inner)
  theta
}

update_f_exponential_IRLS <- function(theta, X, y, r, lambda,
                                      max_inner = 20, tol_inner = 1e-6, verbose_inner = FALSE) {
  tag <- "EXP-IRLS"
  X_aug <- cbind(1, X); n <- nrow(X); p <- ncol(X)
  w_min <- 1e-5; eps_rd <- 1e-4
  .vopen(tag, "inner optimization START", verbose_inner)
  for (it in seq_len(max_inner)) {
    f_val <- as.vector(X_aug %*% theta)
    u_eff <- y * f_val - r
    e_neg <- exp(-u_eff)
    w_vec <- pmax(e_neg, w_min)
    grad  <- c((-1 / n) * sum(e_neg * y),
               (-1 / n) * (t(X) %*% (e_neg * y)) + lambda * theta[-1])
    W <- diag(as.vector(w_vec))
    H <- (1 / n) * t(X_aug) %*% W %*% X_aug + diag(c(eps_rd, rep(lambda + eps_rd, p)))
    delta <- solve(H, grad); theta_new <- theta - delta
    d <- sqrt(sum(delta^2))
    .update_iter_msg("exp-IRLS", it, d, verbose_inner)
    theta <- theta_new
    if (d < tol_inner) break
  }
  .vok(tag, "inner optimization END", verbose_inner)
  theta
}

###########################################
# (6) Dual SVM (quadprog)
###########################################
dual_svm_hinge <- function(X, y, r, lambda, verbose = FALSE) {
  tag <- "DUAL-HINGE"
  .vopen(tag, "QP solve START", verbose)
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
  .vok(tag, sprintf("QP solve END (num_SV=%d)", length(sv)), verbose)
  list(theta = c(b0, beta), alpha = alpha)
}

dual_svm_sqhinge <- function(X, y, r, lambda, verbose = FALSE) {
  tag <- "DUAL-SQHINGE"
  .vopen(tag, "QP solve START", verbose)
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
  .vok(tag, sprintf("QP solve END (num_SV=%d)", length(sv)), verbose)
  list(theta = c(b0, beta), alpha = alpha)
}

#################### (7) 라인 서치 ###################################
line_search_choice <- function(theta_old, theta_raw, X, y, shift_fun, loss_fun, lambda,
                               step_set = 2 ^ c(-2, -1, 0, 1, 2), verbose = FALSE) {
  tag <- "LINE-SEARCH"
  candidates <- lapply(step_set, function(s) theta_old + s * (theta_raw - theta_old))
  objs <- vapply(candidates, function(th)
    objective_value(th, X, y, shift_fun, loss_fun, lambda), numeric(1L))
  best_i <- which.min(objs)
  .vstep(tag, sprintf("candidates=%s -> pick step=%g (obj=%.6g)",
                      paste(step_set, collapse = ","), step_set[best_i], objs[best_i]), verbose)
  candidates[[best_i]]
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
  tag <- "TRAIN(PRIMAL/DUAL)"
  X_mat <- as.matrix(X)
  y_vec <- ifelse(y > 0, 1, -1)
  base_loss <- match.arg(base_loss)
  style     <- match.arg(style)
  kernel    <- match.arg(kernel)
  use_dual  <- svm_dual && base_loss %in% c("hinge", "sqhinge")
  
  .vopen(tag, sprintf("START  [%s | %s | %s]", base_loss, style, if (use_dual) "dual" else "primal"), verbose)
  
  if (standardize) {
    st <- .standardize_fit(X_mat)
    X_proc <- st$Xs
  } else {
    st <- NULL; X_proc <- X_mat
  }
  
  if (kernel == "gaussian") {
    K_train <- rbf_kernel(X_proc, NULL, sigma)
    eig     <- eigen(K_train, symmetric = TRUE)
    vals    <- pmax(eig$values, .Machine$double.eps)
    U       <- eig$vectors
    D_half  <- diag(sqrt(vals))
    X_feat  <- U %*% D_half
  } else X_feat <- X_proc
  
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
  .vstep(tag, paste(
    .vkv("n", n), .vkv("p", p),
    .vkv("lambda", format(lambda, digits=4)),
    .vkv("sigma", if (kernel=="gaussian") format(sigma, digits=4) else "NA"),
    .vkv("alpha", if (style!="none") format(alpha, digits=4) else "NA"),
    .vkv("eta",   if (style!="none") format(eta,   digits=4) else "NA"),
    sep = " | "
  ), verbose)
  
  iter_used <- NA
  for (it in seq_len(max_iter)) {
    X_aug <- cbind(1, X_feat)
    f_val <- as.vector(X_aug %*% theta)
    u_vec <- y_vec * f_val
    r_vec <- shift_fun(u_vec)
    
    theta_raw <- if (use_dual) {
      sol <- if (base_loss == "hinge") dual_svm_hinge(X_feat, y_vec, r_vec, lambda, verbose = verbose)
      else                      dual_svm_sqhinge(X_feat, y_vec, r_vec, lambda, verbose = verbose)
      sol$theta
    } else {
      updater(theta, X_feat, y_vec, r_vec, lambda, max_inner, tol, verbose_inner = FALSE)
    }
    
    theta_new <- if (line_search && !use_dual)
      line_search_choice(theta, theta_raw, X_feat, y_vec, shift_fun, loss_fun, lambda,
                         step_set, verbose = verbose) else theta_raw
    
    obj_new <- objective_value(theta_new, X_feat, y_vec, shift_fun, loss_fun, lambda)
    rel <- abs(obj_new - obj_old) / (abs(obj_old) + 1e-8)
    .vstep(tag, sprintf("[iter=%03d] obj=%.6g  rel=%.3e", it, obj_new, rel), verbose)
    theta <- theta_new; obj_old <- obj_new
    if (rel < tol) { iter_used <- it; break }
    if (it == max_iter) iter_used <- it
  }
  
  .vok(tag, sprintf("END (iters=%d, final_obj=%.6g)", iter_used, obj_old), verbose)
  
  out <- list(theta = theta, base_loss = base_loss, style = style, kernel = kernel,
              sigma = sigma, alpha = alpha, eta = eta, lambda = lambda,
              iter_used = iter_used, step_set = step_set,
              standardize = standardize)
  
  if (isTRUE(standardize)) { out$center <- st$mu; out$scale <- st$sd }
  if (kernel == "gaussian") {
    out$U <- U; out$D_inv_sqrt <- 1 / sqrt(vals); out$X_train_orig <- X_mat
  }
  out
}

#################### (9) 예측 ########################################
predict_loss_shift <- function(model, X_new, type = "class") {
  X_new_mat <- as.matrix(X_new)
  if (isTRUE(model$standardize)) {
    X_proc <- .standardize_apply(X_new_mat, model$center, model$scale)
  } else X_proc <- X_new_mat
  
  if (model$kernel == "gaussian") {
    X_tr <- model$X_train_orig
    X_tr_proc <- if (isTRUE(model$standardize)) .standardize_apply(X_tr, model$center, model$scale) else X_tr
    K_new <- rbf_kernel(X_proc, X_tr_proc, model$sigma)
    Z_new <- K_new %*% model$U %*% diag(model$D_inv_sqrt)
    X_aug <- cbind(1, Z_new)
  } else X_aug <- cbind(1, X_proc)
  
  f_val <- as.vector(X_aug %*% model$theta)
  if (type == "score") return(f_val)
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
    base_seed = NULL,
    verbose = FALSE) {
  
  tag <- "CV-ACC(RESTART)"
  best_acc <- -Inf
  .vopen(tag, sprintf("START %s", paste(
    .vkv("lambda", format(lambda, digits=4)),
    .vkv("sigma",  format(sigma,  digits=4)),
    .vkv("alpha",  if (style!="none") format(alpha, digits=4) else "NA"),
    .vkv("eta",    if (style!="none") format(eta,   digits=4) else "NA"),
    sep = " | ")), verbose)
  
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
    cur <- mean(accs, na.rm = TRUE)
    best_acc <- max(best_acc, cur)
    .vstep(tag, sprintf("restart=%02d  fold_mean_acc=%.4f  best=%.4f", r, cur, best_acc), verbose)
  }
  .vok(tag, sprintf("END   best_cv=%.4f", best_acc), verbose)
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
  
  tag <- "TUNE-GRID"
  .vopen(tag, sprintf("START [%s | %s | %s] grid-search",
                      base_loss, style, kernel), verbose)
  
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
      base_seed = if (is.null(seed)) NULL else seed + i * 1000,
      verbose = verbose)
    if (.v_on(verbose)) {
      .vstep(tag, sprintf("grid %03d/%03d : (%s)  -> cv=%.4f",
                          i, nrow(grid),
                          paste(c(
                            sprintf("lambda=%s", format(grid$lambda[i], digits=4)),
                            sprintf("sigma=%s",  format(grid$sigma[i],  digits=4)),
                            sprintf("alpha=%s",  if (style!="none") format(grid$alpha[i], digits=4) else "NA"),
                            sprintf("eta=%s",    if (style!="none") format(grid$eta[i],   digits=4) else "NA")
                          ), collapse=", "), acc_vec[i]), TRUE)
    } else {
      cat(sprintf("(lambda=%.3g, sigma=%.3g, alpha=%.3g, eta=%.3g)  cv-acc=%.4f  [%d/%d]\n",
                  grid$lambda[i], grid$sigma[i], grid$alpha[i], grid$eta[i], acc_vec[i], i, nrow(grid)))
    }
  }
  
  # --- Tie-breaking: choose simplest among ties ---
  best_acc <- max(acc_vec, na.rm = TRUE)
  tie_idx  <- which(acc_vec == best_acc)
  
  if (length(tie_idx) == 1) {
    best_idx <- tie_idx
  } else {
    G <- grid[tie_idx, , drop = FALSE]
    o <- order(
      -G$lambda,                                                   # 큰 lambda 먼저
      if (kernel == "gaussian") -G$sigma else rep(0, nrow(G)),     # 큰 sigma 먼저
      if (style != "none")  G$alpha else rep(0, nrow(G)),          # 작은 alpha 먼저
      if (style != "none") -G$eta   else rep(0, nrow(G)),          # 큰 eta 먼저
      method = "radix"
    )
    best_idx <- tie_idx[o[1]]
  }
  
  .vok(tag, sprintf("END best at #%d (cv=%.4f)", best_idx, acc_vec[best_idx]), verbose)
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
    sigma_grid  = NULL,
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
  tag <- "FIT(WRAP)"
  if (!is.null(seed)) set.seed(seed)
  
  .vopen(tag, "START full-fit + final restarts", verbose)
  
  tune <- fullgrid_tune_loss_shift(
    X, y, base_loss, style, kernel,
    lambda_grid, sigma_grid = NULL, alpha_grid, eta_grid,
    restarts, n_folds, val_frac,
    svm_dual, max_iter, line_search, step_set, verbose,
    seed = if (is.null(seed)) NULL else seed + 123,
    sigma_mult = sigma_mult)
  
  bp <- tune$best_pars
  if (.v_on(verbose)) {
    .vstep(tag, sprintf("BEST GRID: (%s)",
                        paste(c(
                          sprintf("lambda=%s", format(bp$lambda, digits=4)),
                          sprintf("sigma=%s",  format(bp$sigma,  digits=4)),
                          sprintf("alpha=%s",  if (style!="none") format(bp$alpha, digits=4) else "NA"),
                          sprintf("eta=%s",    if (style!="none") format(bp$eta,   digits=4) else "NA")
                        ), collapse=", ")), TRUE)
    .vstep(tag, sprintf("CV accuracy = %.4f", tune$best_cv_acc), TRUE)
  } else {
    cat("\n[fit_loss_shift] best hyper-parameters\n"); print(bp)
    cat(sprintf("CV accuracy = %.4f\n\n", tune$best_cv_acc))
  }
  
  best_mod <- NULL; best_obj <- Inf
  p_full <- if (kernel == "gaussian") nrow(X) else ncol(X)
  for (r in seq_len(restarts)) {
    if (r == 1 && !is.null(init_theta) && length(init_theta) == p_full + 1) {
      init_th <- init_theta
      .vstep(tag, "restart=01 using provided init_theta", verbose)
    } else {
      if (!is.null(seed)) set.seed(seed + 100000 + r)
      init_th <- rnorm(p_full + 1, 0, 0.1)
      .vstep(tag, sprintf("restart=%02d random init", r), verbose)
    }
    m <- loss_shift_classifier_all(
      X, y, base_loss, style, kernel,
      sigma = bp$sigma, alpha = bp$alpha, eta = bp$eta, lambda = bp$lambda,
      max_iter = max_iter, svm_dual = svm_dual,
      line_search = line_search, step_set = step_set,
      init_theta = init_th, verbose = verbose,
      standardize = TRUE)
    obj <- .compute_obj_for_model(m, X, y)
    .vstep(tag, sprintf("restart=%02d  obj=%.6g  %s", r, obj,
                        if (obj < best_obj) "<<< best so far" else ""), verbose)
    if (obj < best_obj) { best_obj <- obj; best_mod <- m }
  }
  .vok(tag, sprintf("END best_obj=%.6g", best_obj), verbose)
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
    sigma_grid  = NULL,
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
    pre_trained_initial_params = NULL
) {
  tag <- "TRAIN(WRAP)"
  base_loss <- match.arg(base_loss)
  style     <- match.arg(style)
  kernel    <- match.arg(kernel)
  if (!is.null(seed)) set.seed(seed)
  
  .vopen(tag, sprintf("START [%s | %s | %s]", base_loss, style, kernel), verbose)
  
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
  if (.v_on(verbose)) {
    .vstep(tag, sprintf("BEST GRID: (%s)",
                        paste(c(
                          sprintf("lambda=%s", format(best$lambda, digits=4)),
                          sprintf("sigma=%s",  format(best$sigma,  digits=4)),
                          sprintf("alpha=%s",  if (style!="none") format(best$alpha, digits=4) else "NA"),
                          sprintf("eta=%s",    if (style!="none") format(best$eta,   digits=4) else "NA")
                        ), collapse=", ")), TRUE)
    .vstep(tag, sprintf("CV accuracy = %.4f", tune_out$best_cv_acc), TRUE)
  }
  
  p_full <- if (kernel == "gaussian") nrow(X_train) else ncol(X_train)
  best_model <- NULL; best_obj <- Inf
  for (r in seq_len(restarts)) {
    init_th <- NULL
    if (r == 1 && !is.null(pre_trained_initial_params)) {
      if (.v_on(verbose)) .vstep(tag, "restart=01 using pre-trained initial parameters", TRUE)
      init_th <- c(pre_trained_initial_params$b, as.vector(pre_trained_initial_params$w))
      if (length(init_th) != p_full + 1) {
        warning("Pre-trained initial parameters dimension mismatch. Reverting to random init.", call. = FALSE)
        init_th <- NULL
      }
    }
    if (is.null(init_th)) {
      if (!is.null(seed)) set.seed(seed + 200000 + r)
      init_th <- rnorm(p_full + 1, 0, 0.1)
      if (.v_on(verbose)) .vstep(tag, sprintf("restart=%02d random init", r), TRUE)
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
    if (.v_on(verbose)) .vstep(tag, sprintf("restart=%02d  obj=%.6g", r, obj), TRUE)
    if (obj < best_obj) { best_obj <- obj; best_model <- m }
  }
  
  test_pred <- if (!is.null(X_test)) predict_loss_shift(best_model, X_test) else NULL
  .vok(tag, "END training wrapper", verbose)
  
  list(model = best_model, best_param = best,
       cv_table = tune_out$cv_table, cv_best_acc = tune_out$best_cv_acc,
       test_pred = test_pred)
}
