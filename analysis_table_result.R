# ─────────────────────────────────────────────────────────────
# 패키지
# ─────────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(tidyverse)
  library(stringr)
  library(ggrepel)   # 끝점 라벨용(선택)
})

# ─────────────────────────────────────────────────────────────
# 1) 데이터 로드 & "0.947 (0.012)" → mean/sd 파싱
# ─────────────────────────────────────────────────────────────
path <- "C:/Users/Tae/Downloads/xor_0918_revised.csv"
raw  <- readr::read_csv(path, show_col_types = FALSE)

parse_mean <- function(x) readr::parse_number(x)
parse_sd   <- function(x) {
  s <- str_match(as.character(x), "\\(([^\\)]+)\\)")[,2]
  readr::parse_number(s)
}

stopifnot(all(c("rate_pos","true","e1071","glm") %in% names(raw)))

# base (oracle/e1071/glm)
base_long <- raw %>%
  transmute(
    rate_pos,
    true   = true,
    e1071  = e1071,
    glm    = glm
  ) %>%
  pivot_longer(-rate_pos, names_to = "series", values_to = "val") %>%
  mutate(
    mean  = parse_mean(val),
    sd    = parse_sd(val),
    group = "baseline"
  ) %>%
  select(rate_pos, group, series, mean, sd)

# loss 계열
loss_groups <- c("hinge","sqhinge","logistic","exp")
modes       <- c("none","soft","hard")
loss_cols   <- intersect(
  unlist(lapply(loss_groups, \(g) paste0(g, "_", modes))),
  names(raw)
)

loss_long <- raw %>%
  select(rate_pos, all_of(loss_cols)) %>%
  pivot_longer(-rate_pos, names_to = "series", values_to = "val") %>%
  mutate(
    mean  = parse_mean(val),
    sd    = parse_sd(val),
    group = sub("_(none|soft|hard)$","", series),
    mode  = sub("^(.*)_","", series)
  ) %>%
  select(rate_pos, group, mode, series, mean, sd)

# ─────────────────────────────────────────────────────────────
# 2) 색/선형/마커 팔레트 (고정)
# ─────────────────────────────────────────────────────────────
#  — loss 모드
mode_colors <- c(
  "none" = "#1f77b4",  # 파랑
  "soft" = "#ff7f0e",  # 주황
  "hard" = "#2ca02c"   # 초록
)
mode_shapes <- c("none"=16, "soft"=15, "hard"=17)
mode_lty    <- c("none"="solid", "soft"="solid", "hard"="solid")

#  — 베이스라인
base_colors <- c(
  "true"  = "#9467bd", # 보라
  "e1071" = "#17becf", # 청록
  "glm"   = "#d62728"  # 빨강
)
base_lty <- c("true"="dashed", "e1071"="dotted", "glm"="dotdash")
base_shapes <- c("true"=22, "e1071"=4, "glm"=8)

# ─────────────────────────────────────────────────────────────
# 3) 그리기 함수: 그룹별로 한 장씩
# ─────────────────────────────────────────────────────────────
plot_one_group <- function(g, show_sd_ribbon = FALSE, label_ends = TRUE) {
  dg  <- filter(loss_long, group == g)
  db  <- base_long  # baseline은 모든 그래프 공통
  
  p <- ggplot() +
    # (선택) ±1 sd 리본
    { if (show_sd_ribbon)
      geom_ribbon(
        data = dg, aes(x = rate_pos, ymin = mean - sd, ymax = mean + sd, fill = mode),
        alpha = 0.12, inherit.aes = FALSE
      )
    } +
    # loss 모드 라인
    geom_line(data = dg, aes(rate_pos, mean, color = mode, linetype = mode), linewidth = 1.2) +
    geom_point(data = dg, aes(rate_pos, mean, color = mode, shape = mode), size = 2.7) +
    
    # 베이스라인들
    geom_line(data = db, aes(rate_pos, mean, color = series, linetype = series),
              linewidth = 1.0, inherit.aes = FALSE) +
    geom_point(data = db, aes(rate_pos, mean, color = series, shape = series),
               size = 2.4, inherit.aes = FALSE) +
    
    scale_color_manual(
      values = c(mode_colors, base_colors),
      breaks = c(names(mode_colors), names(base_colors))
    ) +
    scale_shape_manual(values = c(mode_shapes, base_shapes)) +
    scale_linetype_manual(values = c(mode_lty, base_lty)) +
    
    coord_cartesian(ylim = c(0.80, 1.00)) +
    labs(
      title = sprintf("XOR: %s group vs baselines", g),
      x = "rate_pos",
      y = "Accuracy",
      color = NULL, shape = NULL, linetype = NULL, fill = "mode"
    ) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom")
  
  # 선 끝점 라벨(선택)
  if (label_ends) {
    last_x <- max(dg$rate_pos, na.rm = TRUE)
    end_loss <- dg %>%
      group_by(mode) %>%
      filter(rate_pos == last_x) %>%
      mutate(lbl = paste0(g, "_", mode))
    end_base <- db %>%
      group_by(series) %>%
      filter(rate_pos == last_x) %>%
      mutate(lbl = series)
    
    p <- p +
      ggrepel::geom_text_repel(
        data = end_loss, aes(rate_pos, mean, label = lbl, color = mode),
        nudge_x = 0.005, direction = "y", size = 3.2, segment.alpha = 0.3, show.legend = FALSE
      ) +
      ggrepel::geom_text_repel(
        data = end_base, aes(rate_pos, mean, label = lbl, color = series),
        nudge_x = 0.005, direction = "y", size = 3.2, segment.alpha = 0.3, show.legend = FALSE
      )
  }
  p
}

# ─────────────────────────────────────────────────────────────
# 4) 출력 & 저장 (그룹별로 따로)
# ─────────────────────────────────────────────────────────────
plots <- setNames(vector("list", length(loss_groups)), loss_groups)
for (g in loss_groups) {
  p <- plot_one_group(g, show_sd_ribbon = FALSE, label_ends = TRUE)
  plots[[g]] <- p
  print(p)
  ggsave(sprintf("xor_%s_group_vs_baselines.png", g),
         p, width = 7.5, height = 5.5, dpi = 150)
}

# 결과는 plots[["hinge"]] 등으로 접근 가능

