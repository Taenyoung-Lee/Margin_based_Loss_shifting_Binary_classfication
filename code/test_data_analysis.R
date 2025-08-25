# 필요한 라이브러리 로드
library(ggplot2)
library(dplyr)
library(stringr)

# 1) CSV 파일 경로
#csv_path <- "R/loss_shifting/test_result/0711_seed2025_unsym.csv"
#csv_path <- "R/loss_shifting/test_result/0711_seed123456_unsym.csv"
csv_path <- "0806_30rep_400sam.csv"



# 2) 데이터 불러오기
df <- read.csv(csv_path, stringsAsFactors = FALSE)

# 3) 토큰 분리해서 base_loss, style 추출
tokens <- str_split_fixed(df$model, "_", 4)
df$base_loss <- tokens[,2]
df$style     <- tokens[,3]

# 4) category 생성: LossShift → base_loss-style, 그 외는 모델명 그대로
df <- df %>%
  mutate(category = ifelse(
    str_detect(model, "^LossShift"),
    paste0(base_loss, "-", style),
    model
  ))

# 5) LossShift에서 사용된 base_loss 순서(등장 순서대로) 뽑기
base_loss_levels <- unique(df$base_loss[str_detect(df$model, "^LossShift")])

# 6) 스타일 순서 정의
style_order <- c("none", "soft", "hard")

# 7) 원하는 x축 레벨 순서 만들기: base_loss별로(style 순서 내에서) 묶기
lossshift_levels <- unlist(lapply(base_loss_levels, function(b) {
  paste0(b, "-", style_order)
}))
# 벤치마크 모델들(그 외) 레벨
other_levels <- sort(setdiff(unique(df$category), lossshift_levels))
# 최종 레벨 순서
all_levels <- c(lossshift_levels, other_levels)

# 8) factor 레벨 지정
df$category <- factor(df$category, levels = all_levels)

# 9) 그래프 그리기
ggplot(df, aes(x = category, y = accuracy)) + # accuracy, f1으로 변경 가능능
  geom_boxplot(outlier.shape = 1) +
  stat_summary(
    fun = mean, geom = "point",
    shape = 17, size = 3, color = "red"
  ) +
  labs(
    title = "Accuracy distribution by LossShift base_loss–style and baselines",
    x     = "Category (base_loss-style or baseline)",
    y     = "Accuracy"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title     = element_text(hjust = 0.5)
  )

