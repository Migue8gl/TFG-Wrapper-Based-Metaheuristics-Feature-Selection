library(rio)
library(dplyr)

# Import data
df_binary <- import("results/binary/analysis_results.csv")
df_real <- import("results/real/analysis_results.csv")

# Perform pairwise Wilcoxon rank sum tests and save results
metrics <- c("selected_rate", "acc", "avg", "execution_time")
for (metric in metrics) {
    write.csv(pairwise.wilcox.test(df_binary[[metric]], df_binary$optimizer, p.adjust.method = "holm", paired = TRUE)$p.value,
        file = paste0("results/stats/wilcox_test_", metric, "_binary.csv")
    )
    write.csv(pairwise.wilcox.test(df_real[[metric]], df_real$optimizer, p.adjust.method = "holm", paired = TRUE)$p.value,
        file = paste0("results/stats/wilcox_test_", metric, "_real.csv")
    )
}
