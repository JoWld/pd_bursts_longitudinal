# 2x18 Mixed ANOVA 
# Between-subjects factor: Group (2 levels)
# Within-subjects factor: Atom (18 levels)

# Load required packages
if (!require("ez")) install.packages("ez")
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("rstatix")) install.packages("rstatix")
if (!require("emmeans")) install.packages("emmeans")
if (!require("ggplot2")) install.packages("ggplot2")
library(ez)
library(tidyverse)
library(rstatix)
library(emmeans)
library(ggplot2)

# Read your data
data_wide <- read.csv("your_data_file.csv")
# data_wide should have columns: subject, group, atom1, atom2, ..., atom18

# Convert the data from wide to long format
reshape_data <- function(data_wide) {
  # Ensure subject is character for consistency
  data_wide$subject <- as.character(data_wide$subject)
  
  # Convert data from wide to long format
  data_long <- data_wide %>%
    pivot_longer(
      cols = starts_with("atom"), 
      names_to = "atom",
      values_to = "value"
    )
  
  # Clean up atom names if needed (removing 'atom' prefix)
  data_long$atom <- factor(data_long$atom, levels = paste0("atom", 1:18))
  
  return(data_long)
}

# Apply the reshape function to your wide data
data <- reshape_data(data_wide)

# Summary statistics
summary_stats <- data %>%
  group_by(group, atom) %>%
  summarise(
    mean = mean(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    n = n(),
    se = sd / sqrt(n)
  )

print(summary_stats)

# Visualize the data
ggplot(summary_stats, aes(x = atom, y = mean, group = group, color = group)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2) +
  theme_minimal() +
  labs(
    title = "Mean Values by Group and Atom",
    x = "Atom",
    y = "Mean Value",
    color = "Group"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Check assumptions for mixed ANOVA

# 1. Check for outliers
outliers <- data %>%
  group_by(group, atom) %>%
  identify_outliers(value)

# Display significant outliers (if any)
outliers %>% filter(is.extreme == TRUE)

# 2. Test for normality (Shapiro-Wilk test)
normality_test <- data %>%
  group_by(group, atom) %>%
  shapiro_test(value)

# Display groups/atoms that deviate from normality
normality_violations <- normality_test %>% filter(p < .05)
print(normality_violations)

# 3. Test for homogeneity of variances (Levene's test)
levene_test <- data %>%
  group_by(atom) %>%
  levene_test(value ~ group)

# Display atoms with significant variance differences
print(levene_test %>% filter(p < .05))

# Run the mixed ANOVA using ezANOVA from ez package
anova_result <- ezANOVA(
  data = data,
  dv = value,
  wid = subject,
  between = group,
  within = atom,
  detailed = TRUE,
  type = 3
)

# Print ANOVA results
print(anova_result)

# Check if Mauchly's test indicates violation of sphericity
if (anova_result$`Mauchly's Test for Sphericity`$p[1] < 0.05) {
  cat("\nSphericity assumption violated. Use Greenhouse-Geisser or Huynh-Feldt corrections.\n")
  cat("Greenhouse-Geisser Epsilon:", anova_result$`Sphericity Corrections`$GGe[1], "\n")
  cat("Huynh-Feldt Epsilon:", anova_result$`Sphericity Corrections`$HFe[1], "\n")
  
  # Adjusted p-values for within-subjects effects
  cat("\nGreenhouse-Geisser Corrected p-value for Atom effect:", 
      anova_result$`Sphericity Corrections`$`p[GG]`[1], "\n")
  cat("Greenhouse-Geisser Corrected p-value for Group:Atom interaction:", 
      anova_result$`Sphericity Corrections`$`p[GG]`[2], "\n")
} else {
  cat("\nSphericity assumption met.\n")
}

# Determine which post-hoc tests to run based on ANOVA results

# Main effect of Group (if significant)   # No need for further post-hoc test for Group since there are only 2 levels
if (anova_result$ANOVA$p[1] < 0.05) {
  cat("\nSignificant main effect of Group. Group differences:\n")
  
  # Simple comparison of group means
  group_comp <- data %>%
    group_by(group) %>%
    summarise(mean = mean(value), sd = sd(value))
  
  print(group_comp)
}

# Main effect of Atom (if significant)
if (anova_result$ANOVA$p[2] < 0.05) {
  cat("\nSignificant main effect of Atom. Running pairwise comparisons with Bonferroni correction:\n")
  
  # Create a linear model for emmeans
  model <- lm(value ~ group * atom + Error(subject/atom), data = data)
  
  # Pairwise comparisons for Atom main effect
  atom_posthoc <- emmeans(model, ~ atom) %>%
    pairs(adjust = "fdr") # adjust for multiple comparison using False Discovery Method
  print(atom_posthoc %>% as.data.frame() %>% filter(p.value < 0.05) %>% arrange(p.value))
}

# Interaction effect gorup x atom (if significant)
if (anova_result$ANOVA$p[3] < 0.05) {
  cat("\nSignificant Group x Atom interaction. Running simple effects analysis:\n")
  
  # 1. Simple effects of group at each level of atom
  cat("\nSimple effects of Group at each Atom level:\n")
  
  # Create a list to store results
  simple_effects_results <- list()
  
  # Loop through all atom levels
  for (a in levels(data$atom)) {
    # Subset data for current atom
    atom_data <- data %>% filter(atom == a)
    
    # Run t-test to compare groups
    ttest_result <- t.test(value ~ group, data = atom_data)
    
    # Store results
    simple_effects_results[[a]] <- data.frame(
      Atom = a,
      t_value = ttest_result$statistic,
      df = ttest_result$parameter,
      p_value = ttest_result$p.value,
      Group1_mean = ttest_result$estimate[1],
      Group2_mean = ttest_result$estimate[2],
      mean_diff = ttest_result$estimate[1] - ttest_result$estimate[2]
    )
  }
  
  # Combine results and apply Bonferroni correction
  simple_effects_df <- bind_rows(simple_effects_results)
  simple_effects_df$p_adjusted <- p.adjust(simple_effects_df$p_value, method = "fdr")
  
  # Print significant results only, might be interesting to check unadjusted p values and results to see any non-sig trends
  significant_simple_effects <- simple_effects_df %>% filter(p_adjusted < 0.05) %>% arrange(p_adjusted)
  print(significant_simple_effects)
  
  # 2. Simple effects of atom within each group
  cat("\nSimple effects of Atom within each Group:\n")
  
  # Group 1
  group1_data <- data %>% filter(group == "Group1")
  group1_aov <- aov(value ~ atom + Error(subject/atom), data = group1_data)
  cat("\nANOVA for Atom effect within Group1:\n")
  print(summary(group1_aov))
  
  # Group 2
  group2_data <- data %>% filter(group == "Group2")
  group2_aov <- aov(value ~ atom + Error(subject/atom), data = group2_data)
  cat("\nANOVA for Atom effect within Group2:\n")
  print(summary(group2_aov))
  
  # If atom effect is significant within a group, run post-hoc tests
  # This example uses emmeans for Group1
  if (summary(group1_aov)[[2]][[1]]["atom", "Pr(>F)"] < 0.05) {
    cat("\nPost-hoc tests for Atom effect within Group1:\n")
    group1_posthoc <- emmeans(group1_aov, ~ atom) %>%
      pairs(adjust = "fdr")
    
    # Print significant comparisons only
    print(group1_posthoc %>% 
            as.data.frame() %>% 
            filter(p.value < 0.05) %>% 
            arrange(p.value) %>%
            head(15)) # Showing top 15 to avoid excessive output
  }
  
  # This example uses emmeans for Group2
  if (summary(group2_aov)[[2]][[1]]["atom", "Pr(>F)"] < 0.05) {
    cat("\nPost-hoc tests for Atom effect within Group2:\n")
    group2_posthoc <- emmeans(group2_aov, ~ atom) %>%
      pairs(adjust = "fdr")
    
    # Print significant comparisons only
    print(group2_posthoc %>% 
            as.data.frame() %>% 
            filter(p.value < 0.05) %>% 
            arrange(p.value) %>%
            head(15)) # Showing top 15 to avoid excessive output
  }
}

# Effect size calculations
# Partial eta-squared for main effects and interaction
cat("\nEffect sizes (partial eta-squared):\n")
anova_df <- as.data.frame(anova_result$ANOVA)
anova_df$partial_eta_sq <- anova_df$SSn / (anova_df$SSn + anova_df$SSd)
print(anova_df[, c("Effect", "partial_eta_sq")])

# Output final visualization with significant differences highlighted
# This will create a more informative plot based on the results

# Create a plot highlighting significant differences between groups (if interaction is significant)
if (anova_result$ANOVA$p[3] < 0.05) {
  # Add significance markers to summary stats
  summary_stats$sig <- FALSE
  
  for (a in unique(summary_stats$atom)) {
    sig_test <- simple_effects_df %>% filter(Atom == a & p_adjusted < 0.05)
    if (nrow(sig_test) > 0) {
      summary_stats$sig[summary_stats$atom == a] <- TRUE
    }
  }
  
  # Create enhanced plot
  ggplot(summary_stats, aes(x = atom, y = mean, group = group, color = group)) +
    geom_line(size = 1) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2) +
    geom_text(data = filter(summary_stats, sig == TRUE & group == unique(summary_stats$group)[1]),
              aes(label = "*", y = mean + se + 2), 
              color = "black", size = 5) +
    theme_minimal() +
    labs(
      title = "Mean Values by Group and Atom with Significant Differences",
      subtitle = "* indicates significant difference between groups (p < 0.05, Bonferroni-corrected)",
      x = "Atom",
      y = "Mean Value",
      color = "Group"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Saving results to file (optional)
# write.csv(summary_stats, "mixed_anova_summary_stats.csv", row.names = FALSE)
# Save the plots using ggsave() if needed

cat("\nMixed ANOVA analysis complete.\n")
