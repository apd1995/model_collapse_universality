G = 100

num_mc = 1000000

s_mean = 1
s_median = sqrt(pi/2)

# Replace

dat_mean = matrix(0, nrow = G, ncol = num_mc)
dat_median = matrix(0, nrow = G, ncol = num_mc)

dat_mean[1,] = rnorm(num_mc, 0, s_mean)
cond_mean = 1/s_mean^2
cond_var = s_median^2 - 1/s_mean^2
cond_s = sqrt(cond_var)
dat_median[1,] = cond_mean*dat_mean[1,] + rnorm(num_mc, 0, cond_s)

for(g in 2:G){
  dat_mean[g,] = rnorm(num_mc, 0, s_mean) + s_mean^2*dat_median[(g-1),]
  cond_mean = 1/s_mean^2
  cond_var = s_median^2 - 1/s_mean^2
  cond_s = sqrt(cond_var)
  dat_median[g,] = cond_mean*dat_mean[g,] + rnorm(num_mc, 0, cond_s)
}

row_variances_replace = apply(dat_median, 1, var)
row_variances_ratio_replace = row_variances_replace/row_variances_replace[1]


# Augment

dat_augment = matrix(0, nrow = 2*G, ncol = num_mc)

dat_augment[1,] = rnorm(num_mc, 0, s_mean)
Sigma = matrix(c(s_median^2, 1, 1, s_mean^2), nrow = 2, ncol = 2)
cond_mean = 1/s_mean^2
cond_var = s_median^2 - (1/s_mean^2)
cond_s = sqrt(cond_var)
dat_augment[2,] = cond_mean*dat_augment[1,] + rnorm(num_mc, 0, cond_s)

for(g in 2:G){
  dat_augment[(2*g-1),] = rnorm(num_mc, 0, s_mean) + s_mean^2*dat_augment[(2*(g-1)),]
  Sigma_new = matrix(0, nrow = 2*g, ncol = 2*g)
  Sigma_new[3:(2*g), 3:(2*g)] = Sigma
  for(i in 1:g){
    Sigma_new[1,(2*i)] = 1/g
    Sigma_new[(2*i), 1] = 1/g
    Sigma_new[1, (2*i - 1)] = s_median^2/g
    Sigma_new[(2*i - 1), 1] = s_median^2/g
  }
  Sigma_new[1,1] = s_median^2/g
  Sigma_new[2,2] = s_mean^2
  cond_mean = t(Sigma_new[1, 2:(2*g)]) %*% solve(Sigma_new[2:(2*g), 2:(2*g)]) %*% dat_augment[((2*g-1):1),]
  cond_var = Sigma_new[1,1] - t(Sigma_new[1, 2:(2*g)]) %*% solve(Sigma_new[2:(2*g), 2:(2*g)]) %*% Sigma_new[1, 2:(2*g)]
  cond_s = sqrt(cond_var)
  dat_augment[(2*g),] = cond_mean + rnorm(num_mc, 0, cond_s)
  Sigma = Sigma_new
}

row_variances_augment = apply(dat_augment, 1, var)
row_variances_augment = row_variances_augment[seq(2, 2*G, by = 2)]
row_variances_ratio_augment = row_variances_augment/row_variances_augment[1]

# Subsample

dat_subsample = matrix(0, nrow = 2*G, ncol = num_mc)

dat_subsample[1,] = rnorm(num_mc, 0, s_mean)
Sigma = matrix(c(s_median^2, 1, 1, s_mean^2), nrow = 2, ncol = 2)
cond_mean = 1/s_mean^2
cond_var = s_median^2 - (1/s_mean^2)
cond_s = sqrt(cond_var)
dat_subsample[2,] = cond_mean*dat_subsample[1,] + rnorm(num_mc, 0, cond_s)

for(g in 2:G){
  dat_subsample[(2*g-1),] = rnorm(num_mc, 0, s_mean) + s_mean^2*dat_subsample[(2*(g-1)),]
  Sigma_new = matrix(0, nrow = 2*g, ncol = 2*g)
  Sigma_new[3:(2*g), 3:(2*g)] = Sigma
  for(i in 1:g){
    Sigma_new[1,(2*i)] = 1/g
    Sigma_new[(2*i), 1] = 1/g
    Sigma_new[1, (2*i - 1)] = s_median^2/g
    Sigma_new[(2*i - 1), 1] = s_median^2/g
  }
  Sigma_new[1,1] = s_median^2
  Sigma_new[2,2] = s_mean^2
  cond_mean = t(Sigma_new[1, 2:(2*g)]) %*% solve(Sigma_new[2:(2*g), 2:(2*g)]) %*% dat_subsample[((2*g-1):1),]
  cond_var = Sigma_new[1,1] - t(Sigma_new[1, 2:(2*g)]) %*% solve(Sigma_new[2:(2*g), 2:(2*g)]) %*% Sigma_new[1, 2:(2*g)]
  cond_s = sqrt(cond_var)
  dat_subsample[(2*g),] = cond_mean + rnorm(num_mc, 0, cond_s)
  Sigma = Sigma_new
}

row_variances_subsample = apply(dat_subsample, 1, var)
row_variances_subsample = row_variances_subsample[seq(2, 2*G, by = 2)]
row_variances_ratio_subsample = row_variances_subsample/row_variances_subsample[1]

# combine the data

variance_ratio_replace = data.frame(
  generation = 1:G,
  variance_ratio = row_variances_ratio_replace,
  workflow = rep("discard", G)
)

variance_ratio_augment = data.frame(
  generation = 1:G,
  variance_ratio = row_variances_ratio_augment,
  workflow = rep("augment", G)
)

variance_ratio_subsample = data.frame(
  generation = 1:G,
  variance_ratio = row_variances_ratio_subsample,
  workflow = rep("subsample", G)
)

variance_ratio_data = rbind(variance_ratio_replace,
                            variance_ratio_augment,
                            variance_ratio_subsample)

# plot
library(ggplot2)

ggplot(dplyr::filter(variance_ratio_data, workflow != "discard"), aes(x = generation, y = variance_ratio, color = workflow)) +
  geom_line() +       
  geom_hline(yintercept = pi^2/6, linetype = "dashed", color = "black") +  # Dashed red line for emphasis
  labs(x = "Generation",
       y = "Variance Ratio") +
  theme_minimal() +
  theme(axis.title.x = element_text(size = 14),  # Increase x-axis label font size
                    axis.title.y = element_text(size = 14),  # Increase y-axis label font size
                    axis.text.x = element_text(size = 12),   # Increase x-axis tick marks font size
                    axis.text.y = element_text(size = 12),   # Increase y-axis tick marks font size
                    legend.text = element_text(size = 12),   # Increase legend text size
                    legend.title = element_text(size = 14)) +
  scale_color_manual(values = c("augment" = "blue", "subsample" = "green", "discard" = "red"))
ggsave("model_collapse_pisquareplot.png")