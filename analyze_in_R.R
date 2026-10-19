# Independent research project: A computational experiment in referential choice
# A full analysis of the INCLUDED synthetic trial-level dataset.
# Run from the root of the unpacked folder:
#     Rscript code/analyze_in_R.R
# Requires base R. Optional packages: ggplot2, sandwich, lme4.
# This R code has been reviewed but NOT executed in the package preparation environment
# (R was not available). Results in the report are from a verified Python mirror.

options(stringsAsFactors = FALSE)
set.seed(71173)
input <- "data/synthetic_reference_trials.csv"
stopifnot(file.exists(input))
d <- read.csv(input)
required <- c("participant", "item", "shared", "surprise", "explicit_choice", "speaker_p_explicit")
stopifnot(all(required %in% colnames(d)))
stopifnot(nrow(d) == 2304, all(d$explicit_choice %in% 0:1))
dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

# Four conditions. No human subjects took part: 'participants' are virtual agents.
summary <- aggregate(d$explicit_choice,
     list(common_ground = d$common_ground, event_type = d$event_type),
     FUN = function(z) c(n = length(z), full_np = sum(z), rate = mean(z)))
summary_out <- data.frame(summary[1:2], summary$x)
names(summary_out)[3:5] <- c("n", "full_np", "rate")
write.csv(summary_out, "results/condition_summary_R.csv", row.names = FALSE)
print(summary_out)

# Logistic regression adjusting for agent and item fixed effects.
# This is equivalent in model formula to the independent Python verification.
m <- glm(explicit_choice ~ surprise * shared + factor(participant) + factor(item),
         data = d, family = binomial(link = "logit"))
coef_tab <- coef(summary(m))[c("surprise", "shared", "surprise:shared"), , drop = FALSE]
print(coef_tab)
write.csv(data.frame(term = rownames(coef_tab), coef_tab),
          "results/model_coefficients_R_naiveSE.csv", row.names = FALSE)

# Cluster-robust SE by virtual participant, if sandwich is installed.
# With 48 clusters this is a useful supplement to naive GLM SEs.
if (requireNamespace("sandwich", quietly = TRUE)) {
    V <- sandwich::vcovCL(m, cluster = d$participant, type = "HC1")
    keys <- c("surprise", "shared", "surprise:shared")
    b <- coef(m)[keys]
    se <- sqrt(diag(V))[keys]
    z <- b / se
    out <- data.frame(term = keys, log_odds = b,
         cluster_SE = se, z = z, p_value = 2 * pnorm(-abs(z)),
         odds_ratio = exp(b))
    write.csv(out, "results/model_coefficients_R_clustered.csv", row.names = FALSE)
    print(out)
} else {
    message("Optional package 'sandwich' not installed; clustered SEs not computed.")
}

# Participant-block bootstrap for contrasts. Resample whole participants,
# not single trials, to preserve the within-participant 2x2 design.
by_p <- aggregate(explicit_choice ~ participant + shared + surprise, d, mean)
ids <- unique(by_p$participant)
X <- matrix(NA_real_, nrow = length(ids), ncol = 4)
colnames(X) <- c("shared_surprise", "shared_expected", "notshared_surprise", "notshared_expected")
for (i in seq_along(ids)) {
    z <- by_p[by_p$participant == ids[i], ]
    pick <- function(g, s) z$explicit_choice[z$shared == g & z$surprise == s]
    X[i, ] <- c(pick(1,1), pick(1,0), pick(0,1), pick(0,0))
}
contrasts <- function(M) c(
   shared_surprise_effect = mean(M[,1] - M[,2]),
   notshared_surprise_effect = mean(M[,3] - M[,4]),
   interaction_DID = mean(M[,1] - M[,2] - M[,3] + M[,4]))
point <- contrasts(X)
boot <- replicate(3000, contrasts(X[sample.int(nrow(X), nrow(X), TRUE), , drop = FALSE]))
ci <- t(apply(boot, 1, quantile, probs = c(.025,.975), names = FALSE))
res <- data.frame(contrast = names(point), estimate_pp = as.numeric(point)*100,
       lower_95_pp = ci[,1]*100, upper_95_pp = ci[,2]*100)
write.csv(res, "results/cluster_bootstrap_contrasts_R.csv", row.names = FALSE)
print(res)

# Optional mixed-effects logistic sensitivity check:
if (requireNamespace("lme4", quietly = TRUE)) {
    mixed <- lme4::glmer(explicit_choice ~ surprise * shared +
            (1 | participant) + (1 | item),
            data = d, family = binomial(link = "logit"),
            control = lme4::glmerControl(optimizer = "bobyqa"))
    capture.output(summary(mixed), file = "results/mixed_effects_R.txt")
} else {
    message("Optional package 'lme4' not installed; GLMM check skipped.")
}

# Base R plot: full NP probability by event and common ground.
png("figures/condition_plot_from_R.png", width=1100, height=700, res=145)
vals <- matrix(NA_real_, nrow = 2, ncol = 2,
               dimnames = list(c("Shared","Not shared"), c("Predictable","Surprising")))
for (g in c(0,1)) for (s in c(0,1)) {
   vals[ifelse(g==1, 1,2), ifelse(s==0, 1,2)] <-
    mean(d$explicit_choice[d$shared==g & d$surprise==s])
}
barplot(t(vals), beside=TRUE, col=c("#216C7A","#EAAE5B"),
        ylim=c(0,1), ylab="Proportion of explicit noun phrases",
        main="Model-generated reference choices")
legend("topleft",legend=colnames(vals),fill=c("#216C7A","#EAAE5B"), bty="n")
dev.off()
cat("\nCompleted R analysis (simulation, not human experiment).\n")
