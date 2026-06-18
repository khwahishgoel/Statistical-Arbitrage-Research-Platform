# Statistical Arbitrage Research Platform
# Phase 3: Cointegration Testing (R)
#
# Reads candidate pairs from MongoDB, runs:
#   1. ADF test on each price series (confirm non-stationarity)
#   2. Engle-Granger cointegration test on each pair
#   3. Half-life of mean reversion estimation
# Writes results to CSV → picked up by phase3_save.py → MongoDB

library(mongolite)
library(tseries)
library(urca)
library(ggplot2)
library(zoo)
library(dplyr)
library(tidyr)
library(dotenv)

# ── Config ────────────────────────────────────────────────────────────────────
load_dot_env("src/.env")                         # reads MONGO_URI from .env
MONGO_URI  <- Sys.getenv("MONGO_URI")
DB         <- "stat_arb"
PVALUE_THRESH  <- 0.10                      # cointegration significance level
MIN_HALFLIFE   <- 5                          # days — too fast = microstructure noise
MAX_HALFLIFE   <- 126                        # days — too slow = capital tied up too long
OUTPUT_DIR <- "output"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

cat("── Phase 3: Cointegration Testing ──\n\n")

# ── Load prices from MongoDB ──────────────────────────────────────────────────
# replace the whole load_prices function with this
load_prices <- function() {
  cat("Connecting to MongoDB Atlas...\n")
  col  <- mongo(collection = "prices", db = DB, url = MONGO_URI)
  docs <- col$find('{}', fields = '{"date":1,"ticker":1,"close":1,"_id":0}')
  col$disconnect()
  
  docs$date <- as.Date(format(as.POSIXct(docs$date, tz="UTC"), "%Y-%m-%d"))
  
  prices <- docs %>%
    tidyr::pivot_wider(names_from = ticker, values_from = close) %>%
    arrange(date)
  
  # keep date as a proper column for rownames
  date_col <- as.character(prices$date)
  prices$date <- NULL
  prices <- as.data.frame(prices)
  rownames(prices) <- date_col
  
  cat(sprintf("Loaded: %d days × %d tickers\n\n", nrow(prices), ncol(prices)))
  return(prices)
}

# ── Load candidate pairs from MongoDB ────────────────────────────────────────
load_pairs <- function() {
  col   <- mongo(collection = "candidate_pairs", db = DB, url = MONGO_URI)
  pairs <- col$find('{}', fields = '{"pair":1,"ticker_a":1,"ticker_b":1,"full_corr":1,"_id":0}')
  col$disconnect()
  cat(sprintf("Loaded %d candidate pairs from MongoDB\n\n", nrow(pairs)))
  return(pairs)
}

# ── ADF Test (Augmented Dickey-Fuller) ───────────────────────────────────────
# Tests whether a price series is non-stationary (has a unit root).
# We WANT this to fail (p > 0.05) — prices should be non-stationary.
# If prices were stationary, they'd mean-revert on their own — no arb needed.
run_adf <- function(series, name) {
  result <- adf.test(na.omit(series), alternative = "stationary")
  stationary <- result$p.value < 0.05
  cat(sprintf("  ADF %s: p=%.4f → %s\n",
              name, result$p.value,
              ifelse(stationary, "STATIONARY (unexpected)", "non-stationary (good)")))
  return(list(pvalue = result$p.value, stationary = stationary))
}

# ── Engle-Granger Cointegration Test ─────────────────────────────────────────
# Step 1: Regress price_A on price_B to get hedge ratio β
# Step 2: Run ADF on the residuals (the spread)
# If residuals are stationary → the pair is cointegrated → spread mean-reverts
run_engle_granger <- function(price_a, price_b, name_a, name_b) {
  # OLS regression to estimate hedge ratio β
  model   <- lm(price_a ~ price_b)
  beta    <- coef(model)[2]           # hedge ratio
  alpha   <- coef(model)[1]           # intercept
  resids  <- residuals(model)         # the spread
  
  # ADF test on the residuals
  adf_res <- adf.test(na.omit(resids), alternative = "stationary")
  pvalue  <- adf_res$p.value
  cointegrated <- pvalue < PVALUE_THRESH
  
  cat(sprintf("  EG test %s/%s: β=%.4f, p=%.4f → %s\n",
              name_a, name_b, beta, pvalue,
              ifelse(cointegrated, "COINTEGRATED ✓", "not cointegrated ✗")))
  
  return(list(
    beta         = beta,
    alpha        = alpha,
    residuals    = resids,
    eg_pvalue    = pvalue,
    cointegrated = cointegrated
  ))
}

# ── Half-Life of Mean Reversion ───────────────────────────────────────────────
# Fits an AR(1) model to the spread: ΔS_t = λ·S_{t-1} + ε
# Half-life = -log(2) / log(1 + λ)
# Tells you how long (in days) the spread takes to revert halfway to its mean.
# This determines your holding period — too short = noise, too long = capital trap.
estimate_halflife <- function(spread) {
  spread     <- na.omit(spread)
  delta      <- diff(spread)
  lagged     <- spread[-length(spread)]
  model      <- lm(delta ~ lagged - 1)   # force through origin
  lambda     <- coef(model)[1]
  halflife   <- -log(2) / log(1 + lambda)
  return(round(halflife, 1))
}

# ── Z-Score of Spread ─────────────────────────────────────────────────────────
compute_zscore <- function(spread, window = 60) {
  roll_mean <- rollmean(spread, window, fill = NA, align = "right")
  roll_sd   <- rollapply(spread, window, sd, fill = NA, align = "right")
  zscore    <- (spread - roll_mean) / roll_sd
  return(zscore)
}

# ── Plot: Spread + Z-Score ────────────────────────────────────────────────────
plot_spread <- function(dates, spread, zscore, pair_name) {
  df <- data.frame(
    date   = as.Date(as.POSIXct(dates, tz = "UTC")),
    spread = as.numeric(spread),
    zscore = as.numeric(zscore)
  ) %>% filter(!is.na(zscore))
  
  # Spread plot
  p1 <- ggplot(df, aes(x = date, y = spread)) +
    geom_line(color = "#378ADD", linewidth = 0.7) +
    geom_hline(yintercept = mean(df$spread), color = "#888780", linetype = "dashed", linewidth = 0.5) +
    labs(title = sprintf("Spread: %s", pair_name),
         subtitle = "S_t = Price_A − β × Price_B",
         x = NULL, y = "Spread ($)") +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(size = 12, face = "bold"),
          plot.subtitle = element_text(size = 10, color = "#5F5E5A"),
          panel.grid.minor = element_blank())
  
  # Z-score plot with entry/exit bands
  p2 <- ggplot(df, aes(x = date, y = zscore)) +
    geom_ribbon(aes(ymin = -2, ymax = 2), fill = "#E6F1FB", alpha = 0.4) +
    geom_line(color = "#534AB7", linewidth = 0.7) +
    geom_hline(yintercept =  2.0, color = "#E24B4A", linetype = "dashed", linewidth = 0.6) +
    geom_hline(yintercept = -2.0, color = "#3B6D11", linetype = "dashed", linewidth = 0.6) +
    geom_hline(yintercept =  0.0, color = "#888780", linetype = "solid",  linewidth = 0.4) +
    annotate("text", x = min(df$date), y = 2.15,  label = "short signal (z > 2)",  hjust = 0, size = 3, color = "#A32D2D") +
    annotate("text", x = min(df$date), y = -2.15, label = "long signal  (z < -2)", hjust = 0, size = 3, color = "#3B6D11") +
    labs(title = sprintf("Z-Score: %s", pair_name),
         subtitle = "z_t = (S_t − μ) / σ   |   60-day rolling window",
         x = NULL, y = "Z-Score") +
    ylim(-4, 4) +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(size = 12, face = "bold"),
          plot.subtitle = element_text(size = 10, color = "#5F5E5A"),
          panel.grid.minor = element_blank())
  
  # Save both charts
  fname1 <- file.path(OUTPUT_DIR, sprintf("spread_%s.png",   gsub("/", "_", pair_name)))
  fname2 <- file.path(OUTPUT_DIR, sprintf("zscore_%s.png",   gsub("/", "_", pair_name)))
  ggsave(fname1, p1, width = 10, height = 3.5, dpi = 150)
  ggsave(fname2, p2, width = 10, height = 3.5, dpi = 150)
  cat(sprintf("    Saved charts → %s, %s\n", fname1, fname2))
}

# ── Main ──────────────────────────────────────────────────────────────────────
prices <- load_prices()
pairs  <- load_pairs()
results <- list()

for (i in seq_len(nrow(pairs))) {
  a    <- pairs$ticker_a[i]
  b    <- pairs$ticker_b[i]
  pair <- pairs$pair[i]
  
  cat(sprintf("── Testing pair: %s ──\n", pair))
  
  price_a <- as.numeric(prices[[a]])
  price_b <- as.numeric(prices[[b]])
  dates   <- rownames(prices)
  
  # Remove NAs (should be minimal after phase 1 cleaning)
  valid   <- !is.na(price_a) & !is.na(price_b)
  price_a <- price_a[valid]
  price_b <- price_b[valid]
  dates   <- dates[valid]
  
  # Step 1: ADF on raw prices (confirm non-stationarity)
  adf_a <- run_adf(price_a, a)
  adf_b <- run_adf(price_b, b)
  
  # Step 2: Engle-Granger cointegration test
  eg <- run_engle_granger(price_a, price_b, a, b)
  
  # Step 3: Half-life (only if cointegrated)
  halflife <- NA
  tradeable <- FALSE
  if (eg$cointegrated) {
    halflife  <- estimate_halflife(eg$residuals)
    tradeable <- halflife >= MIN_HALFLIFE & halflife <= MAX_HALFLIFE
    cat(sprintf("  Half-life: %.1f days → %s\n", halflife,
                ifelse(tradeable, "tradeable ✓", "out of range ✗")))
    
    # Step 4: Z-score and charts (only for tradeable pairs)
    if (tradeable) {
      spread <- price_a - eg$beta * price_b
      zscore <- compute_zscore(spread)
      plot_spread(dates, spread, zscore, pair)
    }
  }
  
  results[[i]] <- data.frame(
    pair             = pair,
    ticker_a         = a,
    ticker_b         = b,
    full_corr        = pairs$full_corr[i],
    beta             = round(eg$beta, 6),
    alpha            = round(eg$alpha, 6),
    eg_pvalue        = round(eg$eg_pvalue, 6),
    cointegrated     = eg$cointegrated,
    halflife_days    = halflife,
    tradeable        = tradeable,
    adf_pvalue_a     = round(adf_a$pvalue, 6),
    adf_pvalue_b     = round(adf_b$pvalue, 6),
    tested_at        = as.character(Sys.time())
  )
  cat("\n")
}

# ── Save results ──────────────────────────────────────────────────────────────
results_df <- do.call(rbind, results)

cat("── Results summary ──\n")
print(results_df[, c("pair", "eg_pvalue", "cointegrated", "halflife_days", "tradeable")])

csv_path <- file.path(OUTPUT_DIR, "cointegration_results.csv")
write.csv(results_df, csv_path, row.names = FALSE)
cat(sprintf("\nSaved → %s\n", csv_path))
cat("Run phase3_save.py next to push results to MongoDB.\n")
