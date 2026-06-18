# Statistical Arbitrage Research Platform
# Phase 4: Spread Modeling (R)
#
# For each tradeable pair from phase 3:
#   1. Compute spread S_t = Price_A - β × Price_B
#   2. Rolling z-score with dynamic entry/exit bands
#   3. Annotate entry/exit signals on charts
#   4. Half-life sensitivity analysis
#   5. Spread stationarity diagnostics
#   6. Save results to CSV → phase4_save.py → MongoDB

setwd("/Users/khwahishgoel/quantproject")

library(mongolite)
library(ggplot2)
library(zoo)
library(dplyr)
library(tidyr)
library(tseries)
library(dotenv)
library(gridExtra)  # arrange multiple ggplots on one page

load_dot_env("src/.env")
MONGO_URI  <- Sys.getenv("MONGO_URI")
DB         <- "stat_arb"
OUTPUT_DIR <- "output"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

ROLL_WINDOW   <- 60     # days for rolling mean/sd
ENTRY_Z       <-  2.0   # enter trade when |z| > this
EXIT_Z        <-  0.5   # exit trade when |z| < this
STOP_Z        <-  3.5   # stop-loss when |z| > this

cat("── Phase 4: Spread Modeling ──\n\n")

# ── Load prices ───────────────────────────────────────────────────────────────
load_prices <- function() {
  col  <- mongo(collection = "prices", db = DB, url = MONGO_URI)
  docs <- col$find('{}', fields = '{"date":1,"ticker":1,"close":1,"_id":0}')
  col$disconnect()
  docs$date <- as.Date(format(as.POSIXct(docs$date, tz = "UTC"), "%Y-%m-%d"))
  prices <- docs %>%
    pivot_wider(names_from = ticker, values_from = close) %>%
    arrange(date)
  date_col      <- as.character(prices$date)
  prices$date   <- NULL
  prices        <- as.data.frame(prices)
  rownames(prices) <- date_col
  cat(sprintf("Loaded: %d days × %d tickers\n", nrow(prices), ncol(prices)))
  return(prices)
}

# ── Load tradeable pairs from MongoDB ─────────────────────────────────────────
load_tradeable_pairs <- function() {
  col   <- mongo(collection = "cointegration", db = DB, url = MONGO_URI)
  pairs <- col$find(
    '{"tradeable": true}',
    fields = '{"pair":1,"ticker_a":1,"ticker_b":1,"beta":1,"alpha":1,"eg_pvalue":1,"halflife_days":1,"_id":0}'
  )
  col$disconnect()
  cat(sprintf("Loaded %d tradeable pairs\n\n", nrow(pairs)))
  return(pairs)
}

# ── Compute spread and rolling z-score ────────────────────────────────────────
compute_spread <- function(price_a, price_b, beta, alpha) {
  # S_t = Price_A - β × Price_B  (demeaned by alpha)
  spread <- price_a - beta * price_b
  return(spread)
}

compute_zscore <- function(spread, window = ROLL_WINDOW) {
  roll_mean <- rollmean(spread, window, fill = NA, align = "right")
  roll_sd   <- rollapply(spread, window, sd,      fill = NA, align = "right")
  zscore    <- (spread - roll_mean) / roll_sd
  return(list(zscore = zscore, roll_mean = roll_mean, roll_sd = roll_sd))
}

# ── Generate trade signals ─────────────────────────────────────────────────────
# Returns a vector: "long", "short", "exit", or NA for each day
generate_signals <- function(zscore) {
  n       <- length(zscore)
  signals <- rep(NA_character_, n)
  in_trade <- FALSE
  direction <- NA_character_
  
  for (i in seq_along(zscore)) {
    z <- zscore[i]
    if (is.na(z)) next
    
    if (!in_trade) {
      if (z > ENTRY_Z) {
        signals[i] <- "short"   # spread too high → short A, long B
        in_trade   <- TRUE
        direction  <- "short"
      } else if (z < -ENTRY_Z) {
        signals[i] <- "long"    # spread too low → long A, short B
        in_trade   <- TRUE
        direction  <- "long"
      }
    } else {
      # Exit when z reverts toward 0 or hits stop-loss
      if (abs(z) < EXIT_Z || abs(z) > STOP_Z) {
        signals[i] <- "exit"
        in_trade   <- FALSE
        direction  <- NA_character_
      }
    }
  }
  return(signals)
}

# ── Half-life sensitivity ──────────────────────────────────────────────────────
# Tests how half-life changes over rolling 252-day windows
halflife_sensitivity <- function(spread, dates) {
  results <- list()
  n       <- length(spread)
  win     <- 252   # 1 year rolling
  
  for (i in seq(win, n, by = 21)) {   # step every ~1 month
    window_spread <- spread[(i - win + 1):i]
    window_spread <- na.omit(window_spread)
    if (length(window_spread) < 100) next
    
    delta  <- diff(window_spread)
    lagged <- window_spread[-length(window_spread)]
    tryCatch({
      model    <- lm(delta ~ lagged - 1)
      lambda   <- coef(model)[1]
      halflife <- -log(2) / log(1 + lambda)
      if (is.finite(halflife) && halflife > 0 && halflife < 500) {
        results[[length(results) + 1]] <- data.frame(
          date     = as.Date(dates[i]),
          halflife = round(halflife, 1)
        )
      }
    }, error = function(e) NULL)
  }
  return(do.call(rbind, results))
}

# ── Plots ──────────────────────────────────────────────────────────────────────
plot_spread_full <- function(dates, spread, zscore_list, signals, pair_name, beta, halflife) {
  df <- data.frame(
    date      = as.Date(as.character(dates)),
    spread    = as.numeric(spread),
    zscore    = as.numeric(zscore_list$zscore),
    roll_mean = as.numeric(zscore_list$roll_mean),
    roll_sd   = as.numeric(zscore_list$roll_sd)
  )
  
  # Signal points
  sig_df <- df %>%
    mutate(signal = signals) %>%
    filter(!is.na(signal))
  
  long_pts  <- sig_df %>% filter(signal == "long")
  short_pts <- sig_df %>% filter(signal == "short")
  exit_pts  <- sig_df %>% filter(signal == "exit")
  
  upper_band <- df$roll_mean + ENTRY_Z * df$roll_sd
  lower_band <- df$roll_mean - ENTRY_Z * df$roll_sd
  df$upper   <- upper_band
  df$lower   <- lower_band
  
  # ── Plot 1: Spread with rolling bands ──
  p1 <- ggplot(df, aes(x = date)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#E6F1FB", alpha = 0.5) +
    geom_line(aes(y = spread), color = "#378ADD", linewidth = 0.7) +
    geom_line(aes(y = roll_mean), color = "#888780", linewidth = 0.5, linetype = "dashed") +
    geom_line(aes(y = upper), color = "#E24B4A", linewidth = 0.4, linetype = "dotted") +
    geom_line(aes(y = lower), color = "#3B6D11", linewidth = 0.4, linetype = "dotted") +
    geom_point(data = long_pts,  aes(y = spread), color = "#3B6D11", shape = 24, size = 2.5, fill = "#3B6D11") +
    geom_point(data = short_pts, aes(y = spread), color = "#E24B4A", shape = 25, size = 2.5, fill = "#E24B4A") +
    geom_point(data = exit_pts,  aes(y = spread), color = "#888780", shape = 4,  size = 2,   stroke = 1.2) +
    labs(
      title    = sprintf("Spread: %s   |   β = %.4f   |   half-life = %.1f days", pair_name, beta, halflife),
      subtitle = sprintf("S_t = Price_%s − %.4f × Price_%s   |   Shaded: ±%.0f σ entry band   |   60-day rolling",
                         strsplit(pair_name, "/")[[1]][1], beta, strsplit(pair_name, "/")[[1]][2], ENTRY_Z),
      x = NULL, y = "Spread ($)"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title    = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 9,  color = "#5F5E5A"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#F1EFE8")
    )
  
  # ── Plot 2: Z-score with signals ──
  df_z <- df %>% filter(!is.na(zscore))
  sig_z <- data.frame(
    date   = as.Date(as.character(dates)),
    zscore = as.numeric(zscore_list$zscore),
    signal = signals
  ) %>% filter(!is.na(signal) & !is.na(zscore))
  
  long_z  <- sig_z %>% filter(signal == "long")
  short_z <- sig_z %>% filter(signal == "short")
  exit_z  <- sig_z %>% filter(signal == "exit")
  
  p2 <- ggplot(df_z, aes(x = date, y = zscore)) +
    geom_ribbon(aes(ymin = -ENTRY_Z, ymax = ENTRY_Z), fill = "#EAF3DE", alpha = 0.35) +
    geom_ribbon(aes(ymin =  ENTRY_Z, ymax =  STOP_Z),  fill = "#FAEEDA", alpha = 0.25) +
    geom_ribbon(aes(ymin = -STOP_Z,  ymax = -ENTRY_Z), fill = "#FAEEDA", alpha = 0.25) +
    geom_hline(yintercept = c( ENTRY_Z,  STOP_Z), color = "#E24B4A", linetype = c("dashed","dotted"), linewidth = 0.5) +
    geom_hline(yintercept = c(-ENTRY_Z, -STOP_Z), color = "#3B6D11", linetype = c("dashed","dotted"), linewidth = 0.5) +
    geom_hline(yintercept = c(-EXIT_Z,   EXIT_Z), color = "#888780", linetype = "dashed", linewidth = 0.4) +
    geom_hline(yintercept = 0,                     color = "#888780", linetype = "solid",  linewidth = 0.3) +
    geom_line(color = "#534AB7", linewidth = 0.7) +
    geom_point(data = long_z,  aes(y = zscore), color = "#3B6D11", shape = 24, size = 2.5, fill = "#3B6D11") +
    geom_point(data = short_z, aes(y = zscore), color = "#E24B4A", shape = 25, size = 2.5, fill = "#E24B4A") +
    geom_point(data = exit_z,  aes(y = zscore), color = "#888780", shape = 4,  size = 2,   stroke = 1.2) +
    annotate("text", x = min(df_z$date), y =  ENTRY_Z + 0.15, label = sprintf("short entry  (z > %.1f)",  ENTRY_Z), hjust = 0, size = 2.8, color = "#A32D2D") +
    annotate("text", x = min(df_z$date), y = -ENTRY_Z - 0.15, label = sprintf("long entry   (z < -%.1f)", ENTRY_Z), hjust = 0, size = 2.8, color = "#27500A", vjust = 1) +
    annotate("text", x = min(df_z$date), y =  EXIT_Z  + 0.08, label = sprintf("exit zone  |z| < %.1f",   EXIT_Z),  hjust = 0, size = 2.5, color = "#5F5E5A") +
    coord_cartesian(ylim = c(-5, 5)) +
    labs(
      title    = sprintf("Z-Score: %s", pair_name),
      subtitle = sprintf("z_t = (S_t − μ) / σ   |   60-day rolling window   |   ▲ long   ▼ short   ✕ exit"),
      x = NULL, y = "Z-Score (σ)"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title    = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 9,  color = "#5F5E5A"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#F1EFE8")
    )
  
  # ── Save combined chart ──
  fname <- file.path(OUTPUT_DIR, sprintf("phase4_%s.png", gsub("/", "_", pair_name)))
  ggsave(fname, arrangeGrob(p1, p2, ncol = 1), width = 12, height = 8, dpi = 150)
  cat(sprintf("  Saved → %s\n", fname))
}

plot_halflife_sensitivity <- function(hl_df, pair_name, base_hl) {
  if (is.null(hl_df) || nrow(hl_df) == 0) return()
  
  p <- ggplot(hl_df, aes(x = date, y = halflife)) +
    geom_line(color = "#185FA5", linewidth = 0.8) +
    geom_hline(yintercept = base_hl, color = "#E24B4A", linetype = "dashed", linewidth = 0.6) +
    geom_hline(yintercept = 126,     color = "#888780", linetype = "dotted", linewidth = 0.5) +
    geom_hline(yintercept = 5,       color = "#888780", linetype = "dotted", linewidth = 0.5) +
    geom_ribbon(aes(ymin = 5, ymax = 126), fill = "#EAF3DE", alpha = 0.15) +
    annotate("text", x = min(hl_df$date), y = 130, label = "max tradeable (126d)", hjust = 0, size = 3, color = "#5F5E5A") +
    annotate("text", x = min(hl_df$date), y = 1,   label = "min tradeable (5d)",   hjust = 0, size = 3, color = "#5F5E5A") +
    annotate("text", x = max(hl_df$date), y = base_hl + 3,
             label = sprintf("full-period: %.1fd", base_hl), hjust = 1, size = 3, color = "#A32D2D") +
    labs(
      title    = sprintf("Half-Life Sensitivity: %s", pair_name),
      subtitle = "Rolling 252-day estimation — how stable is the mean-reversion speed?",
      x = NULL, y = "Half-Life (days)"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title    = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 9,  color = "#5F5E5A"),
      panel.grid.minor = element_blank()
    )
  
  fname <- file.path(OUTPUT_DIR, sprintf("phase4_halflife_%s.png", gsub("/", "_", pair_name)))
  ggsave(fname, p, width = 10, height = 4, dpi = 150)
  cat(sprintf("  Saved → %s\n", fname))
}

# ── Main ──────────────────────────────────────────────────────────────────────
prices <- load_prices()
pairs  <- load_tradeable_pairs()
results <- list()

for (i in seq_len(nrow(pairs))) {
  a        <- pairs$ticker_a[i]
  b        <- pairs$ticker_b[i]
  pair     <- pairs$pair[i]
  beta     <- pairs$beta[i]
  alpha    <- pairs$alpha[i]
  halflife <- pairs$halflife_days[i]
  
  cat(sprintf("── Modeling pair: %s  (β=%.4f, hl=%.1fd) ──\n", pair, beta, halflife))
  
  price_a <- as.numeric(prices[[a]])
  price_b <- as.numeric(prices[[b]])
  dates   <- rownames(prices)
  
  valid   <- !is.na(price_a) & !is.na(price_b)
  price_a <- price_a[valid]
  price_b <- price_b[valid]
  dates   <- dates[valid]
  
  # Step 1: spread
  spread <- compute_spread(price_a, price_b, beta, alpha)
  
  # Step 2: rolling z-score
  zs <- compute_zscore(spread)
  
  # Step 3: signals
  signals <- generate_signals(zs$zscore)
  
  n_long  <- sum(signals == "long",  na.rm = TRUE)
  n_short <- sum(signals == "short", na.rm = TRUE)
  n_exit  <- sum(signals == "exit",  na.rm = TRUE)
  cat(sprintf("  Signals — long: %d, short: %d, exits: %d\n", n_long, n_short, n_exit))
  
  # Step 4: half-life sensitivity
  hl_df <- halflife_sensitivity(spread, dates)
  hl_mean <- if (!is.null(hl_df) && nrow(hl_df) > 0) round(mean(hl_df$halflife, na.rm = TRUE), 1) else NA
  hl_sd   <- if (!is.null(hl_df) && nrow(hl_df) > 0) round(sd(hl_df$halflife,   na.rm = TRUE), 1) else NA
  cat(sprintf("  Half-life stability — mean: %.1fd, sd: %.1fd\n", hl_mean, hl_sd))
  
  # Step 5: spread diagnostics
  spread_mean <- round(mean(spread, na.rm = TRUE), 4)
  spread_sd   <- round(sd(spread,   na.rm = TRUE), 4)
  current_z   <- round(tail(na.omit(zs$zscore), 1), 4)
  cat(sprintf("  Current z-score: %.4f\n", current_z))
  
  # Step 6: plots
  plot_spread_full(dates, spread, zs, signals, pair, beta, halflife)
  plot_halflife_sensitivity(hl_df, pair, halflife)
  
  results[[i]] <- data.frame(
    pair          = pair,
    ticker_a      = a,
    ticker_b      = b,
    beta          = beta,
    spread_mean   = spread_mean,
    spread_sd     = spread_sd,
    current_zscore = current_z,
    n_long_signals = n_long,
    n_short_signals = n_short,
    n_exits        = n_exit,
    halflife_mean  = hl_mean,
    halflife_sd    = hl_sd,
    roll_window    = ROLL_WINDOW,
    entry_z        = ENTRY_Z,
    exit_z         = EXIT_Z,
    modeled_at     = as.character(Sys.time())
  )
  cat("\n")
}

results_df <- do.call(rbind, results)

cat("── Spread modeling summary ──\n")
print(results_df[, c("pair", "current_zscore", "n_long_signals", "n_short_signals", "halflife_mean")])

csv_path <- file.path(OUTPUT_DIR, "spread_model_results.csv")
write.csv(results_df, csv_path, row.names = FALSE)
cat(sprintf("\nSaved → %s\n", csv_path))
cat("Run phase4_save.py next to push results to MongoDB.\n")