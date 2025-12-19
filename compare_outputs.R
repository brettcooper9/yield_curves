# R Script to Generate Test Outputs for Comparison with Python
# Run this script, then compare with Python notebook outputs

library(lubridate)

# Source the original helper functions
source("helper_fn.R")

cat("========================================\n")
cat("R Test Outputs for Python Comparison\n")
cat("========================================\n\n")

# Test 1: Monthly Bond Return
cat("Test 1: Monthly Bond Return\n")
cat("----------------------------\n")
test_yields <- c(2.5, 2.6, 2.4, 2.7, 2.5)
cat("Yields:", test_yields, "\n\n")

for(i in 2:length(test_yields)) {
  ret <- monthly_bond_ret(test_yields, i)
  cat(sprintf("Month %d: %.10f (%.6f%%)\n", i-1, ret, ret*100))
}

# Test 2: Time to Maturity
cat("\n\nTest 2: Time to Maturity\n")
cat("----------------------------\n")
test_cases <- list(
  c("2020-01-15", "2025-01-15"),
  c("2020-01-15", "2020-07-15"),
  c("2018-12-31", "2026-12-31")
)

for(tc in test_cases) {
  ttm <- round_mo_to_maturity(tc[1], tc[2], NUM_DIG_MATURITY=4)
  cat(sprintf("%s to %s: %.4f years\n", tc[1], tc[2], ttm))
}

# Test 3: Month-End Series
cat("\n\nTest 3: Month-End Series Generation\n")
cat("----------------------------\n")
orig_date <- as.Date("2018-12-31")
mat_date <- as.Date("2026-12-31")

xts_res <- generate_month_end_xts(orig_date, mat_date, cnames=c("Value"))
cat(sprintf("Generated %d month-end dates\n", nrow(xts_res)))
cat(sprintf("First date: %s\n", index(xts_res)[1]))
cat(sprintf("Last date: %s\n", index(xts_res)[length(index(xts_res))]))
cat("\nFirst 5 dates:\n")
print(index(xts_res)[1:5])

# Test 4: Comprehensive Bond Return Test
cat("\n\nTest 4: Comprehensive Bond Return Test\n")
cat("----------------------------\n")
comp_yields <- c(2.0, 2.1, 2.2, 2.3, 2.4, 2.5)
cat("Yields:", comp_yields, "\n\n")

results <- data.frame(
  Index = 1:(length(comp_yields)-1),
  Prior_Yield = comp_yields[-length(comp_yields)],
  Current_Yield = comp_yields[-1],
  Return = sapply(2:length(comp_yields), function(i) monthly_bond_ret(comp_yields, i))
)

print(results, digits=10, row.names=FALSE)

cat("\n\nR test outputs complete!\n")
cat("Compare these results with the Python notebook outputs.\n")
