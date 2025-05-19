#' Convert a TukeyHSD matrix to an N×N LaTeX table of effect sizes
#'
#' @param tukey_mat A numeric matrix, e.g. TukeyHSD(aov_model)$factor
#'                  with rownames like "B-A" and columns including "diff" and "p adj".
#' @param alpha     Significance threshold (default 0.05).
#' @param digits    Number of digits to display (default 2).
#' @param caption   LaTeX table caption.
#' @param label     LaTeX table label.
#' @return          Invisibly returns the LaTeX code (also prints it to stdout).
#' @examples
#'   aov_mod <- aov(y ~ group, data = d)
#'   tuk <- TukeyHSD(aov_mod)$group
#'   tukey_to_latex(tuk, alpha = 0.05, digits = 3,
#'                 caption = "Pairwise differences", label = "tab:tukey")
                

ACL_TEXTWIDTH = (455.24411/72.27)
ACL_COLUMNWIDTH = (219.08614/72.27)

PLOT_TXT = 2.9*.35
SMALL_TXT = 8
LABEL_TXT = 10
LINE_SZ = .25
SPACING = 1.1
KEY_HT = unit(1, "mm")
LEGEND_SPACE = unit(2, "mm") 


tukey_to_latex <- function(tukey_mat,
                           alpha   = 0.05,
                           digits  = 2,
                           caption = "Tukey HSD pairwise differences",
                           label   = "tab:tukey") {
  # if they passed the whole TukeyHSD() object, grab the first component
  if (inherits(tukey_mat, "TukeyHSD")) {
    tukey_mat <- tukey_mat[[1]]
  }
  # must now be a numeric matrix with rownames
  if (!is.matrix(tukey_mat) ||
      !is.numeric(tukey_mat) ||
      is.null(rownames(tukey_mat))) {
    stop("`tukey_mat` must be a numeric matrix (e.g. `TukeyHSD(aov)$factor`).")
  }
  
  # ensure rownames are character
  rn   <- as.character(rownames(tukey_mat))
  comps <- do.call(rbind, strsplit(rn, "-", fixed = TRUE))
  if (ncol(comps) != 2) {
    stop("Row names must be of the form 'Level2-Level1'.")
  }
  colnames(comps) <- c("level2","level1") 
  
  diffs <- tukey_mat[ , "diff"]
  pvals <- tukey_mat[ , "p adj"]
  
  levs <- sort(unique(c(comps)))
  n    <- length(levs)
  M    <- matrix("", n, n, dimnames = list(levs, levs))
  M[cbind(levs, levs)] <- "---"
  
  for (k in seq_len(nrow(comps))) {
    i <- comps[k,"level2"]; j <- comps[k,"level1"]
    d <- round(diffs[k], digits)
    txt <- formatC(d, format="f", digits=digits)
    if (pvals[k] < alpha) {
      txt <- paste0("\\cellcolor{gray!25}\\textbf{", txt, "}")
    }
    M[i,j] <- txt
    # symmetric entry flips the sign
    negtxt <- formatC(-d, format="f", digits=digits)
    if (pvals[k] < alpha) negtxt <- paste0("\\cellcolor{gray!25}\\textbf{", negtxt, "}")
    M[j,i] <- negtxt
  }
  
  # build LaTeX
  cols <- paste0("l", strrep("c", n))
  tex  <- c(
    "% needs: \\usepackage{booktabs,colortbl,xcolor}",
    "\\begin{table}[ht]",
    "\\centering",
    sprintf("\\caption{%s}", caption),
    sprintf("\\label{%s}", label),
    sprintf("\\begin{tabular}{%s}", cols),
    "\\toprule",
    paste(c("", levs), collapse=" & "), "\\\\",
    "\\midrule"
  )
  for (i in levs) {
    tex <- c(tex, paste(c(i, M[i,]), collapse=" & "), "\\\\")
  }
  tex <- c(tex, "\\bottomrule", "\\end{tabular}", "\\end{table}")
  
  cat(paste(tex, collapse="\n"), "\n")
  invisible(tex)
}

tukey_to_latex_heatcell <- function(tukey_mat,
                                    alpha   = 0.05,
                                    digits  = 2,
                                    caption = "Tukey HSD: effect sizes",
                                    label   = "tab:tukey") {
  # allow passing whole TukeyHSD object
  if (inherits(tukey_mat, "TukeyHSD")) tukey_mat <- tukey_mat[[1]]
  stopifnot(is.matrix(tukey_mat),
            is.numeric(tukey_mat),
            !is.null(rownames(tukey_mat)))
  
  # parse comparisons
  rn    <- as.character(rownames(tukey_mat))
  comps <- do.call(rbind, strsplit(rn, "-", fixed = TRUE))
  if (ncol(comps)!=2)
    stop("Row names must be 'Level2-Level1'")
  colnames(comps) <- c("to","from")
  
  diffs <- tukey_mat[,"diff"]
  pvals <- tukey_mat[,"p adj"]
  
  # levels and empty matrix
  levs <- sort(unique(c(comps)))
  n    <- length(levs)
  M    <- matrix("", n, n, dimnames = list(levs, levs))
  diag(M) <- "---"
  
  # fill both triangles with \heatcell
  for (k in seq_len(nrow(comps))) {
    i <- comps[k,"to"]
    j <- comps[k,"from"]
    d <- round(diffs[k], digits)
    fmt <- sprintf(paste0("%%.%df"), digits)
    d_txt <- sprintf(fmt, d)
    p_txt <- sprintf("%.3f", pvals[k])
    cell_ij <- sprintf("\\heatcell{%s}{%s}", d_txt, p_txt)
    # symmetric entry flips sign
    d2_txt <- sprintf(fmt, -d)
    cell_ji <- sprintf("\\heatcell{%s}{%s}", d2_txt, p_txt)
    M[i,j] <- cell_ij
    M[j,i] <- cell_ji
  }
  
  # build LaTeX table
  cols <- paste0("l", strrep("c", n))
  tex  <- c(
    "% requires: xcolor, pgfmath and your \\heatcell macro",
    "\\begin{table}[ht]",
    "\\centering",
    sprintf("\\caption{%s}", caption),
    sprintf("\\label{%s}", label),
    sprintf("\\begin{tabular}{%s}", cols),
    "\\toprule",
    paste(c("", levs), collapse = " & "), "\\\\",
    "\\midrule"
  )
  
  for (i in levs) {
    tex <- c(tex,
             paste(c(i, M[i,]), collapse = " & "),
             "\\\\")
  }
  
  tex <- c(tex,
           "\\bottomrule",
           "\\end{tabular}",
           "\\end{table}")
  
  cat(paste(tex, collapse = "\n"), "\n")
  invisible(tex)
}


filter_japanese <- function(dataset){
  new_dataset <- dataset %>% filter(substring(language, 1, 4)!='jpn_')
  return(new_dataset)
}

cnn_orth_df <- read.csv('granular_orth_cnn_small_1layer_ker3.csv') %>% mutate(set.type = 'Orthography to Phonology') %>% 
  mutate(compressibility = mutual.algorithmic.information/best.complexity)%>% mutate_at(vars(trial.number), factor) %>% 
  mutate(transcription.type = ifelse(language=='kor', 'narrow', transcription.type)) %>%
  filter(data.size == max(data.size)) %>% filter_japanese()
cnn_phon_df <- read.csv('granular_phon_cnn_small_1layer_ker3.csv') %>% mutate(set.type = 'Phonology to Orthography') %>% 
  mutate(compressibility = mutual.algorithmic.information/best.complexity)%>% mutate_at(vars(trial.number), factor) %>% 
  mutate(transcription.type = ifelse(language=='kor', 'narrow', transcription.type)) %>%
  filter(data.size == max(data.size)) %>% filter_japanese()


tidy_tukey <- function(x) {
  mat <- if (inherits(x, "TukeyHSD")) x[[1]] else x
  tibble::as_tibble(mat, rownames = "cmp") %>%
    separate(cmp, into = c("group2","group1"), sep = "-", remove = TRUE) %>%
    rename(diff = diff, p.adj = `p adj`) %>%
    mutate(
      group1 = factor(group1, levels = sort(unique(c(group1,group2)))),
      group2 = factor(group2, levels = levels(group1))
    )
}

#–– 2. Complete to a full N×N grid
make_full_grid <- function(df_tuk) {
  levs <- levels(df_tuk$group1)
  expand_grid(
    group1 = factor(levs, levels = levs),
    group2 = factor(levs, levels = levs)
  ) %>%
    left_join(df_tuk, by = c("group1","group2")) %>%
    replace_na(list(diff = 0, p.adj = 1))
}

#–– 3. One‐stop plotting function
plot_tukey_full <- function(x, alpha = 0.05, digits = 2) {
  df <- tidy_tukey(x)
  # build full N×N grid
  levs <- levels(df$group1)
  df_full <- expand_grid(
    group1 = factor(levs, levels = levs),
    group2 = factor(levs, levels = levs)
  ) %>% 
    left_join(df, by = c("group1","group2")) %>%
    drop_na()
  
  df_rev = df_full %>% rename(group_1 = group2) %>% 
    rename(group2 = group1) %>% 
    rename(group1 = group_1) %>% 
    mutate(diff = -diff)
  
  df_full = rbind(df_rev, df_full)
  
  ggplot(df_full, aes(x = group1, y = group2, group = mean(lwr))) +
    # full‐matrix tiles
    geom_tile(aes(fill = abs(diff)), color = "white") +
    scale_fill_gradient2(
      low      = "steelblue",
      mid      = "yellow",
      high     = "red",
      midpoint = 0.2,
      name     = "Absolute Difference",
      limits=c(0.0, .65)
    ) +
    # overlay signed diff, bold if significant
    geom_text(aes(
      label    = sprintf(paste0("%.", digits, "f"), diff),
      fontface = ifelse(p.adj < alpha, "bold", "plain")
    ), size = PLOT_TXT) +
    coord_fixed() +
    theme_minimal() + #(base_size = 14) +
    labs(
      x     = NULL,
      y     = NULL,
      #title = paste0("Tukey HSD: effect sizes (bold if p<", alpha, ")"),
      size=LABEL_TXT
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size=SMALL_TXT),
      panel.grid  = element_blank(),
      legend.text = element_text(size=SMALL_TXT),
      legend.title = element_text(size=LABEL_TXT),
      legend.position = "none"
    )
}


fit_orth <- aov(compressibility ~ language, data = cnn_orth_df)
fit_phon <- aov(compressibility ~ language, data = cnn_phon_df)

tuk_orth <- TukeyHSD(fit_orth)$language
tuk_phon <- TukeyHSD(fit_phon)$language

tukey_to_latex_heatcell(tuk_orth, alpha = 0.05, digits = 3,
               caption = "Treatment differences",
               label   = "tab:tukey")

tukey_to_latex_heatcell(tuk_phon, alpha = 0.05, digits = 3,
               caption = "Treatment differences",
               label   = "tab:tukey")

orth_plot = plot_tukey_full(tuk_orth)
phon_plot = plot_tukey_full(tuk_phon)
ggsave("orth_table_draft.pdf", orth_plot, width=ACL_COLUMNWIDTH, height=ACL_COLUMNWIDTH)
ggsave("phon_table_draft.pdf", phon_plot, width=ACL_COLUMNWIDTH, height=ACL_COLUMNWIDTH)
