rm(list = ls())
library(tidyverse)


ACL_TEXTWIDTH = (455.24411/72.27)
ACL_COLUMNWIDTH = (219.08614/72.27)

PLOT_TXT = 6*.35
SMALL_TXT = 6
LABEL_TXT = 9
LINE_SZ = .35
SPACING = 1.125
KEY_HT = unit(1, "mm")
LEGEND_SPACE = unit(2, "mm") 


setwd("C:\\Users\\torre\\PycharmProjects\\opacity-measure")

cnn_orth_df <- read.csv('granular_orth_cnn_small_1layer_ker3.csv') %>% mutate(set.type = 'Orthography to Phonology') %>% mutate(compressibility = mutual.algorithmic.information/best.complexity) %>% mutate_at(vars(trial.number), factor) %>% mutate(transcription.type = ifelse(language=='kor', 'narrow', transcription.type))
cnn_phon_df <- read.csv('granular_phon_cnn_small_1layer_ker3.csv') %>% mutate(set.type = 'Phonology to Orthography') %>% mutate(compressibility = mutual.algorithmic.information/best.complexity) %>% mutate_at(vars(trial.number), factor) %>% mutate(transcription.type = ifelse(language=='kor', 'narrow', transcription.type))

filtered_orth <- read.csv('granular_orth_cnn_small_1layer_ker3_filtered.csv') %>% mutate(set.type = 'Orthography to Phonology') %>% mutate(compressibility = mutual.algorithmic.information/best.complexity) %>% mutate_at(vars(trial.number), factor) %>% mutate(transcription.type = ifelse(language=='kor', 'narrow', transcription.type))
filtered_phon <- read.csv('granular_phon_cnn_small_1layer_ker3_filtered.csv') %>% mutate(set.type = 'Phonology to Orthography') %>% mutate(compressibility = mutual.algorithmic.information/best.complexity) %>% mutate_at(vars(trial.number), factor) %>% mutate(transcription.type = ifelse(language=='kor', 'narrow', transcription.type))

sanity_orth <- read.csv('granular_orth_cnn_small_1layer_ker3_updated.csv') %>% mutate(set.type = 'Orthography to Phonology') %>% mutate(compressibility = mutual.algorithmic.information/best.complexity) %>% mutate_at(vars(trial.number), factor) %>% mutate(transcription.type = ifelse(language=='kor', 'narrow', transcription.type))
sanity_phon <- read.csv('granular_phon_cnn_small_1layer_ker3_updated.csv') %>% mutate(set.type = 'Phonology to Orthography') %>% mutate(compressibility = mutual.algorithmic.information/best.complexity) %>% mutate_at(vars(trial.number), factor) %>% mutate(transcription.type = ifelse(language=='kor', 'narrow', transcription.type))

h.onset.b = c(.24, .55, .47, .45, .15, .21, .42)
lang.b = c('dutch', 'english', 'french', 'german', 'italian', 'hungarian', 'portuguese')

h.onset = c(0.0, 0.17, 0.23, 0.42, .46, .83)
lang = c('finnish', 'hungarian', 'dutch', 'portuguese', 'french', 'english')


ziegler = data.frame(lang, h.onset)
colnames(ziegler) = c('language', 'onset.entropy')

borgwaldt = data.frame(lang.b, h.onset.b)
colnames(borgwaldt) = c('language', 'onset.entropy')





filter_japanese <- function(dataset){
  new_dataset <- dataset %>% filter(substring(language, 1, 4)!='jpn_')
  return(new_dataset)
}

filter_sanity <- function(dataset){
  new_dataset <- dataset %>% filter(language =='zzz'|language=='zzy')
  return(new_dataset)
}

only_japanese <- function(dataset){
  new_dataset <- dataset %>% filter(substring(language, 1, 3)=='jpn')
  new_dataset = new_dataset %>% mutate(script=substring(language, 5, 10)) %>%
    mutate(script = ifelse(script=='', 'all', script))
  return(new_dataset)
  
}

derivative <- function(dataset){
  df_deriv <- dataset %>%
    group_by(language, trial.number) %>%
    arrange(data.size) %>%
    mutate(
      d_compressibility = c(NA, diff(compressibility)),
      d_data_size = c(NA, diff(data.size)),
      derivative = d_compressibility / d_data_size
    ) %>%
    ungroup()
  return(df_deriv)
}

cnn_orth_jpn = derivative(only_japanese(cnn_orth_df))
cnn_orth_other = derivative(filter_japanese(cnn_orth_df))
cnn_phon_jpn = derivative(only_japanese(cnn_phon_df))
cnn_phon_other = derivative(filter_japanese(cnn_phon_df))

cnn_no_borrowings_orth = derivative(filter_japanese(filtered_orth))
cnn_no_borrowings_phon = derivative(filter_japanese(filtered_phon))

sanity_check = rbind(filter_sanity(sanity_phon), filter_sanity(sanity_orth)) %>% 
  mutate(language = ifelse(language=='zzz', 'max', 'min transparency'))

cnn_other = rbind(cnn_orth_other, cnn_phon_other)%>%
  mutate(script = if_else(script=='arab', 'Arabic', script)) %>%
  mutate(script = if_else(script=='hani', 'Chinese', script)) %>%
  mutate(script = if_else(script=='cyrl', 'Cyrillic', script)) %>%
  mutate(script = if_else(script=='deva', 'Devanagari', script)) %>%
  mutate(script = if_else(script=='hang', 'Hangul', script)) %>%
  mutate(script = if_else(script=='grek', 'Greek', script)) %>%
  mutate(script = if_else(script=='latn', 'Latin', script)) %>%
  mutate(script = if_else(script=='jpn', 'Japanese', script))
cnn_jpn = rbind(cnn_orth_jpn, cnn_phon_jpn)%>%
  mutate(script = if_else(script=='arab', 'Arabic', script)) %>%
  mutate(script = if_else(script=='hani', 'Chinese', script)) %>%
  mutate(script = if_else(script=='cyrl', 'Cyrillic', script)) %>%
  mutate(script = if_else(script=='deva', 'Devanagari', script)) %>%
  mutate(script = if_else(script=='hang', 'Hangul', script)) %>%
  mutate(script = if_else(script=='grek', 'Greek', script)) %>%
  mutate(script = if_else(script=='latn', 'Latin', script)) %>%
  mutate(script = if_else(script=='jpn', 'Japanese', script))
cnn_no_borrowings = rbind(cnn_no_borrowings_phon, cnn_no_borrowings_orth)%>%
  mutate(script = if_else(script=='arab', 'Arabic', script)) %>%
  mutate(script = if_else(script=='hani', 'Chinese', script)) %>%
  mutate(script = if_else(script=='cyrl', 'Cyrillic', script)) %>%
  mutate(script = if_else(script=='deva', 'Devanagari', script)) %>%
  mutate(script = if_else(script=='hang', 'Hangul', script)) %>%
  mutate(script = if_else(script=='grek', 'Greek', script)) %>%
  mutate(script = if_else(script=='latn', 'Latin', script)) %>%
  mutate(script = if_else(script=='jpn', 'Japanese', script))

#ggplot(cnn_orth_other, aes(x=data.size, y=compressibility)) + geom_smooth() + geom_line(aes(color=trial.number, alpha=.1)) + facet_wrap(~language) + theme_bw()
#ggplot(cnn_phon_other, aes(x=data.size, y=compressibility)) + geom_smooth() + geom_line(aes(color=trial.number, alpha=.1)) + facet_wrap(~language) + theme_bw()
#ggplot(cnn_orth_other, aes(x=data.size, y=derivative)) + geom_smooth() + facet_wrap(~language) + theme_bw()
#ggplot(cnn_phon_other, aes(x=data.size, y=derivative)) + geom_smooth() + facet_wrap(~language) + theme_bw()

means <- cnn_other %>% filter(data.size==max(data.size)) %>%
  group_by(language) %>%
  summarise(compressibility = mean(compressibility, na.rm = TRUE))

plot_means <- cnn_other %>% filter(data.size==max(data.size)) %>% filter(set.type=="Orthography to Phonology") %>%
  group_by(language) %>%
  summarise(compressibility = mean(compressibility, na.rm = TRUE))

ordered_languages <- plot_means %>%
  arrange(compressibility) %>%
  pull(language)

cnn_other$language <- factor(cnn_other$language, levels = ordered_languages)
#cnn_no_borrowings$language <- factor(cnn_other$language, levels = ordered_languages)

#ggplot(cnn_other%>%filter(data.size==max(data.size)), aes(x=language, y=compressibility)) + geom_smooth() + geom_point(aes(color=script, alpha=.1)) + facet_wrap(~set.type) + theme_bw()
#ggplot(cnn_phon_other%>%filter(data.size==15808), aes(x=language, y=compressibility)) + geom_smooth() + geom_point(aes(color=trial.number, alpha=.1)) + facet_wrap(~set.type) + theme_bw()


#ggplot(cnn_orth_jpn, aes(x=data.size, y=compressibility)) + geom_smooth() + geom_line(aes(color=trial.number, alpha=.1)) + facet_wrap(~language) + theme_bw()
#ggplot(cnn_phon_jpn, aes(x=data.size, y=compressibility)) + geom_smooth() + geom_line(aes(color=trial.number, alpha=.1)) + facet_wrap(~language) + theme_bw()

# Plotting experiment
cnn_other_final = cnn_other%>%filter(data.size==max(data.size))

summary_other <- cnn_other_final %>%
  group_by(language, set.type, script, transcription.type) %>%
  summarize(
    mean_compressibility = mean(compressibility, na.rm = TRUE),
    sd = sd(compressibility, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    se = sd / sqrt(n),
    ci95 = 1.96 * se,  # 95% confidence interval for normal distribution
    ymin = mean_compressibility - ci95,
    ymax = mean_compressibility + ci95
  ) %>% mutate(language = fct_recode(language, arabic = 'ara', chinese = 'zho', japanese = 'jpn', german = 'deu',
                                     english = 'eng', russian = 'rus', korean = 'kor', hindi = 'hin', ukrainian = 'ukr',
                                     vietnamese = 'vie', bulgarian = 'bul', greek = 'ell', dutch = 'nld', french = 'fra',
                                     finnish = 'fin', turkish = 'tur', hungarian = 'hun', portuguese = 'por', polish = 'pol',
                                     czech = 'ces', italian = 'ita', spanish = 'spa'))

summary_no_borrowings <- cnn_no_borrowings %>% filter(data.size==max(data.size)) %>%
  group_by(language, set.type, script, transcription.type) %>%
  summarize(
    mean_compressibility = mean(compressibility, na.rm = TRUE),
    sd = sd(compressibility, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    se = sd / sqrt(n),
    ci95 = 1.96 * se,  # 95% confidence interval for normal distribution
    ymin = mean_compressibility - ci95,
    ymax = mean_compressibility + ci95
  )

cnn_jpn_final = cnn_jpn %>% filter(data.size==max(data.size)) %>%
  mutate(script = if_else(script=='kataka', 'Katakana', script)) %>% 
  mutate(script = if_else(script=='all', 'All', script)) %>%
  mutate(script = if_else(script=='kanji', 'Kanji', script))

summary_jpn <- cnn_jpn_final %>%
  group_by(language, set.type, script, transcription.type) %>%
  summarize(
    mean_compressibility = mean(compressibility, na.rm = TRUE),
    sd = sd(compressibility, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    se = sd / sqrt(n),
    ci95 = 1.96 * se,  # 95% confidence interval for normal distribution
    ymin = mean_compressibility - ci95,
    ymax = mean_compressibility + ci95
  )

# COMBINED

main_plot = ggplot(data = summary_other,
       aes(y = mean_compressibility, ymin = ymin, ymax = ymax, x = set.type, color = script, linetype=transcription.type, group=language)) +
  geom_errorbar(width = SPACING, position = position_dodge2(padding = .5), size=LINE_SZ) +
  geom_text(aes(label = language),
            position = position_dodge2(width = SPACING, padding = .15), 
            hjust = .5,  # horizontal adjustment    # vertical text
            vjust = 1.25, # vertical adjustment
            size=PLOT_TXT) + geom_hline(yintercept=c(0), size=LINE_SZ, linetype='dotted', color='black') + 
  geom_text(aes(y=0.012, label="Uncompressed"), color='black', angle=270, size=PLOT_TXT+.2) +  
  facet_grid(set.type~., scales = "free_y") +
  theme_bw() + theme(
    legend.position = c(.4, .25),
    legend.text = element_text(size=SMALL_TXT),
    legend.title = element_text(size=LABEL_TXT),
    legend.key.height = KEY_HT,
    legend.spacing.y = LEGEND_SPACE,
    axis.title.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line( linewidth=.05, color="gray" ),
    panel.grid.minor.x = element_line( linewidth=.05, color="gray" ),
    strip.text = element_text(size=LABEL_TXT),
    axis.text.x = element_text(size=LABEL_TXT)
  ) + labs(
    y = "Compressibility", 
    color = "Script", 
    linetype = "Transcription"
  ) + coord_flip()

main_plot


summary_for_borrows <- rbind(summary_other %>% mutate(borrowings=TRUE) %>% filter(language %in% summary_no_borrowings$language), 
                                  summary_no_borrowings %>% mutate(borrowings=FALSE))

borrow_plot = ggplot(data = summary_for_borrows %>% filter(script=='Latin'),
                   aes(y = mean_compressibility, ymin = ymin, ymax = ymax, x = set.type, color = script, linetype=transcription.type, group=language)) +
  geom_errorbar(width = SPACING, position = position_dodge2(padding = .5), size=LINE_SZ) +
  geom_text(aes(label = language),
            position = position_dodge2(width = SPACING, padding = .15), 
            hjust = .5,  # horizontal adjustment    # vertical text
            vjust = -1.25, # vertical adjustment
            size=PLOT_TXT) + geom_hline(yintercept=c(0), size=LINE_SZ, linetype='dotted', color='black') + 
  geom_text(aes(y=0.012, label="Uncompressed"), color='black', angle=270, size=PLOT_TXT+.2) +  
  facet_grid(set.type~borrowings, scales = "free_y") +
  theme_bw() + theme(
    legend.position = c(.4, .25),
    legend.text = element_text(size=SMALL_TXT),
    legend.title = element_text(size=LABEL_TXT),
    legend.key.height = KEY_HT,
    legend.spacing.y = LEGEND_SPACE,
    axis.title.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line( linewidth=.05, color="gray" ),
    panel.grid.minor.x = element_line( linewidth=.05, color="gray" ),
    strip.text = element_text(size=LABEL_TXT),
    axis.text.x = element_text(size=LABEL_TXT)
  ) + labs(
    y = "Compressibility", 
    color = "Script", 
    linetype = "Transcription"
  ) + coord_flip()

borrow_plot

arabic_plot = ggplot(data = summary_other %>% filter(language=='ara'),
       aes(y = mean_compressibility, ymin = ymin, ymax = ymax, x = set.type, linetype=transcription.type)) +
  geom_jitter(data=cnn_other_final%>% filter(language=='ara'), 
              aes(y=compressibility, ymin=0, ymax=0, color=script),
              width=.05, alpha=.5) +
  geom_errorbar(width = .1, linewidth=1, size=LINE_SZ) +
  geom_text(aes(label = set.type),
            position = position_dodge2(width = 2.0, padding = .15), 
            hjust = .5,  # horizontal adjustment    # vertical text
            vjust = 4.95, # vertical adjustment
            size=PLOT_TXT) +
  geom_hline(yintercept=0.0, size=LINE_SZ, linetype='dotted') +
  theme_bw() + theme(
    legend.position = "None",
    axis.title.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank(),
    strip.text = element_text(size=LABEL_TXT),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line( linewidth=.05, color="gray" ),
    panel.grid.minor.y = element_line( linewidth=.05, color="gray" ) 
  ) + labs(
    y = "Compressibility", 
    color = "Script", 
    linetype = "Transcription",
    size = LABEL_TXT
  )


jpn_plot = ggplot(data = summary_jpn %>% mutate(set.type=fct_recode(set.type, `Orthography to P.` = 'Orthography to Phonology', `Phonology to O.` = 'Phonology to Orthography')),
       aes(y = mean_compressibility, ymin = ymin, ymax = ymax, x = language)) +
  geom_jitter(data=cnn_jpn_final%>% mutate(set.type=fct_recode(set.type, `Orthography to P.` = 'Orthography to Phonology', `Phonology to O.` = 'Phonology to Orthography')), 
              aes(y=compressibility, ymin=0, ymax=0, color=script),
              width=.05, alpha=.5) +
  geom_errorbar(width = 2.97, linewidth=.5, position = position_dodge2(padding = .9), size=LINE_SZ) +
  geom_text(aes(label = script, color=script),
            position = position_dodge2(width = 2.97, padding = 0), 
            hjust = .5,  # horizontal adjustment    # vertical text
            vjust = 2.5, # vertical adjustment
            size=PLOT_TXT) +  
  facet_grid(set.type~., scales = "free_y") +
  geom_hline(yintercept=0.0, size=LINE_SZ, linetype='dotted')+
  theme_bw() + theme(
    legend.position = "None",
    axis.title.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    axis.title.x = element_text(size=LABEL_TXT),
    strip.text = element_text(size=LABEL_TXT),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line( linewidth=.05, color="gray" ),
    panel.grid.minor.x = element_line( linewidth=.05, color="gray" ) 
  ) + labs(
    y = "Compressibility", 
    color = "Script", 
    linetype = "Transcription",
    size = LABEL_TXT
  ) + coord_flip()

jpn_plot

summary_sanity <- sanity_check %>% filter(data.size==max(data.size)) %>%
  group_by(language, set.type, script, transcription.type) %>%
  summarize(
    mean_compressibility = mean(compressibility, na.rm = TRUE),
    sd = sd(compressibility, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    se = sd / sqrt(n),
    ci95 = 1.96 * se,  # 95% confidence interval for normal distribution
    ymin = mean_compressibility - ci95,
    ymax = mean_compressibility + ci95
  )

sanity_plot = ggplot(data = summary_sanity,
                  aes(y = mean_compressibility, ymin = ymin, ymax = ymax, x = language)) +
  geom_jitter(data=sanity_check%>% filter(data.size==max(data.size)), 
              aes(y=compressibility, ymin=0, ymax=0, color=script),
              width=.05, alpha=.5) +
  geom_errorbar(width = .1, linewidth=.5, size=LINE_SZ) +
  geom_text(aes(label = language), color="black",
            hjust = .5
            ,  # horizontal adjustment    # vertical text
            vjust = 2.5, # vertical adjustment
            size=PLOT_TXT) +  
  facet_grid(set.type~., scales = "free_y") +
  geom_hline(yintercept=0.0, size=LINE_SZ, linetype='dotted')+
  theme_bw() + theme(
    legend.position = "None",
    axis.title.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    strip.text = element_text(size=LABEL_TXT*.75),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line( linewidth=.05, color="gray" ),
    panel.grid.minor.x = element_line( linewidth=.05, color="gray" ) 
  ) + labs(
    y = "Compressibility", 
    color = "Script", 
    linetype = "Transcription",
    size = LABEL_TXT
  ) + coord_flip()

sanity_plot

script_summary = rbind(cnn_jpn_final %>% filter(language!='jpn'), cnn_other_final) %>% group_by(set.type, script) %>%
  summarize(
    mean_compressibility = mean(compressibility, na.rm = TRUE),
    sd = sd(compressibility, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    se = sd / sqrt(n),
    ci95 = 1.96 * se,  # 95% confidence interval for normal distribution
    ymin = mean_compressibility - ci95,
    ymax = mean_compressibility + ci95
  ) 
  

ordered_scripts <- script_summary %>% filter(set.type=="Orthography to Phonology") %>%
  arrange(mean_compressibility) %>%
  pull(script)


script_summary$script <- factor(script_summary$script, levels = unique(ordered_scripts))

script_plot = ggplot(data = script_summary,
       aes(y = mean_compressibility, ymin = ymin, ymax = ymax, x = set.type, color = script, group=script)) +
  geom_errorbar(width = SPACING, position = position_dodge2(padding = .5), size=LINE_SZ) +
  geom_text(aes(label = script),
            position = position_dodge2(width = SPACING, padding = .15), 
            hjust = .5,  # horizontal adjustment    # vertical text
            vjust = 1.65, # vertical adjustment
            size=PLOT_TXT) + geom_hline(yintercept=c(0), size=LINE_SZ, linetype='dotted', color='black') + 
  geom_text(aes(y=0.012, label="Uncompressed"), color='black', angle=270, size=PLOT_TXT+.2) +
  facet_grid(set.type~., scales = "free_y") +
  theme_bw() + theme(
    legend.position = "none",
    legend.text = element_text(size=SMALL_TXT),
    legend.title = element_text(size=LABEL_TXT),
    legend.key.height = KEY_HT,
    legend.spacing.y = LEGEND_SPACE,
    axis.title.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line( linewidth=.05, color="gray" ),
    panel.grid.minor.x = element_line( linewidth=.05, color="gray" ),
    strip.text = element_text(size=LABEL_TXT),
    axis.text.x = element_text(size=LABEL_TXT)
  ) + labs(
    y = "Compressibility", 
    color = "Script", 
    linetype = "Transcription"
  ) + coord_flip()

main_plot
script_plot
jpn_plot
arabic_plot
borrow_plot
ggsave("main_draft_full_squeezed.pdf", main_plot, width=ACL_TEXTWIDTH, height=ACL_COLUMNWIDTH*1.8)
ggsave("script_draft.pdf", script_plot, width=ACL_TEXTWIDTH, height=ACL_COLUMNWIDTH*1.5)
ggsave("jpn_draft_side_squeezed.pdf", jpn_plot, width=ACL_COLUMNWIDTH, height=ACL_COLUMNWIDTH)
ggsave("arabic_draft.pdf", arabic_plot, width=ACL_COLUMNWIDTH, height=ACL_COLUMNWIDTH)
ggsave("sanity_plot_side.pdf", sanity_plot, width=ACL_COLUMNWIDTH, height=ACL_COLUMNWIDTH)
ggsave("summary_no_borrowings_draft.pdf", borrow_plot, width=ACL_TEXTWIDTH, height=ACL_COLUMNWIDTH*1.75)

orth = summary_other %>% filter(set.type=="Orthography to Phonology")
comparison = inner_join(orth, borgwaldt, by='language')
#ggplot(comparison, aes(x=onset.entropy, y=mean_compressibility)) + geom_point()
model = lm(mean_compressibility~onset.entropy, data=comparison)
coefs = coef(model)

empirical_plot = ggplot(comparison, aes(x = onset.entropy, y = mean_compressibility)) +
  geom_point(aes(color=language)) +
  geom_abline(
    intercept = coefs[1],
    slope     = coefs[2]
  ) +
  # geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color='black') +
  labs(
    x     = "Onset Entropy",
    y     = "Mean Compressibility",
  ) +
  geom_text(aes(label=language, color=language), size = PLOT_TXT, vjust=-1.0) +
  theme_minimal() + 
  coord_cartesian(xlim=c(-0.1,1.0), ylim=c(0.45, 0.7)) +
  theme(
    legend.position='none',
    axis.title = element_text(size=LABEL_TXT)
  )

empirical_plot

ggsave("borgwaldt_empirical_half.pdf", empirical_plot, width=ACL_COLUMNWIDTH, height=ACL_COLUMNWIDTH)

