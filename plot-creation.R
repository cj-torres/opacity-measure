rm(list = ls())
library(tidyverse)

setwd("F:\\PycharmProjects\\opacity-measure")

type_token_df = read.csv("type_token_df.csv") %>% mutate(language=substring(language, 1, 3)) %>% 
  mutate(type.token.ratio=types/tokens)

# Read the CSV file
orth_data <- read.csv("preq_data_no_index_orth.csv") %>% mutate(language=substring(language, 1, 3)) %>% inner_join(type_token_df, by=c('language'))
phon_data <- read.csv("preq_data_no_index_phon.csv") %>% mutate(language=substring(language, 1, 3)) %>% inner_join(type_token_df, by=c('language'))


orth_data <- read.csv("wikipron_massively_multilingual_orth.csv")  %>% mutate(language=substring(language, 1, 3)) %>% filter(!(language %in% c('yue', 'ang', 'lat', 'grc')))

orth_data <- read.csv('lemma_orth_cnn_ker3.csv')
phon_data <- read.csv('lemma_phon_cnn.csv')

schmalz <- read.csv("schmalz_data.csv", encoding="UTF-8") %>% rename(language = X.U.FEFF.language) 

# Run LM test
truth_data = inner_join(orth_data, schmalz, by='language')
truth_lm = lm(mutual.algorithmic.information ~ context*irregular, data=truth_data)
summary(truth_lm)




# Calculate the mean mutual.algorithmic.information for each language
orth_means <- orth_data %>%
  group_by(language) %>%
  summarise(mutual.algorithmic.information = mean(mutual.algorithmic.information, na.rm = TRUE))


# Order the languages by the mean mutual.algorithmic.information
ordered_languages <- orth_means %>%
  arrange(mutual.algorithmic.information) %>%
  pull(language)

# Convert the language column to a factor with the levels ordered by the mean values
orth_data$language <- factor(orth_data$language, levels = ordered_languages)

plot_means <- orth_data %>%
  group_by(language, transcription.type) %>%
  summarise(mutual.algorithmic.information = mean(mutual.algorithmic.information, na.rm = TRUE))

plot_ci <- orth_data %>%
  group_by(language, transcription.type, script) %>%
  summarise(
    mean = mean(mutual.algorithmic.information),
    ymin = mean(mutual.algorithmic.information) - 1.96 * (sd(mutual.algorithmic.information) / sqrt(n())),
    ymax = mean(mutual.algorithmic.information) + 1.96 * (sd(mutual.algorithmic.information) / sqrt(n()))
  )


ggplot(data = orth_data, aes(x = language, y = mutual.algorithmic.information, color=script)) + 
  geom_violin(color='black') + geom_jitter(width=.1, height=0) +
  theme_bw() + geom_point(data=plot_means, aes(y=mutual.algorithmic.information), color='black', size=3)  + facet_grid(transcription.type~.) +
  stat_summary(geom = "errorbar", fun.data = mean_se, position = "dodge", color="black")



mix_lm = lm(mutual.algorithmic.information ~ types*tokens, data=orth_data)
# Calculate the mean mutual.algorithmic.information for each language
adjusted_data <- orth_data  %>% 
  mutate(mutual.algorithmic.information = 
           mutual.algorithmic.information-residuals(mix_lm)-residuals(typ_lm))

adjusted_means <- adjusted_data %>%
  group_by(language) %>%
  summarise(mutual.algorithmic.information = mean(mutual.algorithmic.information, na.rm = TRUE))

# Order the languages by the mean mutual.algorithmic.information
ordered_languages <- adjusted_means %>%
  arrange(mutual.algorithmic.information) %>%
  pull(language)


# Convert the language column to a factor with the levels ordered by the mean values
adjusted_data$language <- factor(orth_data$language, levels = ordered_languages)

ggplot(data = adjusted_data, aes(x = language, y = mutual.algorithmic.information)) + 
  geom_violin() + geom_jitter(width=.1, height=0) +
  theme_bw() + geom_point(data=adjusted_means, aes(y=mutual.algorithmic.information), color='blue', size=3)







phon_means <- phon_data %>%
  group_by(language) %>%
  summarise(mutual.algorithmic.information = mean(mutual.algorithmic.information, na.rm = TRUE))


# Order the languages by the mean mutual.algorithmic.information
ordered_languages <- phon_means %>%
  arrange(mutual.algorithmic.information) %>%
  pull(language)

# Convert the language column to a factor with the levels ordered by the mean values
phon_data$language <- factor(phon_data$language, levels = ordered_languages)

ggplot(data = phon_data, aes(x = language, y = mutual.algorithmic.information)) + 
  geom_violin() + geom_jitter(width=.1, height=0) +
  theme_bw() + geom_point(data=phon_means, aes(y=mutual.algorithmic.information), color='blue', size=3)

phon_data = phon_data %>% mutate(dataset='Phonology to Orthography')
orth_data = orth_data %>% mutate(dataset='Orthography to Phonology')
big_set = rbind(phon_data, orth_data)

big_means <- big_set %>%
  group_by(language) %>%
  summarise(mutual.algorithmic.information = mean(mutual.algorithmic.information, na.rm = TRUE))


# Order the languages by the mean mutual.algorithmic.information
ordered_languages <- big_means %>%
  arrange(mutual.algorithmic.information) %>%
  pull(language)

dataset_means <- big_set %>%
  group_by(language, dataset) %>%
  summarise(mutual.algorithmic.information = mean(mutual.algorithmic.information, na.rm = TRUE))


big_set$language <- factor(big_set$language, levels = ordered_languages)

ggplot(data = big_set, aes(x = language, y = mutual.algorithmic.information)) + 
  geom_violin() + geom_jitter(width=.1, height=0) +
  theme_bw() + geom_point(data=dataset_means, aes(y=mutual.algorithmic.information), color='blue', size=3) +
  facet_grid(.~dataset) + ylab('Mutual Algorithmic Information') + xlab('Language')


ggplot(data = big_set %>% filter(language=='cmn'), aes(x = language, y = mutual.algorithmic.information)) + 
  geom_violin() + geom_jitter(width=.1, height=0) +
  theme_bw() + geom_point(data=dataset_means%>% filter(language=='cmn'), aes(y=mutual.algorithmic.information), color='blue', size=3) +
  facet_grid(.~dataset) + ylab('Mutual Algorithmic Information') + xlab('Language')


cmn_data = big_set %>% filter(language=='cmn')
cmn_tt <- t.test(mutual.algorithmic.information ~ dataset, data = cmn_data)
summary(cmn_tt)


ggplot(data = big_set %>% filter(language=='eng'), aes(x = language, y = mutual.algorithmic.information)) + 
  geom_violin() + geom_jitter(width=.1, height=0) +
  theme_bw() + geom_point(data=dataset_means%>% filter(language=='eng'), aes(y=mutual.algorithmic.information), color='blue', size=3) +
  facet_grid(.~dataset) + ylab('Mutual Algorithmic Information') + xlab('Language')


eng_data = big_set %>% filter(language=='eng')
eng_tt <- t.test(mutual.algorithmic.information ~ dataset, data = eng_data)


#data_2 = data %>% mutate(mi.2 = best.conditional.complexity - )

# Perform ANOVA
orth_anova_result <- aov(mutual.algorithmic.information ~ language, data = orth_data)
summary(orth_anova_result)


# If ANOVA is significant, perform post-hoc tests
if (summary(orth_anova_result)[[1]]["Pr(>F)"][[1]][1] < 0.05) {
  posthoc_result <- TukeyHSD(orth_anova_result)
  print(posthoc_result)
}