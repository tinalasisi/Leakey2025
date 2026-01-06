library(ape)        
library(phytools)   
library(tidyverse)  
# library(corHMM)
devtools::load_all("~/corHMM/")

setwd("~/Leakey2025/")
tree   <- read.nexus("data/updated_pruned_tree.nex")
traits <- read_csv("data/primate_hair_traits_prelim_data.csv", show_col_types = FALSE)

traits <- traits %>%
  mutate(
    Genus_clean   = str_to_title(str_remove(genus,   "\\s*\\(.*$")) %>% str_replace_all("[-\\s]+", "_"),
    Species_clean = str_to_lower(str_remove(species, "\\s*\\(.*$")) %>% str_replace_all("[-\\s]+", "_"),
    Genus_species = str_c(Genus_clean, Species_clean, sep = "_") %>%
      str_replace_all("_+", "_") %>% str_remove("^_") %>% str_remove("_$")
  )

in_both        <- intersect(tree$tip.label, traits$Genus_species)
only_in_tree   <- setdiff(tree$tip.label, traits$Genus_species)
only_in_traits <- setdiff(traits$Genus_species, tree$tip.label)

tree   <- drop.tip(tree, setdiff(tree$tip.label, in_both))
traits <- filter(traits, Genus_species %in% tree$tip.label)

# classic setup
dat <- data.frame(sp = traits$Genus_species, 
  natal_coat = traits$natal_coat, 
  hair_dic = traits$hair_dichromatism_any)
table(dat[,-1])
dredge_run <- corHMMDredge(tree, dat, max.rate.cat = 1, root.p = "maddfitz", return.all = TRUE)
saveRDS(dredge_run, "output/dredge_1.RDS")
pdf("plots/dredge_trace_binary-state.pdf", width = 10)
corHMM:::plotDredgeTrace(dredge_run)
dev.off()

cor_test <- fitCorrelationTest(tree, dat, "maddfitz")
saveRDS(cor_test, "output/cor_test.RDS")
getModelTable(cor_test)

# new setup
dat2 <- data.frame(sp = traits$Genus_species, 
  ont_col = traits$ontogenetic_trajectory_color)
table(dat2[,-1])
dat2$ont_col <- as.numeric(as.factor(dat2$ont_col))
dredge_run2 <- corHMMDredge(tree, dat2, max.rate.cat = 1, root.p = "maddfitz", return.all = TRUE)
saveRDS(dredge_run2, "output/dredge_2.RDS")
pdf("plots/dredge_trace_5-state.pdf", width = 10)
corHMM:::plotDredgeTrace(dredge_run2)
dev.off()

table_1 <- getModelTable(dredge_run$all_models)
table_2 <- getModelTable(dredge_run2$all_models)

write.csv(table_1, file = "tables/model_table_binary-state.csv", row.names = FALSE)
write.csv(table_2, file = "tables/model_table_5-state.csv", row.names = FALSE)


best_1 <- dredge_run$all_models[[5]]
best_2 <- dredge_run2$all_models[[33]]


setNames(levels(as.factor(traits$ontogenetic_trajectory_color)), 1:5)
cols_1 <- RColorBrewer::brewer.pal(4, "Set1")
cols_2 <- RColorBrewer::brewer.pal(5, "Set1")

pdf("plots/marginal_recon.pdf", width = 14, height = 10)
par(mfrow=c(1,2))
plot(tree, show.tip.label = TRUE, cex = 0.3, no.margin = TRUE, label.offset = 0.5)
nodelabels(pie = best_1$states, cex = 0.4, piecol = cols_1)
tiplabels(pch = 16, 
  col = cols_1[as.numeric(best_1$data.legend[match(best_1$phy$tip.label, best_1$data.legend[,1]),2])], 
  cex = 0.75)
legend("topleft", title = "natal_coat|hair_dic", legend = colnames(best_1$solution), col = cols_1, pch = 16)
plot(tree, show.tip.label = TRUE, cex = 0.3, no.margin = TRUE, label.offset = 0.5)
nodelabels(pie = best_2$states, cex = 0.4, piecol = cols_2)
tiplabels(pch = 16, 
  col = cols_2[as.numeric(best_2$data.legend[match(best_2$phy$tip.label, best_2$data.legend[,1]),2])], 
  cex = 0.75)
legend("topleft", title = "5-state", 
  legend = levels(as.factor(traits$ontogenetic_trajectory_color)), 
  col = cols_2, pch = 16)
dev.off()

simmaps_1 <- makeSimmap(tree=best_1$phy, data=best_1$data, model=best_1$solution, rate.cat=1, nSim=100, nCores=1)
simmaps_2 <- makeSimmap(tree=best_2$phy, data=best_2$data, model=best_2$solution, rate.cat=1, nSim=100, nCores=1)

simmap_summaries_1 <- lapply(simmaps_1, summarize_single_simmap)
summary_df_1 <- summarize_transition_stats(simmap_summaries_1)
print(summary_df_1)
plot_transition_summary(simmap_summaries_1)

simmap_summaries_2 <- lapply(simmaps_2, summarize_single_simmap)
summary_df_2 <- summarize_transition_stats(simmap_summaries_2)
print(summary_df_2)
plot_transition_summary(simmap_summaries_2)

write.csv(summary_df_1, file = "tables/simmap_table_binary-state.csv", row.names = FALSE)
write.csv(summary_df_2, file = "tables/simmap_table_5-state.csv", row.names = FALSE)
pdf("plots/simmap_summary_binary-state.pdf", width = 10)
plot_transition_summary(simmap_summaries_1)
dev.off()
pdf("plots/simmap_summary_5-state.pdf", width = 10)
plot_transition_summary(simmap_summaries_2)
dev.off()



