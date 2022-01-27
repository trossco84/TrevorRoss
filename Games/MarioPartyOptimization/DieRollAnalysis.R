# Importing Libraries
library(tidyr)
library(dplyr)
library(ggplot2)

# Reading Data
character_dice = read.csv("~/Desktop/My Projects/TrevorRoss/Games/MarioPartyOptimization/CharacterDice.csv")
head(character_dice)

# Cleaning Data
## Assuming that we are interested in maximizing the distance traveled
## Replace all "Coin" related rolls with 0
cd2 = character_dice

cd3 = data.frame(lapply(cd2, function(x) {x[grepl("coin", tolower(x), fixed = TRUE)] <- 0; x}))
cd3$P1 = as.numeric(cd3$P1)
cd3$P2 = as.numeric(cd3$P2)

# Expected Value of each roll
exp_value_dice = function(lx){
  evd = 0
  for (val in lx){
    evd = evd + ((1/6)*val)
  }
  return (evd)
}

cd4 = cd3 %>% 
  mutate("Expected_Roll" = exp_value_dice(list(P1,P2,P3,P4,P5,P6)))

cd4[order(-cd4$Expected_Roll),]

# Variance of each roll
var_dice = function(lx){
  vd = 0
  evd = exp_value_dice(lx)
  for (val in lx){
    vd = vd + ((val - evd)*(val-evd)*(1/6))
  }
  return (vd)
}

cd5 = cd4 %>% 
  mutate("Roll_Variance" = var_dice(list(P1,P2,P3,P4,P5,P6))) %>% 
  mutate("Roll_Std" = Roll_Variance^.5)

cd5[order(-cd5$Expected_Roll,cd5$Roll_Std),]

