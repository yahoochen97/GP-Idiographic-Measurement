require(tidyverse); require(glmnet); require(psych);require(splitstackshape)

## choose one age range

Data = read_csv("LOOPRDataAgeGender.csv") %>% filter(Age <= 50) %>%
  mutate(agegroup = cut(Age, breaks = c(18, 25, 30, 35, 40, 45, 50), include.lowest = T, labels = 1:6)) %>%
  stratified(., c("agegroup"), 300, list(agegroup=1:6)) %>% as_tibble
items = Data[,4:63] 

Data = read_csv("LOOPRDataAgeGender.csv") %>% filter(Age >= 18 & Age <= 25) 
items = Data[,4:63]

## reverse items where necessary (indicated in the "reverse" vector below)

reverse = c(16,31,36,51,11,26,17,47,22,37,12,42,3,48,8,23,28,58,4,49,9,24,29,44,25,55,5,50,30,45)

items[, reverse] = 6 -  items[, reverse] 

## indices for all facet and ffm domains
sama = rep(1:15, times=4)
sama.ffm = rep(1:5, times=12)

## Check alphas
for(a in 1:15) psych::alpha(items[,sama==a])$tot %>% print
for(a in 1:5) psych::alpha(items[,sama.ffm==a])$tot %>% print

## Calcuted facet and domains

facets = matrix(ncol=15, nrow=nrow(items)) %>% data.frame ## create empty data frame of 15 columns and as many rows as there are rows in "items"
domains = matrix(ncol=5, nrow=nrow(items)) %>% data.frame
for(a in 1:15) facets[,a] = rowMeans(items[,sama == a], na.rm=TRUE) ## T means TRUE
for(a in 1:5) domains[,a] = rowMeans(items[,sama.ffm == a], na.rm=T)

names(facets) =  c("Sociab","Compass","Organiz", "Anxiety","Aesth","Assert","Respect","Product","Depres","Curious","Energy","Trust","Respons","Volat","Creativ")
#names(facets) =  c("Sociab","Assert","Energy","Compass","Respect","Trust","Organiz", "Product","Respons","Anxiety","Depres","Volat","Curious","Aesth","Creativ")
names(domains) =  c("Extraversion","Agreeableness","Conscientiousness","Neuroticism","Openness")

names(items) = paste(names(facets), rep(1:4, each=15), sep=".")

itemres = names(items) %>%
  map(~select(items, -.)) %>%  
  map2(as.list(names(items)), ~ rowMeans(select(.x, contains(strtrim(.y,5))))) %>% 
  map2(as.list(names(items)), ~ tibble(.x, select(facets, -contains(strtrim(.y,5))))) %>%
  map2(as.list(items), ~ residuals(lm(.y ~ as.matrix(.x)))) %>%
  bind_cols


foo1 = function(x1,y,x2=NULL,p=.67) {
  val = !is.na(y)
  y = y[val]
  x1 = x1[val,]
  x2 = x2[val,]
  s = sample(nrow(x1), p*nrow(x1))
  if(is.null(x2)) x2 = x1
  x1 = sapply(x1,scale)
  x2 = sapply(x2,scale)
  cv.glmnet(x1[s,], y[s], alpha = .05) %>% 
    predict(x2[-s,], s = "lambda.min") %>%  
    cor(y[-s])
}

list(domains, facets, items, itemres) %>% 
  map(~replicate(500, foo1(., Data$Age))) %>%
  bind_cols %>% describe %>% select(mean, sd)

keys_names <- names(items)



keys <-
  list(Sociab = keys_names[c(1,16,31,46)],Compass=keys_names[c(1,16,31,46)+1],Organiz = keys_names[c(1,16,31,46)+2], Anxiety = keys_names[c(1,16,31,46)+3],Aesth = keys_names[c(1,16,31,46)+4],Assert = keys_names[c(1,16,31,46)+5],Respect = keys_names[c(1,16,31,46)+6],Product = keys_names[c(1,16,31,46)+7],Depres = keys_names[c(1,16,31,46)+8],Curious = keys_names[c(1,16,31,46)+9],Energy = keys_names[c(1,16,31,46)+10],Trust = keys_names[c(1,16,31,46)+11],Respons = keys_names[c(1,16,31,46)+12],Volat = keys_names[c(1,16,31,46)+13],Creativ = keys_names[c(1,16,31,46)+14])
keys_facet <- c("E:Sociab","A:Compass","C:Organiz", "N:Anxiety","O:Aesth","E:Assert","A:Respect","C:Product","N:Depres","O:Curious","E:Energy","A:Trust","C:Respons","N:Volat","O:Creativ")

par(mfrow = c(1,2), mar=c(4,4,2,0), oma=c(3,1,0,0))
manhattan(items, data.frame(Age = Data$Age), keys = keys, labels = keys_facet, abs=F, adjust="holm", ylim = c(-.1, .1), ylab = "Correlation with", axis=F, main="Items")

colnames(itemres) <- colnames(items)

par(mar = c(4,1,2,3))
manhattan(itemres, data.frame(Age = Data$Age), keys = keys, labels = keys_facet, abs=F, adjust="holm", ylim = c(-.1, .1), ylab="", yaxt='n',axis=F, main="Item Residuals")
