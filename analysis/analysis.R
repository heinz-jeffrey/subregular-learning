library(PMCMRplus) # provides frdAllPairsNemenyiTest()
library(reshape2)  # provides acast()
library(dplyr)     # provides recode_factor
library(ggplot2)   # provides ggplot
library(effsize)   # provides cohen.d

# DATA PREPARATION
cols = c("alph", "tier", "class", "k", "j", "i", "network_type",
         "train_set_size", "test_type", "accuracy", "fscore", "auc", "brier")
eval = read.csv('all_evals.csv', header=TRUE)[cols]

eval$alph = as.factor(eval$alph)
eval$tier = as.factor(eval$tier)
eval$class = as.factor(eval$class)
eval$network_type = as.factor(eval$network_type)
eval$train_set_size = as.factor(eval$train_set_size)
eval$test_type = as.factor(eval$test_type)

cnts = read.table('counts.tsv', header=TRUE, sep='\t')

# Reporting basic stats on sizes of automata representations of the languages
summary(cnts$Size)
summary(cnts$Monoid)

print(sd(cnts$Size))
print(sd(cnts$Monoid))

data = merge(
     eval,
     cnts,
     by.x=c("alph", "tier", "class", "k", "j", "i"),
     by.y=c("Alph", "Tier", "Class", "k", "j", "i")
)

data$network_type = recode_factor(data$network_type,
                                  simple="RNN",
                                  gru="GRU",
                                  lstm="LSTM",
                                  transformer="Transformer")


# checking correlations among accuracy, AUC, Brier, and f-score measures

cor(data[, c('accuracy', 'auc', 'brier', 'fscore')])



#SET UP CAST MATRIX WITH LANG CLASSES AS COLUMNS
df = aggregate(data$accuracy,
               by=list(alph=data$alph,
                       network_type=data$network_type,
                       train_set_size=data$train_set_size,
                       test_type=data$test_type,
                       class=data$class),
               FUN=mean)
cast.matrix = acast(df,
                    alph + network_type + train_set_size + test_type ~ class,
                    value.var="x")
cast.df = as.data.frame(cast.matrix)


# SANITY CHECK:
# ========================================================================
# DOES NN ACCURACY INCREASE WITH TRAIN SET SIZE?
data.matrix = acast(df,
                    alph + class + network_type + test_type ~ train_set_size,
                    value.var="x")
friedman.test(data.matrix)

# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
colMeans(data.matrix)
# ========================================================================



# FRIEDMAN TEST FOR LANGUAGE CLASSES
# ========================================================================
data.matrix = acast(df,
                    alph + test_type + network_type + train_set_size ~ class,
                    value.var="x")
friedman.test(cast.matrix)

frdAllPairsNemenyiTest(cast.matrix)
colMeans(cast.matrix)

# SANITY CHECK : ARE SL/coSL, SP/coSP, TSL/TcoSL all-pairs p-values close to 1?


# ========================================================================
# FRIEDMAN TEST FOR TEST TYPES
# ========================================================================
# DOES NN ACCURACY DECREASE ACROSS THE TEST TYPES SR < (SA <> LR) < LA?
data.matrix = acast(df,
                    alph + class + network_type + train_set_size ~ test_type,
                    value.var="x")
friedman.test(data.matrix)

# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
colMeans(data.matrix)

# is TEST TYPE DIFFICULTY AS FOLLOWS: SR < LR < SA < LA ?
# ========================================================================

networks = c('RNN', 'GRU', 'LSTM', 'Transformer')
for (ntw in networks) {
  print(ntw)
  data.matrix = acast(df[df$network_type == ntw,],
                      alph + class + network_type + train_set_size ~ test_type,
                      value.var="x")
  print(friedman.test(data.matrix))
  
  # POST HOC MULTIPLE COMPARISONS ANALYSIS
  print(frdAllPairsNemenyiTest(data.matrix))
  print(sort(colMeans(data.matrix)))
  
}




# CREATE AGGREGATE MATRIX FOR NNs VS TEST TYPE
trainsizes <- c('Small', 'Mid', 'Large')

for (size in trainsizes) {

  df.tmp = data[data$train_set_size == size,]
  agg.tmp = aggregate(df.tmp$accuracy,
                    by=list( network_type=df.tmp$network_type
                           , test=df.tmp$test_type)
                           , FUN=mean)
  agg = acast(agg.tmp, network_type ~ test, value.var="x")
  
  print(size)
  print(agg[,c(4,2,3,1)])

}



# ========================================================================
# DIFFERENCES BETWEEN CNL, DPL, PROP, FO, REG
# ========================================================================

cnl = c("SL", "SP", "TSL")
dpl = c("coSL", "coSP", "TcoSL")
prop = c("LT", "PLT", "PT", "TLT", "TPLT")
fo = c("LTT", "TLTT", "SF")
reg = c("Zp", "Reg")
data$logic <-
     ifelse(data$class %in% cnl, "CNL",
     ifelse(data$class %in% dpl, "DPL",
     ifelse(data$class %in% prop, "PROP",
     ifelse(data$class %in% fo, "FO",
     ifelse(data$class %in% reg, "REG", "OTHER")))))


logic.agg = aggregate(data$accuracy,
                      by=list(alph=data$alph,
                              network_type=data$network_type,
                              train_set_size=data$train_set_size,
                              test_type=data$test_type,
                              logic=data$logic),
                      FUN=mean)
data.matrix = acast(logic.agg,
                    alph + network_type + train_set_size + test_type ~ logic,
                    value.var="x")

friedman.test(data.matrix)


# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
sort(colMeans(data.matrix))


# REG IS HARDEST TO LEARN. OTHERS ARE NOT SO CLEAR.
# ========================================================================
for (ntw in networks) {
    data.matrix = acast(logic.agg[logic.agg$network_type == ntw,],
                        alph + network_type + train_set_size + test_type ~ logic,
                        value.var="x")
    print(ntw)
    print(sort(colMeans(data.matrix)))
    print(friedman.test(data.matrix))
    # POST HOC MULTIPLE COMPARISONS ANALYSIS
    print(frdAllPairsNemenyiTest(data.matrix))
  }






# ========================================================================
# DIFFERENCES BETWEEN PROPOSITIONAL LOGICS BY ORDER RELATION
# ========================================================================
succ = c("SL", "coSL", "LT", "LTT")
prec = c("coSP", "PT", "SF", "SP")
tsucc = c("TcoSL", "TLT", "TLTT", "TSL")
data$prop <-
  ifelse(data$class %in% succ, "SUCC",
         ifelse(data$class %in% prec, "PREC",
                ifelse(data$class %in% tsucc, "TSUCC", "OTHER")))

prop.agg = aggregate(data$accuracy,
                     by=list(alph=data$alph,
                             network_type=data$network_type,
                             train_set_size=data$train_set_size,
                             test_type=data$test_type,
                             prop=data$prop),
                     FUN=mean)
data.matrix = acast(prop.agg,
                    alph + network_type + train_set_size + test_type ~ prop,
                    value.var="x")

friedman.test(data.matrix)


# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
sort(colMeans(data.matrix))

# ========================================================================

for (ntw in networks) {
    data.matrix = acast(logic.agg[logic.agg$network_type == ntw,],
                        alph + network_type + train_set_size + test_type ~ prop,
                        value.var="x")
    print(ntw)
    print(sort(colMeans(data.matrix)))
    print(friedman.test(data.matrix))
    # POST HOC MULTIPLE COMPARISONS ANALYSIS
    print(frdAllPairsNemenyiTest(data.matrix))
  }




# ========================================================================
# FRIEDMAN TEST FOR ALPHABET SIZES:
# ========================================================================
data.matrix = acast(df,
                    class + network_type + train_set_size + test_type ~ alph,
                    value.var="x")
friedman.test(data.matrix)


# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
colMeans(data.matrix)

for (ntw in networks) {
  data.matrix = acast(df[df$network_type == ntw,],
                      class + network_type + train_set_size + test_type ~ alph,
                      value.var="x")
  print(ntw)
  print(sort(colMeans(data.matrix)))
  print(friedman.test(data.matrix))
  # POST HOC MULTIPLE COMPARISONS ANALYSIS
  print(frdAllPairsNemenyiTest(data.matrix))
}


# ========================================================================


# ========================================================================
# FRIEDMAN TEST FOR NETWORK TYPES
# ========================================================================
# ALL Training Sets
data.matrix = acast(df,
                    alph + class + train_set_size + test_type ~ network_type,
                    value.var="x")
friedman.test(data.matrix)


# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
sort(colMeans(data.matrix))

# ========================================================================

# Small Training Set
# ==================
df.temp = df[df$train_set_size == "Small",]
data.matrix = acast(df.temp,
                    alph + class + train_set_size + test_type ~ network_type,
                    value.var="x")
friedman.test(data.matrix)


# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
colMeans(data.matrix)

# ========================================================================



# Mid Training Set
df.temp = df[df$train_set_size == "Mid",]
data.matrix = acast(df.temp,
                    alph + class + train_set_size + test_type ~ network_type,
                    value.var="x")
friedman.test(data.matrix)


# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
colMeans(data.matrix)

# ========================================================================


# Large Training Set
df.temp = df[df$train_set_size == "Large",]
data.matrix = acast(df.temp,
                    alph + class + train_set_size + test_type ~ network_type,
                    value.var="x")
friedman.test(data.matrix)

# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.matrix)
colMeans(data.matrix)

# ========================================================================



# DO k VALUES MAKE A DIFFERENCE FOR PROP1 AND PROP2?
# ========================================================================
succ = c("SL", "coSL", "LT", "LTT")
prec = c("coSP", "PT", "SP") # removed SF
tsucc = c("TcoSL", "TLT", "TLTT", "TSL")

prop1 = c(succ, prec, tsucc)
prop2 = prop1[!(prop1 %in% c("LTT", "TLTT"))]

df.temp = aggregate(data$accuracy,
                    by=list(alph=data$alph,
                            class=data$class,
                            k=data$k,
                            network_type=data$network_type,
                            train_set_size=data$train_set_size,
                            test_type=data$test_type),
                    FUN=mean)

df.temp1 = df.temp[df.temp$class %in% prop1,]
data.prop1 = acast(df.temp1,
                   alph + class + network_type + train_set_size + test_type ~ k,
                   value.var="x")
df.temp2 = df.temp[df.temp$class %in% prop2,]
data.prop2 = acast(df.temp2,
                   alph + class + network_type + train_set_size + test_type ~ k,
                   value.var="x")

friedman.test(data.prop1)

# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.prop1)
colMeans(data.prop1)


friedman.test(data.prop2)

# POST HOC MULTIPLE COMPARISONS ANALYSIS
frdAllPairsNemenyiTest(data.prop2)
colMeans(data.prop2)

# ========================================================================



# ========================================================================
# VISUALIZATIONS
# ========================================================================

lang.order = c("SL", "coSL", "TSL", "TcoSL", "SP", "coSP", "LT", "TLT",
               "PT", "LTT", "TLTT", "PLT", "TPLT", "SF", "Zp", "Reg")
size.order = c("Small", "Mid", "Large")
test.order = c("SR", "LR", "SA", "LA")
nn.order = c("Simple RNN", "GRU", "LSTM", "2-layer LSTM", "Transformer")
drc.order = c("1Q", "2Q", "3Q", "4Q")

( # VISUALIZE CAST.DF, i.e. CLASS ACCURACY AGGREGATED OVER TIER, k, j, i
  ggplot(stack(cast.df), aes(x=ind, y=values))
  + geom_boxplot()
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy")
)

jpeg("acc_class_alph.jpeg", units="in", width=10, height=5, res=300)
(
  ggplot(data, aes(x=class, y=accuracy, fill=alph))
  + geom_boxplot(outlier.shape=NA)
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class and Alphabet Size")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy", fill="Alphabet Size")
)
dev.off()

jpeg("acc_k_alph.jpeg", units="in", width=10, height=5, res=300)
( # ACCURACY BY k VALUES
  ggplot(data, aes(x=factor(k), y=accuracy, fill=alph))
  + geom_boxplot(outlier.shape=NA)
  + ggtitle("Accuracy by k Values and Alphabet Size")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="k Values", y="Accuracy", fill="Alphabet Size")
)
dev.off()

jpeg("acc_class_trainsize.jpeg", units="in", width=10, height=5, res=300)
(
  ggplot(data, aes(x=class, y=accuracy, fill=factor(train_set_size, levels=size.order)))
  + geom_boxplot(outlier.shape=NA)
  + scale_fill_discrete(limits=size.order)
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class and Training Set Size")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy", fill="Training Set Size")
)
dev.off()

jpeg("acc_class_test.jpeg", units="in", width=10, height=5, res=300)
(
  ggplot(data, aes(x=class, y=accuracy, fill=factor(test_type, levels=test.order)))
  + geom_boxplot(outlier.shape=NA)
  + scale_fill_discrete(limits=test.order)
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class and Test Type")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy", fill="Test Type")
)
dev.off()

jpeg("acc_class_test_large.jpeg", units="in", width=10, height=5, res=300)
(
  ggplot(data[data$train_set_size=="Large",], aes(x=class, y=accuracy, fill=factor(test_type, levels=test.order)))
  + geom_boxplot(outlier.shape=NA)
  + scale_fill_discrete(limits=test.order)
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class and Test Type")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy", fill="Test Type")
)
dev.off()


jpeg("acc_class_nn.jpeg", units="in", width=12, height=5, res=300)
(
  ggplot(data, aes(x=class, y=accuracy, fill=factor(network_type, levels=nn.order)))
  + geom_boxplot(outlier.shape=NA)
  + scale_fill_discrete(limits=nn.order)
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class and Network Type")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy", fill="Network Type")
)
dev.off()

jpeg("acc_class_nn_large.jpeg", units="in", width=12, height=5, res=300)
(
  ggplot(data[data$train_set_size=="Large",], aes(x=class, y=accuracy, fill=factor(network_type, levels=nn.order)))
  + geom_boxplot(outlier.shape=NA)
  + scale_fill_discrete(limits=nn.order)
  + scale_x_discrete(limits=lang.order)
  + ggtitle("Accuracy by Class and Network Type (Large Train Set Size)")
  + theme(plot.title=element_text(hjust=0.5))
  + labs(x="Language Class", y="Accuracy", fill="Network Type")
)
dev.off()


#
# EXTRAS
#

# CREATE AGGREGATE MATRIX FOR NNs VS LANGUAGE CLASSES

trainsizes <- c('Small', 'Mid', 'Large')

for (size in trainsizes) {

  df.tmp = data[data$train_set_size == size,]
  agg.tmp = aggregate(df.tmp$accuracy,
                      by=list(network_type=df.tmp$network_type,
                              class=df.tmp$class),
                      FUN=mean)
  agg = acast(agg.tmp, network_type ~ class, value.var="x")
  print(size)
  print(agg)
}
