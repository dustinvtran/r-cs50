#
# "Data Analysis in R", CS 50 seminar
# Author: Dustin Tran <dustinvtran.com>
#

# Functions!
id <- function(x) {
  x
}
id("hello")
id(123)
x <- c(1234L, 5L)
id(x)

compare <- function(x, a=5) {
  if (x > a) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}
compare(3)
compare(3, a=2)
# What about comparing for more general data structures?

################################################################################
# Interactive visuals
# install.packages("shiny"); library(shiny)
# http://shiny.rstudio.com/gallery/
################################################################################
iris
head(iris)
sapply(iris, class)
# Uhh, factor?

################################################################################
# Data exploration
# https://github.com/berkeley-scf/r-bootcamp-2013/blob/master/modules/module6_analysis.Rmd
################################################################################
library(foreign)
dat <- read.dta("data/2004_labeled_processed_race.dta")

# Exploratory data analysis
# some useful functions: plot(), hist(), boxplot(), coplot(), pairs()
head(dat, 20)
names(dat)
unique(dat$age9)
summary(as.numeric(dat$age9))
boxplot(as.numeric(dat$age9)
ggplot(dat, aes(x=age9)) + geom_bar()

gender <- dat$sex
gender <- gender[!is.na(gender)]
unique(race)

length(gender[gender=="male"])/length(gender)*100
length(gender[gender=="female"])/length(gender)*100
ggplot(dat, aes(x=sex)) + geom_bar()

race <- dat$race
race <- race[!is.na(race)]
unique(race)

length(race[race=="white"])/length(race)*100
length(race[race=="black"])/length(race)*100
length(race[race=="hispanic/latino"])/length(race)*100
length(race[race=="asian"])/length(race)*100
length(race[race=="other"])/length(race)*100

# or...

for (i in unique(race)) {
  print(i)
  print(length(race[race==i])/length(race)*100)
}

# More complicated questions:
# Suppose we wanted to know how voting behavior in the 2004 Presidential
# Election varies by race.

# Set default repo for CRAN.
options(repos = c(CRAN = "http://cran.r-project.org/"))
# How to download packages.
install.packages("dplyr")
library(dplyr)
library(ggplot2)

dat %>%
    group_by(race, pres04) %>%
    filter(pres04 %in% c(1,2)) %>%
    summarize(count=n()) %>%
    ggplot(aes(x=race, y=count, fill=pres04))  +
      geom_bar(stat="identity")

################################################################################
# Topic Modelling
# Blei, David M. and Ng, Andrew and Jordan, Michael. Latent Dirichlet
# allocation. Journal of Machine Learning Research, 2003
################################################################################
library(lda)
library(ggplot2)
library(reshape2)

data(cora.documents)
data(cora.vocab)

theme_set(theme_bw())
set.seed(42)
K <- 10 # Num clusters
result <- lda.collapsed.gibbs.sampler(cora.documents,
                                      K,  ## Num clusters
                                      cora.vocab,
                                      25,  ## Num iterations
                                      0.1,
                                      0.1,
                                      compute.log.likelihood=TRUE)

# Get the top words in the cluster.
top.words <- top.topic.words(result$topics, 5, by.score=TRUE)

# Number of documents to display.
N <- 10

topic.proportions <- t(result$document_sums) / colSums(result$document_sums)
topic.proportions <- topic.proportions[sample(1:dim(topic.proportions)[1], N),]
topic.proportions[is.na(topic.proportions)] <-  1 / K

colnames(topic.proportions) <- apply(top.words, 2, paste, collapse=" ")
topic.proportions.df <- melt(cbind(data.frame(topic.proportions),
                                   document=factor(1:N)),
                             variable.name="topic",
                             id.vars = "document")

qplot(topic, value, fill=document, ylab="proportion",
      data=topic.proportions.df, geom="bar", stat="identity") +
  theme(axis.text.x = element_text(angle=90, hjust=1)) +
  coord_flip() +
  facet_wrap(~ document, ncol=5)

################################################################################
# Machine Learning
# https://github.com/petewerner/ml-examples
################################################################################
# Install and load packages.
install.packages(c("nnet", "e1071", "randomForest"))
library(nnet)
library(e1071)
library(randomForest)

# Data
data <- read.csv("data/simulated.csv")
head(data, 20)

x0 <- data[data$Y == 0, 1:2]
x1 <- data[data$Y == 1, 1:2]
symbols(x0, circles=rep(0.5, nrow(x0)), inches=FALSE, bg="yellow")
symbols(x1, squares=rep(1, nrow(x1)), inches=FALSE, add=TRUE, bg="red")

# Linear Regression
model.lm <- lm(Y~., data=data)

x1.grid <- seq(min(data[,1]), max(data[,1]), length=100)
x2.grid <- seq(min(data[,2]), max(data[,2]), length=100)
df <- data.frame(x1.grid, x2.grid)
names(df) <- names(data)[1:2]

preds <- predict(model.lm, expand.grid(df))
zz <- matrix(as.numeric(preds), nrow=nrow(df), byrow=T)

symbols(x0, circles=rep(0.5, nrow(x0)), inches=FALSE, bg="yellow")
symbols(x1, squares=rep(1, nrow(x1)), inches=FALSE, add=TRUE, bg="red")
contour(x1.grid, x2.grid, zz, add=T, levels=0.5)

# Neural Network
model.nnet <- nnet(Y~., data=data, size=3, rang=0.001, decay=0.0001, maxit=400)

preds <- predict(model.nnet, expand.grid(df))
zz <- matrix(as.numeric(preds), nrow=nrow(df), byrow=T)

symbols(x0, circles=rep(0.5, nrow(x0)), inches=FALSE, bg="yellow")
symbols(x1, squares=rep(1, nrow(x1)), inches=FALSE, add=TRUE, bg="red")
contour(x1.grid, x2.grid, zz, add=T, levels=0.5)

# Support Vector Machine
data[,3] <- as.factor(data[,3])
model.svm <- svm(Y~., data=data, cost=0.03, gamma=1, coef0=10, kernel="polynomial")

preds <- predict(model.svm, expand.grid(df))
zz <- matrix(as.numeric(preds), nrow=nrow(df), byrow=T)

symbols(x0, circles=rep(0.5, nrow(x0)), inches=FALSE, bg="yellow")
symbols(x1, squares=rep(1, nrow(x1)), inches=FALSE, add=TRUE, bg="red")
contour(x1.grid, x2.grid, zz, add=T, levels=2)

# making this slightly more difficult
set.seed(42)
data[sample(which(data$Y == 0), 3), "Y"] <- 1
data[sample(which(data$Y == 1), 3), "Y"] <- 0
# rewriting as functions

