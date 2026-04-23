D = read.table("Distances.txt", header=TRUE)
boxplot(Distance ~ Algorithm, data = D)
aov.out = aov(Distance ~ Algorithm, data = D)
plot(aov.out)
summary(aov.out)
pairwise.t.test(D$Distance,D$Algorithm,p.adjust.method="holm")

D = read.table("Distances_no_Freight.txt", header=TRUE)
boxplot(Distance ~ Algorithm, data = D)
aov.out = aov(Distance ~ Algorithm, data = D)
plot(aov.out)
summary(aov.out)
pairwise.t.test(D$Distance,D$Algorithm,p.adjust.method="holm")

D = read.table("CO2.txt", header=TRUE)
boxplot(CO2 ~ Algorithm, data = D)
aov.out = aov(CO2 ~ Algorithm, data = D)
plot(aov.out)
summary(aov.out)
pairwise.t.test(D$CO2,D$Algorithm,p.adjust.method="holm")





