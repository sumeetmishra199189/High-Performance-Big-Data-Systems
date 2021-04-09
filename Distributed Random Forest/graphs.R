library(ggplot2)

library(readxl)

HVDdata = read_excel("C:/Users/ntihish/Documents/IUB/HPC/Project/Midway report/Horovod Readings.xlsx")
ggplot(HVDdata)%>% geom_smooth(mapping = aes(x=)) 