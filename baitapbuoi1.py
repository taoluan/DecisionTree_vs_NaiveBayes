import pandas as pd
import matplotlib.pyplot as plt
#Bai 1
data = pd.read_csv("baitap1.csv", delimiter=",")
# Bai 2
print(data)
#Bai 3
print(data.iloc[:,2:3])
#Bai 4
print(data.iloc[3:9])
#Bai 5
print(data.iloc[3:4,0:2])
#Bai 6
x = data.iloc[:,1:2]
#print (x)
y = data.iloc[:,2:3]
#print (y)
plt.scatter(x,y)
plt.title("Bai tap 1")
plt.xlabel("Tuoi")
plt.ylabel("Can Nang")
#plt.xticks([],[%w for w'kg'])
plt.grid()
plt.show()
#Bai 7
for i in range(1,50):
    if i % 2 != 0 :
        print (i)
