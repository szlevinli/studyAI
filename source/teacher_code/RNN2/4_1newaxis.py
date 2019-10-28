#https://blog.csdn.net/molu_chase/article/details/78619731

np.arange(0, 10)

#methond 1
y=np.arange(1, 11)
y.shape=(10,1)
print(y)

#method2 
print(np.arange(0, 10)[:, np.newaxis])
