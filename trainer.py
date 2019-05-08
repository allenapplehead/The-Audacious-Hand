# PROCEDURE


# 1. CSV all the data
# 2. Make pd dataframe
# 3. Make all data ints
# 4. DROP WHITE VALUES ROWS
# 5. Normalize
# 6. Plot into one graph with labels on which sign it belongs to
# 7. Use KNN

### IDEAS:

# 1. Find average of color points and align them like that before stripping whitespace
# 2. Use colors as part of the feature set (greyscale)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('ASLalpha.csv')
print(df.head())

# Drop rgb and pixel magick rows
df = df.drop('ImageMagick', axis=1)
df = df.drop('172,172,65535,gray', axis=1)
print(df.head())

# Split x y pixel values into xp yp separate columns
spl1 = df['#'].str.split(',', n=1, expand=True)

df['xp'] = spl1[0]
df['yp'] = spl1[1]

df['yp'] = df['yp'].apply(lambda x: x.strip(":"))

df = df.drop('#', axis=1)

print(df.head())

# Cast the dataframe into int (except for the hex string)
print(df.dtypes)
df['xp'] = df['xp'].astype(int)
df['yp'] = df['yp'].astype(int)
print(df.dtypes)

# Strip all the pixels with white as their color

df = df[~df.pixel_enumeration.str.contains("#FFFFFF")]
df.reset_index()
# Create new index
indices = []
for i in range(len(df['xp'])):
    indices.append(i)
df['indices'] = indices
df.set_index('indices', inplace=True)
df.reset_index()
print("OLD and WORN:", df.head(20))

# Center all images for each image

Xp = df['xp']
Yp = df['yp']
Xp = Xp.tolist()
Yp = Yp.tolist()

XpMAX = [] # Ordered from a to z in training set
XpMIN = []
YpMAX = []
YpMIN = []

lasty = -1000000
count = 0
lastcount = 0
for y in Yp:
    if lasty > y:
        tempx = Xp[lastcount:count]
        tempy = Yp[lastcount:count]
        XpMAX.append(max(tempx))
        XpMIN.append(min(tempx))
        YpMAX.append(max(tempy))
        YpMIN.append(min(tempy))
        lastcount = count
    count += 1
    lasty = y
else:
    tempx = Xp[lastcount:count]
    tempy = Yp[lastcount:count]
    XpMAX.append(max(tempx))
    XpMIN.append(min(tempx))
    YpMAX.append(max(tempy))
    YpMIN.append(min(tempy))

# Now we can take the midpoints of x and y as the center
centerx = []
centery = []
for i in range(26):
    avgx = (XpMAX[i] + XpMIN[i]) / 2
    avgy = (YpMAX[i] + YpMIN[i]) / 2
    centerx.append(avgx)
    centery.append(avgy)
# Where we should really center our data
# Dimensions are 172x172, but listed as 171x171 because it uses 0
# 172 / 2 = 86
# Therefore, the picture should be centered at 85x85
truecenterx = 85
truecentery = 85

# Calculate the x direction error and y direction error
xerror = []
yerror = []

for i in range(26):
    errx = truecenterx - centerx[i]
    erry = truecentery - centery[i]

    xerror.append(errx)
    yerror.append(erry)

# Set and replace all data with the correct orientation of the center based on their error
lasty = -100000
count = 0
countalpha = 0
debug = 0

for y in df['yp']:
    modx = df.at[count, 'xp'] + xerror[countalpha]
    mody = df.at[count, 'yp'] + yerror[countalpha]
    df.at[count, 'xp'] = modx
    df.at[count, 'yp'] = mody
    if lasty > y:
        countalpha += 1
    count += 1
    lasty = y


print("NEW and CENTERED!!!:", df.head(20))


#else:
#    temp = yhat[lastcount:count]
#    modem = mode(temp)
#    yhatletters.append(modem)

"""
# Normalize numerical data
X_xp = df['xp']
X_yp = df['yp']
X_xp = preprocessing.StandardScaler().fit(X_xp).transform(X_xp.astype(float))
X_yp = preprocessing.StandardScaler().fit(X_yp).transform(X_yp.astype(float))
print(X_xp[0:5], X_yp[0:5])

# Plot the data
plt.scatter(df['xp'], df['yp'], c='b', s=0.5)
plt.xlabel("xp")
plt.ylabel("yp")
plt.show()
plt.close()
"""

# Create X, y train sets
Xtrain = df[['xp', 'yp']].values
print(Xtrain[0:5])
ytrain = df['signs'].values
print(ytrain[0:5])

#input("Press ENTER to build test")

### OUR DATASET IS READY, WE USE KNN ###

dftest = pd.read_csv('test.csv')
print(dftest.head())

# Drop rgb and pixel magick rows
dftest = dftest.drop('ImageMagick', axis=1)
dftest = dftest.drop('172,172,65535,srgb', axis=1)
print(dftest.head())

# Split x y pixel values into xp yp separate columns
spl1test = dftest['#'].str.split(',', n=1, expand=True)

dftest['xp'] = spl1test[0]
dftest['yp'] = spl1test[1]

dftest['yp'] = dftest['yp'].apply(lambda x: x.strip(":"))

dftest = dftest.drop('#', axis=1)

print(dftest.head())

# Cast the dataframe into int (except for the hex string)
print(dftest.dtypes)
dftest['xp'] = dftest['xp'].astype(int)
dftest['yp'] = dftest['yp'].astype(int)
print(dftest.dtypes)

# Strip all the pixels with white as their color
dftest = dftest[~dftest.pixel_enumeration.str.contains("#FFFFFF")]
dftest.reset_index()
# Create new index
indicestest = []
for i in range(len(dftest['xp'])):
    indicestest.append(i)
dftest['indices'] = indicestest
dftest.set_index('indices', inplace=True)
dftest.reset_index()

# Center all images for each image

Xptest = dftest['xp']
Yptest = dftest['yp']
Xptest = Xptest.tolist()
Yptest = Yptest.tolist()

XpMAXtest = [] # Ordered from a to z in training set
XpMINtest = []
YpMAXtest = []
YpMINtest = []

lasty = -1000000
count = 0
lastcount = 0
for y in Yptest:
    if lasty > y:
        tempxtest = Xptest[lastcount:count]
        tempytest = Yptest[lastcount:count]
        XpMAXtest.append(max(tempxtest))
        XpMINtest.append(min(tempxtest))
        YpMAXtest.append(max(tempytest))
        YpMINtest.append(min(tempytest))
        lastcount = count
    count += 1
    lasty = y
else:
    tempxtest = Xptest[lastcount:count]
    tempytest = Yptest[lastcount:count]
    XpMAXtest.append(max(tempxtest))
    XpMINtest.append(min(tempxtest))
    YpMAXtest.append(max(tempytest))
    YpMINtest.append(min(tempytest))

# Now we can take the midpoints of x and y as the center
centerxtest = []
centerytest = []
for i in range(26):
    avgx = (XpMAX[i] + XpMIN[i]) / 2
    avgy = (YpMAX[i] + YpMIN[i]) / 2
    centerxtest.append(avgx)
    centerytest.append(avgy)
# Where we should really center our data
# Dimensions are 172x172, but listed as 171x171 because it uses 0
# 172 / 2 = 86
# Therefore, the picture should be centered at 85x85
truecenterxtest = 85
truecenterytest = 85

# Calculate the x direction error and y direction error
xerrortest = []
yerrortest = []

for i in range(26):
    errx = truecenterxtest - centerxtest[i]
    erry = truecenterytest - centerytest[i]

    xerrortest.append(errx)
    yerrortest.append(erry)

# Set and replace all data with the correct orientation of the center based on their error
lasty = -100000
count = 0
countalpha = 0

for y in dftest['yp']:
    modx = dftest.at[count, 'xp'] + xerrortest[countalpha]
    mody = dftest.at[count, 'yp'] + yerrortest[countalpha]
    dftest.at[count, 'xp'] = modx
    dftest.at[count, 'yp'] = mody
    if lasty > y:
        countalpha += 1
    count += 1
    lasty = y

print("NEW and CENTERED TEST!!!:", dftest.head(20))
input("run KNN!")

"""
# Normalize numerical data
X_xp = df['xp']
X_yp = df['yp']
X_xp = preprocessing.StandardScaler().fit(X_xp).transform(X_xp.astype(float))
X_yp = preprocessing.StandardScaler().fit(X_yp).transform(X_yp.astype(float))
print(X_xp[0:5], X_yp[0:5])


# Plot the data
plt.scatter(dftest['xp'], dftest['yp'], c='b', s=0.5)
plt.xlabel("xp")
plt.ylabel("yp")
plt.show()
plt.close()
"""

# Create X, y test sets
Xtest = dftest[['xp', 'yp']].values
print(Xtest[0:5])
ytest = dftest['signs'].values
print(ytest[0:5])

#input("Press ENTER to build model")

### OUR TESTSET IS READY, WE USE KNN ###

k = 2
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(Xtrain, ytrain)
yhat = neigh.predict(Xtest)
print(yhat[0:5])
print(len(yhat), len(ytest))

##### EVALUATION of individual points using the JACCARD index
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(ytrain, neigh.predict(Xtrain)))
print("Test set Accuracy: ", metrics.accuracy_score(ytest, yhat))

##### EVALUATION

yletters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Read and format the machine's predictions
from statistics import mode
count = 0
lastcount = 0
lasty = -10000000
yhatletters = []
temp = None
listcount = 0
yhat = yhat.tolist()


for y in dftest['yp']:
    if lasty > y:
        listcount += 1
        temp = yhat[lastcount:count]
        modem = mode(temp)
        yhatletters.append(modem)
        lastcount = count
    count += 1
    lasty = y
else:
    temp = yhat[lastcount:count]
    modem = mode(temp)
    yhatletters.append(modem)
print(yhatletters)
