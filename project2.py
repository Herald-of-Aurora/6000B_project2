import pandas as pd
import numpy as np
from PIL import Image

width=128

def readdata(filename):
    train=[]
    
    with open(filename) as f:
        line = f.readline()
        while line:
            line=line.split()
            train.append(line)
            # print line
            line = f.readline()
            
    
    train=pd.DataFrame(train)
    #print train
    
    
    add=train[0]
    if (train.shape[1]<2):
        y_train=[]
    else:
        y_train=np.array(train[1], dtype='uint8').reshape(len(add),1)
    
    
    #print y_train
    #print add
    
    
    
    X_train=[]
    
    for i in add:
    
        img = Image.open(i)
        img=img.resize((width,width))
        
        data = img.getdata()
        data = np.array(data)
        data=data.tolist()
        #data = data.reshape(width,width,3)
        
        #new_im = Image.fromarray(data.astype('uint8'))
        #plt.imshow(new_im)
        
        
        X_train.append(data)
        #X_train=np.array(X_train)
        
    X_train=np.array(X_train, dtype='uint8')
    X_train=X_train.reshape(len(add), width, width,3)
    
    #plt.imshow(img)
    #X_train=pd.DataFrame(X_train)
    
    return (X_train, y_train)



import time
start = time.clock()


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
#from keras import regularizers
from keras import utils
import matplotlib.pyplot as plt


#import keras
print("import complete")


(X_train, y_train)=readdata('train.txt')
print("train data read complete")

(X_test, y_test)=readdata('val.txt')
print("test data read complete")

(X_task, _)=readdata('test.txt')
print("task data read complete")



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
y_train = utils.to_categorical(y_train, 5)
y_test = utils.to_categorical(y_test, 5)







model = Sequential()

model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(width,width,3)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
#model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(5, activation='softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer='adam', #'sgd'
              metrics=['accuracy'])

print("training start")

result = model.fit(X_train, y_train, 
                   batch_size=32, epochs=50, verbose=1,
                   validation_data=(X_test, y_test)
                   )

#score = model.evaluate(X_test, y_test, verbose=0)
#print(score)

plt.figure
plt.plot(result.epoch,result.history['acc'],label="acc")
plt.plot(result.epoch,result.history['val_acc'],label="val_acc")
plt.scatter(result.epoch,result.history['acc'],marker='*')
plt.scatter(result.epoch,result.history['val_acc'])
plt.legend(loc='under right')
plt.show()

model.save("8hours.h5")








y_task=model.predict(X_task)

prediction=[]
for i in y_task:
    for j in range(0,5):
        if i[j]>0.5:
            prediction.append(j)

with open('project2_20461901.txt',"w") as f:
    for i in prediction:
        f.write(str(i))
        f.write("\n")
f.close()





elapsed = (time.clock() - start)
print("Time used: "+str(elapsed)+"s")