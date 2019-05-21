from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import keras, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt


batch_size = 32 
input_size = 64
num_classes = 10 
epochs = 80
path = '' 

dataset = np.load(path + 'dataset.npz') 
imgs = dataset['data']
label = dataset['label'] 

print(imgs.shape,label.shape)

from sklearn.model_selection import train_test_split


n,h,w,c = imgs.shape
imgs = imgs.reshape(n, h, w, c).astype('float32')/255


label = np_utils.to_categorical(label)


x_train, x_test, y_train, y_test = train_test_split(imgs, label, test_size=0.2)




model = Sequential() 
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 padding='same',
                 input_shape=(input_size,input_size,3))) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 
model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 
model.add(Flatten()) 
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.25)) 
model.add(Dense(num_classes, activation='softmax'))





model.summary()


model.compile(loss=keras.losses.categorical_crossentropy, 
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy']) 


train_history = model.fit(
          x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split = 0.2)


def sh_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
sh_history(train_history, 'acc', 'val_acc')
sh_history(train_history, 'loss', 'val_loss')


score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


prediction = model.predict_classes(x_test, verbose=1) 
predict_score = model.predict_on_batch(x_test) 


def sh_img_prediction(images, labels, prediction, idx, num=15): 
    fig = plt.gcf()
    fig.set_size_inches(12,14) 
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5, 5, 1+i)
        ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
        #title="idx="+str(idx)+",ans="+str(labels[idx])
        label_o = np.argmax(labels, axis=1)
        title="label=" +str(label_o[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx])
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()

sh_img_prediction(x_test, y_test, prediction, 0)

test_label = np.argmax(y_test, axis=1) 
df = pd.crosstab(test_label, prediction, rownames=['label'], colnames=['predict'])
print(df)