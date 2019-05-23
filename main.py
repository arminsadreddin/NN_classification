import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist # images 28*28 of hand written digits 0-9
(x_train,y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) # 28*28 -- khati shavad

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss,val_acc)

# print(x_train[7])
# plt.imshow(x_train[7], cmap = plt.cm.binary)
# plt.show()

#model.save('mnist_pred.model')
#new_model = tf.keras.models.load_model('mnist_pred.model')


predictions = model.predict([x_test])
#print(predictions)
print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()






