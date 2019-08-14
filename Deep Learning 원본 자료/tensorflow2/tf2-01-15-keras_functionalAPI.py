import tensorflow as tf

# Step 1. Data Load
(train_x,train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

# Step 1. Data Preprocessing
train_x = train_x.reshape(len(train_x), -1).astype('float32') / 255
test_x = test_x.reshape(len(test_x), -1).astype('float32') / 255
train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

# Step 2. Functional API Model
# This returns a tensor 
inputs = tf.keras.layers.Input(shape=(784,)) 

# a layer instance is callable on a tensor, and returns a tensor 
x = tf.keras.layers.Dense(64, activation='relu')(inputs) 
x = tf.keras.layers.Dense(64, activation='relu')(x) 
predictions = tf.keras.layers.Dense(10, activation='softmax')(x) 

# This creates a model that includes the Input layer and three Dense layers 
model = tf.keras.models.Model(inputs=inputs, outputs=predictions) 


# Step 3. Loss,Optimizer, Metric
model.compile (optimizer= 'rmsprop', 
                loss='categorical_crossentropy', metrics = ['accuracy'])

# Step 4. Train the model
model.fit(train_x, train_y, batch_size=100, epochs=10)
print(model.summary())

# Step 5. Test the model
test_loss, test_acc = model.evaluate(test_x, test_y)
print('test_loss = ', test_loss, 'test_acc = ', test_acc)