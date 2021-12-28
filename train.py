import os, math
import tensorflow as tf
import numpy as np
from arcface import ResNet50, ArcFaceLayer
from tqdm import tqdm

# Parameters
training_epochs = 16
learning_rate = 0.025
momentum = 0.9
#feature_dims = 512
#s = 64
#margin = 0.5
batch_size_per_GPU = 128
model_save_path = './models/'
dataset_path = './dataset/VGGFace2/'
dataset_split_random_seed = 48
validation_split_rate = 0.05

# Initiallization
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
class_list = []
with open('../dataset/VGGFace2/class_name.txt', 'r') as f:
    line = f.readline().strip()
    while line: 
        class_list.append(line)
        line = f.readline().strip()
strategy = tf.distribute.MirroredStrategy()
batch_size = strategy.num_replicas_in_sync * batch_size_per_GPU

# Loss Function
with strategy.scope():
    def cross_entropy(y_true, y_pred):
        return tf.reduce_sum(y_true * -1 * tf.math.log(tf.clip_by_value(y_pred,1e-20, 1.0)), axis=1)
    def compute_loss(loss_func, y_true, y_pred):
        return tf.reduce_mean(loss_func(y_true, y_pred))

# Model
with strategy.scope():
    backbone = ResNet50()
    model_input = backbone.input
    model_label_input = tf.keras.Input([len(class_list)])
    classifier = ArcFaceLayer(len(class_list))
    train_output = classifier(backbone(model_input), labels=model_label_input)
    infer_output = classifier(backbone(model_input))
    train_model = tf.keras.Model(inputs=[model_input, model_label_input], outputs=train_output, name='arcface_train_model')
    infer_model = tf.keras.Model(inputs=model_input, outputs=infer_output, name='arcface_infer_model')
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum)


# Loss and Accuracy Tracing
with strategy.scope():
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')


# Load Datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_path, 
    label_mode='categorical', 
    class_names=class_list, 
    batch_size=batch_size, 
    image_size=(112, 112), 
    validation_split=validation_split_rate, 
    seed=dataset_split_random_seed, 
    subset='training')
train_ds = train_ds.map(lambda x, y: ((x - 127.5) / 128., y))
train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
valid_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_path, 
    label_mode='categorical', 
    class_names=class_list, 
    batch_size=batch_size, 
    image_size=(112, 112), 
    validation_split=validation_split_rate,  
    seed=dataset_split_random_seed, 
    subset='validation')
valid_ds = valid_ds.map(lambda x, y: ((x - 127.5) / 128., y))
valid_dist_ds = strategy.experimental_distribute_dataset(valid_ds)

# Model Functions
with strategy.scope():
    def train_step(inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            preds = train_model([images, labels], training=True)
            loss = compute_loss(cross_entropy, labels, preds)
        gradients = tape.gradient(loss, train_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, train_model.trainable_variables))
        accuracy = tf.math.reduce_mean(tf.cast(tf.math.equal(tf.math.argmax(preds, axis=1), 
                                                                    tf.math.argmax(labels, axis=1)), tf.float32)) 
        train_accuracy.update_state(labels, preds)
        return (loss, accuracy)
    def valid_step(inputs):
        images, labels = inputs
        preds = infer_model(images, training=False)
        loss = compute_loss(cross_entropy, labels, preds)
        valid_loss.update_state(loss)
        valid_accuracy.update_state(labels, preds)
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_results = strategy.run(train_step, args=(dataset_inputs,))
        per_replica_losses = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[0], axis=None)
        per_replica_accuracies = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[1], axis=None)
        return (per_replica_losses, per_replica_accuracies)
    @tf.function
    def distributed_valid_step(dataset_inputs):
        return strategy.run(valid_step, args=(dataset_inputs,))

    total_iterations = 0
    best_valid_acc = 0.0
    best_epoch = 0
    for epoch in range(training_epochs):
        # Training
        total_loss = 0.0
        num_batches = 0
        t = tqdm(train_dist_ds, total=len(train_ds), desc='Training')
        for batch_index, train_batch in enumerate(t):
            if batch_index >= len(train_ds) or train_batch[-1].values[0].shape[0] != batch_size_per_GPU:
                break
            loss, accuracy = distributed_train_step(train_batch)
            total_loss += loss
            num_batches += 1
            total_iterations += 1
            if total_iterations == 80000 or total_iterations == 112000:
                learning_rate /= 10
                optimizer.lr = learning_rate
            template = ("Training Loss: {:.4f}, Acc: {:.4f}%    Batch Loss: {:.4f}, Acc: {:.4f}%")
            t.set_description(template.format(total_loss/num_batches, 
                train_accuracy.result()*100, 
                loss, accuracy*100))
            t.refresh()
        train_loss = total_loss / num_batches

        # Validation
        t = tqdm(valid_dist_ds, total=len(valid_ds), desc='Validation')
        for batch_index, valid_batch in enumerate(t):
            distributed_valid_step(valid_batch)
            template = ("Validation Loss: {:.6f}, Acc: {:.4f}%")
            t.set_description(template.format(valid_loss.result(), valid_accuracy.result()*100))
            t.refresh()

        # Print Informations And Save Check Point
        if float(valid_accuracy.result()) >= best_valid_acc:
            best_epoch = epoch+1
            best_valid_acc = float(valid_accuracy.result())
            infer_model.save(model_save_path + 'checkpoint.h5')
        template = ("Epoch {}, Loss: {}, Acc: {}%   Validation Loss: {}, Acc: {}%")
        print(template.format(epoch+1, train_loss, train_accuracy.result()*100, valid_loss.result(), valid_accuracy.result()*100))
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        
    os.rename(model_save_path + 'checkpoint.h5', model_save_path + "{}_{:.4f}.h5".format(best_epoch, best_valid_acc*100))