# train.py
"""
Train script:
python train.py --train_dir /path/to/train --valid_dir /path/to/valid --output_dir ./outputs --model_type custom
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from sklearn.utils import class_weight

from model import build_model
import config

def make_datasets(train_dir, valid_dir, img_size, batch_size, seed=42):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, label_mode='binary', image_size=img_size, batch_size=batch_size, shuffle=True, seed=seed
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_dir, label_mode='binary', image_size=img_size, batch_size=batch_size, shuffle=False
    )
    # normalize inside mapping
    AUTOTUNE = tf.data.AUTOTUNE
    def _prep(x,y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y
    train_ds = train_ds.map(_prep, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.map(_prep, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    return train_ds, val_ds

def compute_class_weights_from_generator(train_dir):
    # count classes from directory structure
    labels = []
    for cls_idx, cls in enumerate(sorted(os.listdir(train_dir))):
        cls_path = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        n = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        labels += [cls_idx] * n
    classes = np.unique(labels)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=np.array(labels))
    return dict(enumerate(weights))

def try_focal_loss():
    try:
        import tensorflow_addons as tfa
        return tfa.losses.SigmoidFocalCrossEntropy(alpha=0.75, gamma=2.0, from_logits=False)
    except Exception:
        return tf.keras.losses.BinaryCrossentropy()

def main(args):
    # config override
    img_size = (args.img_size, args.img_size)
    train_ds, val_ds = make_datasets(args.train_dir, args.valid_dir, img_size, args.batch_size, seed=args.seed)

    # Build model
    model = build_model(choice=args.model_type, input_shape=(img_size[0], img_size[1], 3), dropout_rate=args.dropout)

    # Compile
    opt = optimizers.Adam(learning_rate=args.lr)
    loss = try_focal_loss()
    model.compile(optimizer=opt, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                    tf.keras.metrics.Precision(name='precision'),
                                                    tf.keras.metrics.Recall(name='recall'),
                                                    tf.keras.metrics.AUC(name='auc')])
    model.summary()

    # Callbacks
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = callbacks.ModelCheckpoint(os.path.join(args.output_dir, 'best_model.keras'),
                                           save_best_only=True, monitor='val_auc', mode='max', verbose=1)
    early = callbacks.EarlyStopping(monitor='val_auc', patience=4, mode='max', restore_best_weights=True, verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
    cbs = [checkpoint, early, reduce_lr]

    # class weights
    cw = compute_class_weights_from_generator(args.train_dir)
    print("Class weights:", cw)

    # Fit
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cbs,
        class_weight=cw
    )

    # Save final
    final_path = os.path.join(args.output_dir, 'final_model.keras')
    model.save(final_path)
    print("Saved final model to:", final_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default=config.TRAIN_DIR)
    parser.add_argument('--valid_dir', type=str, default=config.VALID_DIR)
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR)
    parser.add_argument('--model_type', type=str, default=config.MODEL_TYPE, choices=['custom','efficientnet'])
    parser.add_argument('--img_size', type=int, default=config.IMG_SIZE[0])
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--dropout', type=float, default=config.DROP_OUT)
    parser.add_argument('--seed', type=int, default=config.SEED)
    args = parser.parse_args()
    main(args)
