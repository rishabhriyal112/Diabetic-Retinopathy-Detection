# evaluate.py
"""
Usage:
python evaluate.py --model_path outputs/best_model.keras --test_dir /path/to/test --output_dir outputs/eval
"""

import argparse, os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(test_dir, img_size=(224,224), batch_size=32):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir, label_mode='binary', image_size=img_size, batch_size=batch_size, shuffle=False
    )
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds

def ensure_positive_label_is_DR(dataset):
    # Returns (dataset, class_names, dr_index) and ensures we know which label corresponds to 'DR'
    # For binary label_mode, class_names are in dataset.class_names and labels are 0/1 in that order.
    class_names = dataset.class_names
    return class_names

def evaluate_model(model_path, test_dir, output_dir, img_size=(224,224), batch_size=32):
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset(test_dir, img_size, batch_size)
    class_names = ds.class_names  # e.g. ['DR','No_DR'] in alphabetical order

    model = tf.keras.models.load_model(model_path)
    # predict probabilities (sigmoid)
    y_probs = []
    y_true = []
    for x, y in ds:
        preds = model.predict(x, verbose=0).ravel()
        y_probs.extend(preds.tolist())
        y_true.extend(y.numpy().astype(int).tolist())

    y_probs = np.array(y_probs)
    y_true = np.array(y_true)

    # Determine which label is DR in class_names
    # We want probabilities for DR positive class. If class_names[0]=='DR', then current label mapping is:
    #   label 0 -> DR, label 1 -> No_DR
    # Our model outputs probability of class 1 (sigmoid) by convention, but since we trained with labels as-is,
    # the sigmoid probability corresponds to label=1. To get probability for DR, we must:
    #   if class_names[0] == 'DR': prob_DR = 1 - prob_label1  (since label 0 is DR)
    #   else: prob_DR = prob_label1
    if class_names[0].lower() == 'dr':
        prob_dr = 1.0 - y_probs
        dr_label = 0
    else:
        prob_dr = y_probs
        dr_label = 1

    # Predicted class (threshold 0.5 on prob_dr)
    y_pred = (prob_dr >= 0.5).astype(int)
    # But y_true may be label 0/1; convert y_true to DR=1 mapping for report convenience:
    if dr_label == 0:
        y_true_dr = (y_true == 0).astype(int)  # 1 if DR
    else:
        y_true_dr = (y_true == 1).astype(int)

    # classification report expects labels 0/1 -> we'll map 1=DR,0=No_DR for readability
    print("Classification report (DR positive = 1):")
    print(classification_report(y_true_dr, y_pred, target_names=['No_DR','DR'], digits=4))

    # confusion matrix (rows true, cols pred)
    cm = confusion_matrix(y_true_dr, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['No_DR','DR'], yticklabels=['No_DR','DR'])
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (DR positive=1)')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # ROC & AUC
    fpr, tpr, _ = roc_curve(y_true_dr, prob_dr)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (DR positive=1)')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Save CSV with probs and truths
    import pandas as pd
    df = pd.DataFrame({'y_true_dr': y_true_dr, 'prob_dr': prob_dr})
    df.to_csv(os.path.join(output_dir, 'predictions_probs.csv'), index=False)

    print("Saved confusion matrix, ROC, and predictions CSV to", output_dir)
    print("AUC:", roc_auc)

if __name__ == "__main__":
    parser = argparse = __import__('argparse').ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--output_dir', default='./outputs_eval')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    evaluate_model(args.model_path, args.test_dir, args.output_dir, (args.img_size,args.img_size), args.batch_size)
