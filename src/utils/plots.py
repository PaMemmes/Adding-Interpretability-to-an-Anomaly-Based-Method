import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt

def plot_confusion_matrix(cm, savefile, name, cmap=plt.cm.Greens):
    fig, ax = plt.subplots()
    cm = np.around(cm, decimals=6)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.xticks(np.arange(2), ['Normal','Anomaly'], rotation=45)
    plt.yticks(np.arange(2), ['Normal','Anomaly'])
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(savefile)
    plt.close('all')

def plot_accuracy(history, savefile, name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in percent')
    plt.savefig(savefile)
    plt.close('all')
    
def plot_loss(history, savefile, name):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(savefile)
    plt.close('all')
    
def plot_roc(tpr, fpr, roc_auc, savefile, name):
    fig, ax = plt.subplots()
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lime', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(savefile)
    plt.close('all')

def plot_losses(gen_loss, dis_loss, savefile):
    fig, ax = plt.subplots()
    plt.plot(dis_loss, label='Discriminator')
    plt.plot(gen_loss, label='Generator')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(savefile)
    plt.close('all')

def plot_precision_recall(y_test, y_preds, savefile):
    fig, ax = plt.subplots()
    print(y_test.shape)
    print(y_preds.shape)
    skplt.metrics.plot_precision_recall(y_test, y_preds)
    plt.savefig(savefile)
    plt.close('all')
