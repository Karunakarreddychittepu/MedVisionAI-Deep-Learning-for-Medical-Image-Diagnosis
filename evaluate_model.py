import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns

model = tf.keras.models.load_model('models/vgg16_medvision.h5')

# Load your validation generator again
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = 'data/processed'
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

y_true = val_gen.classes
y_pred_prob = model.predict(val_gen)
y_pred = (y_pred_prob > 0.5).astype("int32")

print(classification_report(y_true, y_pred))
print("AUC:", roc_auc_score(y_true, y_pred_prob))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
plt.show()
