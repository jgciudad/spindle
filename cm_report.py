import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score, precision_score, accuracy_score
from matplotlib import pyplot as plt

cm1 = np.array([[66844, 871, 1727, 0], [2326, 8382, 882, 0], [3141, 210, 102344, 0], [75, 10, 494, 0]])
disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=['NREM', 'REM', 'WAKE', 'Art'])
cm_plot = disp.plot(cmap=plt.cm.Blues)
# plt.savefig(os.path.join(save_path, 'cm_whole_model' + '.png'))
plt.show()

cm2 = np.array([[20680, 120, 7420, 0], [276, 3947, 1032, 0], [2034, 1398, 34262, 0], [5710, 732, 8781, 8]])
disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=['NREM', 'REM', 'WAKE', 'Art'])
cm_plot = disp.plot(cmap=plt.cm.Blues)
# plt.savefig(os.path.join(save_path, 'cm_whole_model' + '.png'))
plt.show()

a=8