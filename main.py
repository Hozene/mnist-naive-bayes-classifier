from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np


class MNISTClassifier:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self._load_data()
        self._log_data_info()
        self.num_classes = len(np.unique(self.y))

    def _load_data(self):
        mnist = fetch_openml('mnist_784', version=1)
        self.X = mnist.data
        self.y = mnist.target.astype(int)
        self.images = self.X.to_numpy().reshape(-1, 28, 28)
        self.labels = self.y

    def _log_data_info(self):
        num_samples = self.X.shape[0]
        num_features = self.X.shape[1]
        log_info = f"Number of samples: {num_samples}\nNumber of features: {num_features}\nData structure: {self.X.shape}\n"
        with open('mnist_log.txt', 'w') as log_file:
            log_file.write(log_info)

    def show_sample_images(self, num_samples=5):
        plt.figure(figsize=(10, 4))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(self.images[i], cmap='gray')
            plt.title(f"Label: {self.labels[i]}")
            plt.axis('off')
        plt.show()

    def train_model(self):
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        self.gnb = GaussianNB()
        self.gnb.fit(X_train, y_train)

    def evaluate_model(self):
        self.y_pred = self.gnb.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.error_rates = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if np.sum(self.y_test == i) > 0:
                self.error_rates[i] = 1 - np.mean(self.y_pred[self.y_test == i] == i)
            else:
                self.error_rates[i] = np.nan
        self.report = classification_report(self.y_test, self.y_pred)

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    def results(self):
        print(f"\nOverall accuracy: {self.accuracy:.4f}")
        print("Error rates per class:")
        for i in range(self.num_classes):
            print(f"Class {i}: {self.error_rates[i]:.4f}")
        print("\nClassification Report:\n")
        print(self.report)


if __name__ == "__main__":
    mnist_classifier = MNISTClassifier()
    mnist_classifier.show_sample_images()
    mnist_classifier.train_model()
    mnist_classifier.evaluate_model()
    mnist_classifier.plot_confusion_matrix()
    mnist_classifier.results()
