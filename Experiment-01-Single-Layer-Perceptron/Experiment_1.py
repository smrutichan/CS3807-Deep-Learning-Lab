import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import confusion_matrix

#Load the dataset
banknote = fetch_ucirepo(id=267)

X = banknote.data.features
y = banknote.data.targets

df = pd.concat([X, y], axis=1)
df.head()

#Dataset exploring
print("First Five Samples")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistics")
print(df.describe())

#Histograms
colors = ['skyblue', 'lightgreen', 'salmon', 'gold']

axes = df.hist(figsize=(10,8), edgecolor='black')

for ax, color in zip(axes.flatten(), colors):
    for patch in ax.patches:
        patch.set_facecolor(color)

plt.tight_layout()
plt.show()

#Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="viridis")
plt.title("Correlation Heatmap")
plt.show()

#Scatter plot
plt.figure(figsize=(7,5))
sns.scatterplot(data=df,x='variance',y='skewness',hue='class')
plt.show()

#Boxplots
plt.figure(figsize=(10,5))
sns.boxplot(data=df.iloc[:,:4])
plt.show()

#Data preprocessing
X = df.iloc[:,:4].values
y = df.iloc[:,4].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Building perceptron
class Perceptron:

    def __init__(self, lr=0.01, epochs=20):
        self.lr = lr
        self.epochs = epochs

    def activation(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        self.errors = []
        self.weight_history = []
        self.bias_history = []

        for epoch in range(self.epochs):
            error = 0

            for xi, target in zip(X, y):
                z = np.dot(xi, self.weights) + self.bias
                prediction = self.activation(z)
                update = self.lr * (target - prediction)

                self.weights += update * xi
                self.bias += update

                if update != 0:
                    error += 1

            self.errors.append(error)
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)

            print("Epoch:",epoch+1,
                  "Errors:",error,
                  "Weights:",self.weights,
                  "Bias:",self.bias)

    def predict(self,X):
        z = np.dot(X,self.weights)+self.bias
        return self.activation(z)

#Training
model = Perceptron(lr=0.01,epochs=20)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#Metrics
acc = accuracy_score(y_test,y_pred)
pre = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

print("Accuracy :",acc)
print("Precision:",pre)
print("Recall   :",rec)
print("F1 Score :",f1)

#Confusion matrix
cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='viridis')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Training error vs epoch
plt.plot(model.errors,marker='o')

plt.xlabel("Epoch")
plt.ylabel("Misclassified Samples")
plt.title("Training Error vs Epoch")

plt.grid()
plt.show()

#Weight evolution
weights = np.array(model.weight_history)

for i in range(weights.shape[1]):
    plt.plot(weights[:,i],label=f"W{i+1}")

plt.xlabel("Epoch")
plt.ylabel("Weight Value")
plt.title("Weight vs Epoch")

plt.legend()
plt.grid()
plt.show()

#Bias vs epoch
plt.plot(model.bias_history,marker='o')

plt.xlabel("Epoch")
plt.ylabel("Bias")
plt.title("Bias vs Epoch")

plt.grid()
plt.show()

#Comparing learning rates
learning_rates = [0.001,0.01,0.1]

plt.figure(figsize=(8,5))

for lr in learning_rates:
    p = Perceptron(lr=lr,epochs=20)
    p.fit(X_train,y_train)
    plt.plot(p.errors,label=f"LR={lr}")

plt.xlabel("Epoch")
plt.ylabel("Training Errors")

plt.title("Learning Rate Comparison")
plt.legend()
plt.grid()
plt.show()

# Decision Boundary 
from matplotlib.colors import ListedColormap

# Use only two features for visualization
X_db = df[['variance', 'skewness']].values
y_db = df['class'].values

scaler = StandardScaler()
X_db = scaler.fit_transform(X_db)
X_train_db, X_test_db, y_train_db, y_test_db = train_test_split(
    X_db,
    y_db,
    test_size=0.2,
    random_state=42
)

model_db = Perceptron(lr=0.01, epochs=20)
model_db.fit(X_train_db, y_train_db)

x_min, x_max = X_db[:,0].min()-1, X_db[:,0].max()+1
y_min, y_max = X_db[:,1].min()-1, X_db[:,1].max()+1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = model_db.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


cmap = ListedColormap(['royalblue', 'darkorange'])

# Plot
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

plt.scatter(
    X_db[:,0],
    X_db[:,1],
    c=y_db,
    cmap=cmap,
    edgecolors='black',
    s=35
)

plt.xlabel("Variance")
plt.ylabel("Skewness")
plt.title("Decision Boundary of Single Layer Perceptron")

plt.show()
