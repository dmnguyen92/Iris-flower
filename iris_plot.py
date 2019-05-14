############ IMPORTING LIBRARIES AND DATA ###############
print('Loading libraries and data ... ')
from utils import *

plt.style.use('seaborn')
rcParams['font.size'] = 20
rcParams['figure.dpi'] = 150

import warnings
warnings.filterwarnings('ignore')

iris = load_iris()
df = pd.DataFrame(data=iris['data'],columns=iris['feature_names'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

################ DATA PROCESSING AND MAKING MODELS ###################
print('Data processing and making model ... ')

# Feature scaling
scaler = StandardScaler()
df_pca = df.copy()
df_pca[features] = scaler.fit_transform(df[features])

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_pca[features])
df_pca['principal component 1'] = principal_components[:,0]
df_pca['principal component 2'] = principal_components[:,1]

# Making training data
X = df_pca[['principal component 1','principal component 2']]
Y = df_pca['species'].map({'setosa':0, 'versicolor':1, 'virginica':2})

# Making model
clf_knn = KNeighborsClassifier(5)
clf_knn.fit(X.values,Y)


################ MAIN FUNCTION ###################
if __name__ == '__main__':
    resume = True
    while resume == True:
        print('=====================================')
        # Take input
        sepal_length = take_input('SEPAL LENGTH (cm) within range [3,9]: ', 3.,9.)
        sepal_width = take_input('SEPAL WIDTH (cm) within range [1.5,5]: ', 1.5, 5.)
        petal_length = take_input('PETAL LENGTH (cm) within range [0.5, 7.5]: ', 0.5, 7.5)
        petal_width = take_input('PETAL WIDTH (cm) within range [0,3]: ', 0., 3.)
        
        # Transform data
        input_point = np.asarray([sepal_length, sepal_width, petal_length, petal_width]).reshape(1,-1)
        input_point = scaler.transform(input_point)
        input_point = pca.transform(input_point)
        
        # Find points
        top_N, predict_prob = find_closest(input_point, df_pca, clf_knn)
        
        # Plot
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        plot_closest(input_point, df_pca, top_N, predict_prob, ax)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show(block=False)
        
        # Resume
        resume_input = input("Type 'exit' to exit code, any other key to continue: ")
        if resume_input == 'exit':
            resume = False






