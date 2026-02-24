### QUESTION 1 ###
import numpy as np
from PIL import Image
import os

### QUESTION 2 ###
from sklearn.preprocessing import StandardScaler

### QUESTION 3 and QUESTION 4 ###
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import f1_score, precision_score, recall_score


### QUESTION 5 ###
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import ttest_rel
import numpy as np

#### QUESTION 6 ###
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import numpy as np

# from joblib import Parallel, delayed

# Please write the optimal hyperparameter values you obtain in the global variable 'optimal_hyperparm' below. This
# variable should contain the values when I look at your submission. I should not have to run your code to populate this
# variable.

optimal_hyperparam = {
    'hidden_layer_sizes': (300, 150),
    'alpha': 0.1,
    'learning_rate_init': 0.0001,
    'activation': 'relu',
    'solver': 'adam',
    'batch_size': 'auto',
}

class COC131:
    def __init__(self):
        """
        This function should be used to initialize the class. You can use this function to set up any instance variables
        you need.
        """
        self.x = None
        self.y = None
        self.embedding = LocallyLinearEmbedding()

        self.unique_classes = np.unique(self.y)
        self.scaler = None
        self.label_encoder = None

        self.best_weights = None
        self.best_epoch = None
        self.best_test_acc = None

        self.confusion_matrix = None
        self.confusion_matrix_display = None

        self.best_embedding = None
        self.best_neighbors = None

        self.model = None


    def q1(self, filename=None):
        """
        This function should be used to load the data. To speed-up processing in later steps, lower resolution of the
        image to 32*32. The folder names in the root directory of the dataset are the class names. After loading the
        dataset, you should save it into an instance variable self.x (for samples) and self.y (for labels). Both self.x
        and self.y should be numpy arrays of dtype float.

        :param filename: this is the name of an actual random image in the dataset. You don't need this to load the
        dataset. This is used by me for testing your implementation.
        :return res1: a one-dimensional numpy array containing the flattened low-resolution image in file 'filename'.
        Flatten the image in the row major order. The dtype for the array should be float.
        :return res2: a string containing the class name for the image in file 'filename'. This string should be same as
        one of the folder names in the originally shared dataset.
        """
        image_size = (32, 32)
        dataset_dir = "../dataset"  # path to the dataset folder

        # arrays to store the data
        x_data = []
        y_data = []

        # read the dataset folder-by-folder
        for class_name in os.listdir(dataset_dir):
            class_dir = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        img = Image.open(img_path).resize(image_size)
                        flat_img = np.array(img, dtype=float).flatten()
                        x_data.append(flat_img)
                        y_data.append(class_name)
                    except:
                        continue  # skip broken or unreadable images

        # convert lists to numpy arrays
        self.x = np.array(x_data, dtype=float)  # 2D array (samples × features)
        self.y = np.array(y_data, dtype=str)

        print(f"Loaded {len(self.x)} samples, {len(set(self.y))} unique classes.")
        print(f"Sample shape: {self.x[0].shape if len(self.x) > 0 else 'No samples'}\n")

        # return the test image flatten and class name
        image = np.zeros(1)
        label = ''

        if filename:
            found = False
            for class_name in os.listdir(dataset_dir):
                test_path = os.path.join(dataset_dir, class_name, filename)
                if os.path.exists(test_path):
                    found = True
                    test_img = Image.open(test_path).resize(image_size)
                    image = np.array(test_img, dtype=float).flatten()
                    label = class_name
                    break
            if not found:
                raise FileNotFoundError(f"Image '{filename}' not found in the dataset.")

        return image, label

    def q2(self, inp):
        """
        This function should compute the standardized data from a given 'inp' data. The function should work for a
        dataset with any number of features.

        :param inp: an array from which the standardized data is to be computed.
        :return res2: a numpy array containing the standardized data with standard deviation of 2.5. The array should
        have the same dimensions as the original data
        :return res1: sklearn object used for standardization.
        """
        scaler = StandardScaler()
        standardized = scaler.fit_transform(inp)

        # Rescale to target std = 2.5
        scaled = standardized * 2.5

        return scaled, scaler

    def q3(self, test_size=None, pre_split_data=None, hyperparam=None):
        """
        This function should build a MLP Classifier using the dataset loaded in function 'q1' and evaluate model
        performance. You can assume that the function 'q1' has been called prior to calling this function. This function
        should support hyperparameter optimizations.

        :param test_size: the proportion of the dataset that should be reserved for testing. This should be a fraction
        between 0 and 1.
        :param pre_split_data: Can be used to provide data already split into training and testing.
        :param hyperparam: hyperparameter values to be tested during hyperparameter optimization.
        
        :return: The function should return 1 model object and 3 numpy arrays which contain the loss, training accuracy
        and testing accuracy after each training iteration for the best model you found.
        """
        x, y = self.preprocess()
        
        # Split or unpack existing split
        if pre_split_data is None:
            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size or 0.3, stratify=y, random_state=42, shuffle=True)
        else:
            X_train, X_test, y_train, y_test = pre_split_data

        # default hyperparameters
        if hyperparam is None:
            hyperparam = optimal_hyperparam

        if not isinstance(hyperparam, dict):
            raise ValueError("hyperparam must be a dictionary.")

        model, losses, train_scores, test_scores = self.train_and_track(
            hyperparam, epochs=300, early_stopping_rounds=30
        )

        # create a confusion matrix
        # 1) Get the predictions on your held‐out test set
        y_pred = model.predict(X_test)

        # 2) Compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
    

        # save the confusion matrix to the instance variable
        self.confusion_matrix = cm
        self.confusion_matrix_display = disp

        # save the model to the instance variable
        self.model = model

        # print results
        print(f"model parameters: {hyperparam}\n")

        print(f"final test set score: {model.score(X_test, y_test)}")
        print(f"Final train set score: {model.score(X_train, y_train)}")

        return model, np.array(losses), np.array(train_scores), np.array(test_scores)

    def q4(self):
        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: res should be the data you visualized.
        """
        #set the base hyperparameters to be the optimal hyperparmeters
        # from q3
        hyperparam = optimal_hyperparam
        
        alpha_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
       

        #preprocess data
        x, y = self.preprocess()

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, stratify=y, random_state=42
        )

        accuracies   = []
        final_train_scores = []
        weight_norms = []
        bias_norms   = []
        f1_scores    = []
        precisions   = []
        recalls      = []

        for alpha in alpha_values:
            hyperparam['alpha'] = alpha
            print(f"Training with alpha: {alpha}")
            clf, _, train_scores, test_scores = self.train_and_track(hyperparam, epochs=20)
            y_pred = clf.predict(X_test)

            accuracies.append(test_scores[-1])
            final_train_scores.append(train_scores[-1])

            weight_norms.append(sum(np.linalg.norm(w) for w in clf.coefs_))
            bias_norms.append( sum(np.linalg.norm(b) for b in clf.intercepts_))


            f1_scores.append(f1_score(y_test, y_pred, average='macro'))
            precisions.append(precision_score(y_test, y_pred, average='macro'))
            recalls.append(recall_score(y_test, y_pred, average='macro'))


        res = {
            'alpha':           np.array(alpha_values),
            'accuracy':        np.array(accuracies),
            'train_accuracy':  np.array(final_train_scores),
            'weight_norm':     np.array(weight_norms),
            'bias_norm':       np.array(bias_norms),
            'macro_f1':        np.array(f1_scores),
            'macro_precision': np.array(precisions),
            'macro_recall':    np.array(recalls),
        }

        return res

    def q5(self):
        """
        This function should perform hypothesis testing to study the impact of using CV with and without Stratification
        on the performance of MLPClassifier. Set other model hyperparameters to the best values obtained in the previous
        questions. Use 5-fold cross validation for this question. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: The function should return 4 items - the final testing accuracy for both methods of CV, p-value of the
        test and a string representing the result of hypothesis testing. The string can have only two possible values -
        'Splitting method impacted performance' and 'Splitting method had no effect'.
        """
        

        # Get best hyperparameters
        hyperparam = optimal_hyperparam
        hyperparam['max_iter'] = 20
        
        # Preprocess data
        X, y = self.preprocess()
        
        n_splits = 5

        # Stratified and non-stratified KFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        strat_scores = []
        non_strat_scores = []


        classes = np.unique(y)
        for strat_idx, non_strat_idx in zip(skf.split(X, y), kf.split(X)):
            # Stratified fold
            train_idx, test_idx = strat_idx
            model = self.get_model(hyperparam)
            model.fit(X[train_idx], y[train_idx])
            strat_scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))

            # Non-stratified fold
            train_idx, test_idx = non_strat_idx
            model = self.get_model(hyperparam)
            model.fit(X[train_idx], y[train_idx])
            non_strat_scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))


        # Hypothesis test
        _, p_val = ttest_rel(strat_scores, non_strat_scores)

        conclusion = (
            "Splitting method impacted performance"
            if p_val < 0.05 else
            "Splitting method had no effect"
        )

        return strat_scores[-1], non_strat_scores[-1], p_val, conclusion

    def q6(self):
        """
        This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
        the function 'q1' has been called prior to calling this function.

        :return: The function should return the data you visualize.
        """
        # Preprocess data
        x, y = self.preprocess()

        # Convert labels to numeric
        classes = np.unique(y)
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        y_numeric = np.array([class_to_idx[label] for label in y])

        # Choose neighbor values from ~0.2% to ~1.5% of dataset
        neighbor_vals = [10, 50, 75, 100, 150, 200]
        best_score = -1
        best_embedding = None
        best_neighbors = None

        #save all the results
        results = []

        # Loop to find best silhouette score
        for n_neighbors in neighbor_vals:
            lle = LocallyLinearEmbedding(n_components=2, n_neighbors=n_neighbors, random_state=42)
            embedded = lle.fit_transform(x)
            score = silhouette_score(embedded, y_numeric)

            results.append({"n_neighbors": n_neighbors, "score": score, "embedding": embedded})

        ### THIS WAS WHAT WAS ACTUALLY USED FOR QUESTION 6 ### - commented out for coursework submission 
        # # Parallelize the LLE fitting and silhouette score calculation
        # from joblib import Parallel, delayed

        # # Function to run LLE + silhouette score for one value
        # def run_lle(n_neighbors):
        #     lle = LocallyLinearEmbedding(n_components=2, n_neighbors=n_neighbors, random_state=42)
        #     embedded = lle.fit_transform(x)
        #     score = silhouette_score(embedded, y_numeric)
        #     return {"n_neighbors": n_neighbors, "score": score, "embedding": embedded}

        # # Run in parallel
        # results = Parallel(n_jobs=-1)(delayed(run_lle)(k) for k in neighbor_vals)
        ### End of parallelization

        # Find best result
        best_result = max(results, key=lambda r: r["score"])
        best_embedding = best_result["embedding"]
        best_neighbors = best_result["n_neighbors"]
        best_score = best_result["score"]
    
        print(f"\nBest n_neighbors = {best_neighbors} with silhouette score = {best_score:.3f}")
        # Save the best embedding
        self.best_embedding = best_embedding

        return results
    
    def train_and_track(self, hyperparam, epochs=10, early_stopping_rounds=15):
        """
        Train a model for specified number of epochs and track performance metrics.
        
        Parameters:
            hyperparameters : dict
                Hyperparameters for the model
            epochs : int, default=10
                Number of epochs to train
            early_stopping_rounds : int, default=15
                Number of epochs with no improvement after which training will be stopped
            classes : array-like, default=None
                Unique classes for partial_fit
            
        Returns:
            model, losses, train_scores, test_scores
        """
        # preprocess data for training then split it
        x, y = self.preprocess()
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, stratify=y, random_state=42
        )

        classes = np.unique(y)

        # get the model with the hyperparameters
        model = self.get_model(hyperparam)
        model.set_params(warm_start=True)


        best_test_acc = -np.inf
        best_weights = None
        best_epoch = 0
        wait = 0

        losses, train_scores, test_scores = [], [], []

        for epoch in range(epochs):
            model.partial_fit(X_train, y_train, classes=classes)

            loss = model.loss_
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)

            losses.append(loss)
            train_scores.append(train_acc)
            test_scores.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_weights = model.coefs_, model.intercepts_
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= early_stopping_rounds:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")

        # Save the best weights and metadata as instance variables
        self.best_weights = best_weights
        self.best_epoch = best_epoch
        self.best_test_acc = best_test_acc

        # reset the model to the best weights
        model.coefs_, model.intercepts_ = best_weights

        return model, losses, train_scores, test_scores
    
    def preprocess(self, standardize=True, encode_labels=True):
        """
        Preprocess the dataset by standardizing features and encoding labels.
        
        Arguments:
            standardize : bool, default=True
                Whether to standardize features
            encode_labels : bool, default=True
                Whether to encode string labels to integers
        
        Returns:
            X : numpy array
                Preprocessed features
            y : numpy array
                Preprocessed labels
        """
        X, y = self.x, self.y
        
        # Standardize features
        if standardize and X is not None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        # Encode string labels to integers
        if encode_labels and y is not None:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            y = y.astype(int)
        
        return X, y
    
    def get_model(self, hyperparam=None):
        """
        Create and return an MLPClassifier with the specified hyperparameters.
        
        Arguments:
            hyperparam : dict, default=None
                Hyperparameters for the model. If None, use optimal_hyperparam.
            
        Returns:
            model : MLPClassifier
                The created MLPClassifier model
        """
        if hyperparam is None:
            hyperparam = optimal_hyperparam
        
        model = MLPClassifier(
            **hyperparam,
            random_state=42,
            )
        
        return model
    
    def tune_hyperparameter(self, values, param_name, current_params):
        """
        Tunes a specific hyperparameter by trying different values and selecting the best one.
        
        This method evaluates multiple values for a given hyperparameter while keeping other
        parameters fixed. It trains and evaluates the model for each value, tracks the performance,
        and determines the optimal value based on test scores.
        
        Arguements:
            values (list): List of values to try for the hyperparameter.
            param_name (str): Name of the hyperparameter to tune.
            current_params (dict): Dictionary of current hyperparameter settings.
            
        Returns:
            values, scores, best_value, current_params
                
        Note:
            This method prints the test score for each value tried and the final best value.
        """
        final_scores = []
        training_values = []
        testing_values = []
        loss_values = []
        

        for value in values:
            current_params[param_name] = value
            # Train and evaluate with current_params
            _, losses, train_scores, test_scores = self.train_and_track(hyperparam=current_params)
            # Store the list of scores
            loss_values.append(losses)
            training_values.append(train_scores)
            testing_values.append(test_scores)
            final_scores.append(test_scores[-1])
            print(f"{param_name}: {value}, Test Score: {test_scores[-1]}")

        # Find the value with the best score
        best_value = values[np.argmax(final_scores)]
        print(f"Best {param_name}: {best_value}")

        # Update the current_params with the best value
        current_params[param_name] = best_value

        return values, final_scores, best_value, current_params, training_values, testing_values, loss_values