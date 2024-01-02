# K-Nearest Classifier 
## Creating a Knn classifier
# Start
## importing libraries
import numpy as np 
from collections import Counter
# test data 
points = {
    "blue" : [[2,5] , [7,9]],
    "red" : [[1,4] , [6,2]]
}
new_point = [3,0]
#### creating a function to calculate the euclidean distance.
def euclidean(observed_values , actual_values):
    calculated_distances = np.sqrt(np.sum((np.array(observed_values) - np.array(actual_values)) ** 2))
    return calculated_distances
### Creating class for the Knn classifier
class KNearestClassifier:
    # constructor
    def __init__(self , k = 5):
        self.k = k
        self.points = None
    # creating fit function to feed the training data into the model.
    def fit(self , points):
        self.points = points
    # creating the predict function.
    def predict(self , new_point):
        # To keep track of all the distances
        distances = []
        # Creating a loop for our X_test
        for category in self.points:
            for point in self.points[category]:
                distance = euclidean(point , new_point) 
                distances.append([distance , category])
        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result
    
clf = KNearestClassifier()
clf.fit(points)
p = clf.predict(new_point)
print("Prediction : " , p)
