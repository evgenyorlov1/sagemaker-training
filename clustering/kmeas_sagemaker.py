import numpy as np
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import CSVSerializer
from sklearn.preprocessing import StandardScaler


session = sagemaker.Session()
role = get_execution_role()

# setup model
image = sagemaker.image_uris.retrieve(
    'kmeans',
    session.boto_region_name,
)
kmeans = sagemaker.estimator.Estimator(
    image,
    role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path='s3://../',
    sagemaker_session=session
)
kmeans.set_hyperparameters(k=3, feature_dim=4)

# train model
train_input = TrainingInput(
    's3://../iris.CSV', 
    content_type='text/csv;label_size=0'
)
kmeans.fit({'train': train_input})

# deploy model
predictor = kmeans.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge',
)
predictor.serializer = CSVSerializer()

test_data = np.array(
    [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3],
        [5.9, 3.0, 5.1, 1.8],
        [4.7, 3.2, 1.3, 0.2]
    ]
)
scaler = StandardScaler()
test_data = scaler.fit_transform(test_data)

predicted_clusters = predictor.predict(test_data)
print(predicted_clusters)