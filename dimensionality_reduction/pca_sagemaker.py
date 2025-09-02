import sagemaker
from sagemaker import Session
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer


session = Session()
role = get_execution_role()

# pca image
pca_image = sagemaker.image_uris.retrieve(
    'pca',
    session.boto_region_name,
)

# pca estimator
pca = sagemaker.estimator.Estimator(
    image_uri=pca_image,
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path='s3://amazon-sagemaker-345584767416-us-east-1-c2d0c52ee081/dzd_azs4tevmaqpraf/5wfw2wp2xb3pd3/dev/pca/'
)
pca.set_hyperparameters(
    feature_dim=4,
    num_components=3,
    subtract_mean=True,
    algorithm_mode='regular',
    mini_batch_size=100,
)

# train
train_input = TrainingInput(
    's3://.../dev/pca/iris.CSV', 
    content_type='text/csv;label_size=0'
)
pca.fit({'train': train_input})

# deploy model
predictor = pca.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
)
predictor.serializer = CSVSerializer()
predictor.deserializer = CSVDeserializer()

# test
print(predictor.predict([5.1, 3.5, 1.4, 0.2]))
