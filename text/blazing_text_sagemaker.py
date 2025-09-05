import sagemaker
from sagemaker import Session
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker import get_execution_role


session = Session()
role = get_execution_role()

image = sagemaker.image_uris.retrieve(
    'blazingtext',
    session.boto_region_name
)

bt_estimator = sagemaker.estimator.Estimator(
    image,
    role,
    instance_count=1,
    instance_type='ml.c4.xlarge',
    output_path='s3://amazon-sagemaker-111/dev/blazingtext/',
    sagemaker_session=session,
)

bt_estimator.set_hyperparameters(
    mode='batch_skipgram',
    epochs=10,
    min_count=2,
    learning_rate=0.05,
    window_size=5,
    vector_dim=100,
    negative_samples=5,
    batch_size=11,
    evaluation=True,
    subwords=False,
)

input = TrainingInput(
    's3://amazon-sagemaker-111/dev/blazingtext/dataset.txt', 
    content_type='text/plain'
)
bt_estimator.fit({"train": input}, logs=True)
