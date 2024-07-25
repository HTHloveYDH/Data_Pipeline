from io import BytesIO

from google.cloud import storage

from data_pipeline.dataset.BaseDataset import BaseDataset
import global_vars_manager


class GCSDataset(BaseDataset):
    'Characterizes a gcs dataset for PyTorch'
    def __init__(self, data_objs:list, transform, **kwargs):
        super(GCSDataset, self).__init__(data_objs, transform, **kwargs)
        gcs = storage.Client()
        self.gcs_bucket_name = global_vars_manager.get_global_var('GCS_BUCKET_NAME')
        self.bucket = gcs.get_bucket(self.gcs_bucket_name)

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        img_data_loader = self.data_objs[index]
        blob = self.bucket.blob(img_data_loader.key)
        image_bytes = blob.download_as_string()  # type: bytes
        assert isinstance(image_bytes, bytes)
        # image_bytes = image_bytes.decode('utf-8')
        data = BytesIO(image_bytes)  # type: BytesIO
        image = img_data_loader.load_data(data)
        # image augmentation and rescale
        image = self.transform(image, **self.custom_load_config)
        return image

    def load_image(self, gcs_blob:str):
        return self.bucket.blob(gcs_blob).download_as_string()