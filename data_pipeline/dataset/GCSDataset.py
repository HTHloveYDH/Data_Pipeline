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

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_objs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        img_data_loader = self.data_objs[index]
        blob = self.bucket.blob(img_data_loader.data)
        blob = blob.download_as_string()  # 
        # blob = blob.decode('utf-8')
        img_data_loader.data = BytesIO(blob)
        image = img_data_loader.load_data()
        # image augmentation and rescale
        image = self.transform(image, **{'random_aug_config': self.random_aug_config})
        return image

    def load_image(self, gcs_blob:str):
        return self.bucket.blob(gcs_blob).download_as_string()