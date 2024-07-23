def filename2loc(filename:str):
    if 's3://' in filename:
        return 's3'
    elif 'storage.googleapis.com' in filename:
        return 'gcs'
    else:
        return 'local_disk'