import firebase_admin
from firebase_admin import credentials, storage

class StorageManager:
    def __init__(self):
        # self.bucket_name = 'pose-estimation-c72f5.appspot.com'
        self.bucket_name = 'dancingprojectdatabase.appspot.com'
        self.fb_cred = "key.json"
        # self.fb_cred = "key.json"
        if not firebase_admin._apps:
            cred = credentials.Certificate(self.fb_cred)
            firebase_admin.initialize_app(cred, {
                'storageBucket': self.bucket_name
            })
    

    #if file is on cloud, prints already exists; otherwise uploads file to cloud
    def upload_file(self, file_name, local_path):
        bucket = storage.bucket()
        blob1 = bucket.blob(file_name)

        if blob1.exists():
            print('This file already exists on cloud.')
            return blob1.public_url
        else:
            outfile = local_path
            blob1.upload_from_filename(outfile)
            with open(outfile, 'rb') as fp:
                blob1.upload_from_file(fp)
            print('This file is uploaded to cloud.')
            blob1.make_public()
            return blob1.public_url
        
        