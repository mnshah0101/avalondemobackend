import datetime
import boto3


def getFileName(filepath: str):
    return filepath.split('/')[-1]


def upload_to_s3(docs: list, bucket_name: str):
    """
    @param docs - an list of local file paths
    @param - bucket_name string of the bucket name
    """
    try:
        s3 = boto3.client('s3')
        s3.create_bucket(Bucket=bucket_name)

        s3_keys = []
        for doc in docs:
            try:
                date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

                # add the date to the filename before .pdf but after the original filename
                s3_key = f"{doc.filename[:-4]}_{date}.pdf"
                s3_keys.append(s3_key)

                s3.upload_fileobj(doc, bucket_name, s3_key)
            except Exception as e:
                print(f"Error : {e}")

        print("Successfully uploaded documents to s3")
        return s3_keys
    except Exception as e:
        print(f"Error : {e}")


def delete_from_s3(docs: list, bucket_name: str):
    """
    @param docs - an list of local file paths
    @param - bucket_name string of the bucket name
    """
    try:
        s3 = boto3.client('s3')

        for doc in docs:
            try:
                print(f"Deleting {doc}")
                s3_key = doc
                s3.delete_object(Bucket=bucket_name, Key=s3_key)
            except Exception as e:
                print(f"Error : {e}")

        print("Successfully deleted documents from s3")
    except Exception as e:
        print(f"Error : {e}")
