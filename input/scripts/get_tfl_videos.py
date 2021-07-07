import requests
import xmltodict

print("Start fetching videos from TfL Live Traffic Cameras: ")

# URL of the XML file containing the locations of the videos (and their thumbnail) where they can be found.
url = "http://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/"

response = requests.get(url)
data = xmltodict.parse(response.content)
# Contents is a list of objects, ordered by file name. It follows the following pattern:
# Object for thumbnail 00001.0xxx1.jpg
# Object for video 00001.0xxx1.mp4
# Object for thumbnail 00001.0xxx2.jpg
# Object for video 00001.0xxx2.mp4
contents = data["ListBucketResult"]["Contents"]

# Number of videos to fetch:
N = 500
counter = 0
# Skip every 2 items to only fetch videos (and not thumbnails).
for index in range(1, len(contents), 2):
    # Get the file name of the video.
    video = contents[index]["Key"]
    video_url = url + video
    response = requests.get(video_url)
    if response.status_code == 404:
        print("Can't find video: " + video)
    else:
        with open(video, 'wb') as f:
            f.write(response.content)
            print("Downloaded video: " + video)
            counter += 1
    if counter >= N:
        break
