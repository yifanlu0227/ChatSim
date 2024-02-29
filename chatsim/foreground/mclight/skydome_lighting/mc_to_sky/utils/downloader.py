import requests
import json
import os
from sys import argv
from urllib.request import URLopener

# arguments
name, resolution, category, ext = argv
# url opener
opener = URLopener()

print("Resolution:", resolution)
print("Category:", category)
print("Extention:", ext)

url = 'https://api.polyhaven.com'
hdris = "/assets?t=hdris"
files = "/files"

# get the url for the hdri json
hdri_url = url + hdris
if category != "all":
    hdri_url = hdri_url + "&c=" + category

# get a list of all the hdri keys
hdris = list(requests.get(hdri_url).json().keys())

# make a dir for resolution
save_to = 'hdri_' + resolution
try:
    os.mkdir(save_to)
except Exception as e:
    pass
os.chdir(save_to)

for hdri in hdris:
    file_json = requests.get(url + files + "/" + hdri).json()

    try:
        print("url:", file_json["hdri"][resolution][ext]["url"])
        with open(f'{hdri}.{ext}','wb') as hdri_file:
            response = requests.get(file_json["hdri"][resolution][ext]["url"], allow_redirects = True)
            hdri_file.write(response.content)
    except Exception as e:
        print("Download failed, possibly because", ext, "is not available for this image.")
        continue

print('Done')