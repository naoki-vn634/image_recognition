import sys
import os
from urllib import request
from PIL import Image

import argparse


def download(url, decode=False):
    response = request.urlopen(url)
    if response.geturl() == "https://s.yimg.com/pw/images/en-us/photo_unavailable.png":
        # Flickr :This photo is no longer available iamge.
        raise Exception("This photo is no longer available iamge.")

    body = response.read()
    if decode == True:
        body = body.decode()
    return body

def write(path, img):
    file = open(path, 'wb')
    file.write(img)
    file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",type=str)
    args = parser.parse_args() 
        

    # see http://image-net.org/archive/words.txt
    classes = {"airliner" : "n02690373", 	"warplane" : "n04552348"}

    offset = 0
    max = 2000
    for dir, id in classes.items():
        save_dir = os.path.join(args.root_dir,dir)
        os.makedirs(save_dir,exist_ok=True)
        print(save_dir)

        urls = download("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="+id, decode=True).split()
        print(urls)
        print(len(urls))
        i = 0
        for url in urls:
            print("######URL:",url)
            
            
            # if response.geturl() == "https://s.yimg.com/pw/images/en-us/photo_unavailable.png":
            #     # Flickr :This photo is no longer available iamge.
            #     raise Exception("This photo is no longer available iamge.")
            
            if i < offset:
                continue
            if i > max:
                break

            try:
                response = request.urlopen(url)
                file = os.path.split(url)[1]
                path = save_dir + "/" + dir + "/" + file
                write(path, download(url))
                print("done:" + str(i) + ":" + file)
            except:
                print("EEEEEEEEEEEEEEEEEEEE")
                print("Unexpected error:", sys.exc_info()[0])
                print("error:" + str(i) + ":" + file)
            i = i + 1

    print("end")

if __name__ == "__main__":
    main()