import os
import json
import numpy

import base64
import zlib

import io

from PIL import Image


'''
    dataset source 
    http://rugd.vision
    https://datasetninja.com/rugd
'''
class RUGDAnnLoader:

    def __init__(self, root_path):
        self.ann_files = self._find_files(root_path)
        self.ann_files.sort()

        self.class_name_to_id, self.colors, self.num_classes = self._create_ids()

        print("num_classes ", self.num_classes)
        print("num_items   ", len(self.ann_files))


    def __len__(self):
        return len(self.ann_files)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx):
        result = self._json_to_image(self.ann_files[idx])
        return result

    def get_rgb(self, idx):
        tmp = self.get(idx)

        result_rgb = self.colors[tmp]
        result_rgb = numpy.transpose(result_rgb, (2, 0, 1))

        return result_rgb

    def as_rgb(self, img):
        result_rgb = self.colors[img]
        result_rgb = numpy.transpose(result_rgb, (2, 0, 1))
        return result_rgb

    def _json_to_image(self, file_path):
        f = open(file_path)
        d = json.load(f)
        f.close()   

        height = int(d["size"]["height"])
        width  = int(d["size"]["width"])

        result = numpy.zeros((height, width), dtype=int)

        objects = d["objects"]

        for obj in objects:
            class_name  = obj["classTitle"] 
            ofs_x       = int(obj["bitmap"]["origin"][0])
            ofs_y       = int(obj["bitmap"]["origin"][1])
            bitmap      = obj["bitmap"]["data"]

            # decompress bitmap
            compressed_bytes = base64.b64decode(bitmap)
            png_bytes = zlib.decompress(compressed_bytes)
            
            # load image
            image = Image.open(io.BytesIO(png_bytes))
            image = numpy.array(image)

            img_h = image.shape[0]
            img_w = image.shape[1]

            # convert mask to object ID
            image = self.class_name_to_id[class_name]*image

            # put object mask to correct place
            src = result[ofs_y:ofs_y + img_h, ofs_x:ofs_x + img_w]
            result[ofs_y:ofs_y + img_h, ofs_x:ofs_x + img_w] = numpy.maximum(image, src)

        #result_rgb = self.colors[result]

        return result

    
    def _find_files(self, root_path):
        image_extensions = {'.json', '.Json', '.JSON'}
        image_paths = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                _, ext = os.path.splitext(filename)
                if ext in image_extensions:
                    full_path = os.path.join(dirpath, filename)
                    image_paths.append(full_path)

        return image_paths


    def _create_ids(self):
        
        class_name_to_id = {}

        '''
        class_name_to_id["asphalt"] = 0
        class_name_to_id["bicycle"] = 1
        class_name_to_id["bridge"] = 2
        class_name_to_id["building"] = 3
        class_name_to_id["bush"] = 4
        class_name_to_id["concrete"] = 5
        class_name_to_id["container"] = 6

        class_name_to_id["dirt"] = 7
        class_name_to_id["fence"] = 8
        class_name_to_id["grass"] = 9
        class_name_to_id["gravel"] = 10
        class_name_to_id["log"] = 11
        class_name_to_id["mulch"] = 12
        class_name_to_id["person"] = 13
        class_name_to_id["picnic-table"] = 14
        class_name_to_id["pole"] = 15

        class_name_to_id["rock"] = 16
        class_name_to_id["rock-bed"] = 17
        class_name_to_id["sand"] = 18
        class_name_to_id["sign"] = 19
        class_name_to_id["sky"] = 20
        class_name_to_id["tree"] = 21
        class_name_to_id["vehicle"] = 22
        class_name_to_id["water"] = 23
        '''



        class_name_to_id["asphalt"] = 1
        class_name_to_id["bicycle"] = 0
        class_name_to_id["bridge"] = 0
        class_name_to_id["building"] = 0
        class_name_to_id["bush"] = 0
        class_name_to_id["concrete"] = 1
        class_name_to_id["container"] = 0

        class_name_to_id["dirt"] = 0
        class_name_to_id["fence"] = 0
        class_name_to_id["grass"] = 0
        class_name_to_id["gravel"] = 1
        class_name_to_id["log"] = 0
        class_name_to_id["mulch"] = 0
        class_name_to_id["person"] = 0
        class_name_to_id["picnic-table"] = 0
        class_name_to_id["pole"] = 0

        class_name_to_id["rock"] = 0
        class_name_to_id["rock-bed"] = 0
        class_name_to_id["sand"] = 1
        class_name_to_id["sign"] = 0
        class_name_to_id["sky"] = 0
        class_name_to_id["tree"] = 0
        class_name_to_id["vehicle"] = 0
        class_name_to_id["water"] = 0

        num_classes = 0
        for k in class_name_to_id:
            num_classes = max(num_classes, class_name_to_id[k])

        num_classes = num_classes + 1
       
        colors = numpy.zeros((num_classes, 3), dtype=numpy.float32)

        for n in range(num_classes):
            if n != 0:
                phi = n*2.0*numpy.pi/(num_classes-1)
                colors[n][0] = (numpy.cos(phi + 2.0*numpy.pi*0.0/3.0) + 1.0)/2.0
                colors[n][1] = (numpy.cos(phi + 2.0*numpy.pi*1.0/3.0) + 1.0)/2.0
                colors[n][2] = (numpy.cos(phi + 2.0*numpy.pi*2.0/3.0) + 1.0)/2.0


        return class_name_to_id, colors, num_classes

        