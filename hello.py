from flask import Flask, send_file, make_response, render_template, request, jsonify
from flask_restful import Api, Resource, reqparse
from PIL import Image
import werkzeug
import os, io, sys
import numpy as np
from cv2 import cv2
import base64

import imageprocessor as improc

app = Flask(__name__)
api = Api(app)
imageprocessor = improc.ImageProcessor()

def bgr2rgb_2_params(function):
    def wrapper(*args):
        image, error = function(*args)
        image, _ = imageprocessor.bgr2rgb(image)
        return image, error
    return wrapper

def bgr2rgb_3_params(function):
    def wrapper(*args):
        image, error, values = function(*args)
        image, _ = imageprocessor.bgr2rgb(image)
        return image, error, values
    return wrapper

def get_image(filename):
    parse = reqparse.RequestParser()
    parse.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files')
    args = parse.parse_args()
    image_file = args['image']
    image_file.save(filename)

def image_process(file, method, params = None):
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, flags=1)

    if params:
        img = method(img, *params)[0]
    else:
        img = method(img)[0]
        
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status':str(img_base64)})

class LoadImage(Resource):
    def get(self):
        file_path = os.path.join(app.root_path, "newImage.jpg")
        return_data = io.BytesIO()
        with open(file_path, 'rb') as fo:
            return_data.write(fo.read())
        return_data.seek(0)
        os.remove(file_path)
        return send_file(return_data, mimetype='image/jpg', attachment_filename='image.jpg')

class HomePage(Resource):
    def get(self):
        return make_response(render_template('index.html'))

class Grayscale(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Grayscale", 
            endpoint = "/gray", 
            var1="none",
            var2="none"))
    def post(self):   
        return image_process(request.files['image'].read(), imageprocessor.to_grayscale)

class Median(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Median", 
            endpoint = "/median", 
            var1="block",
            var1Name="ksize", 
            var2="none"))
    def post(self):
        print(request.files)
        return image_process(request.files['image'].read(), bgr2rgb_2_params(imageprocessor.median_blur), [int(request.form['ksize'])])

class Average(Resource):
    def post(self, ksize_x, ksize_y):
        result = image_process(request.files['image'].read(), imageprocessor.average_blur, [ksize_x, ksize_y])

class GaussianBlur(Resource):
    def post(self, ksize_x, ksize_y):
        get_image("newImage.jpg")
        image, err = imageprocessor.read_image("newImage.jpg")
        image, err = imageprocessor.gaussian_blur(image, ksize_x, ksize_y)
        imageprocessor.write_image(image, "newImage.jpg")
        return {'id': "newImage", 'error': err}

class Bilateral(Resource):
    def post(self, d, sigma):
        get_image("newImage.jpg")
        image, err = imageprocessor.read_image("newImage.jpg")
        image, err = imageprocessor.bilateral_filter(image, d, sigma)
        imageprocessor.write_image(image, "newImage.jpg")
        return {'id': "newImage", 'error': err}

class GlobalThresh(Resource):
    def post(self, threshold, value):
        get_image("newImage.jpg")
        image, err = imageprocessor.read_image("newImage.jpg")
        image, err = imageprocessor.global_threshold(image, threshold, value)
        imageprocessor.write_image(image, "newImage.jpg")
        return {'id': "newImage", 'error': err}

class MeanThreshold(Resource):
    def post(self, blocksize, c, value):
        get_image("newImage.jpg")
        image, err = imageprocessor.read_image("newImage.jpg")
        image, err = imageprocessor.mean_threshold(image, blocksize, c, value)
        imageprocessor.write_image(image, "newImage.jpg")
        return {'id': "newImage", 'error': err}

class GaussianThreshold(Resource):
    def post(self, blocksize, c, value):
        get_image("newImage.jpg")
        image, err = imageprocessor.read_image("newImage.jpg")
        image, err = imageprocessor.gaussian_threshold(image, blocksize, c, value)
        imageprocessor.write_image(image, "newImage.jpg")
        return {'id': "newImage", 'error': err}

class Sobel(Resource):
    def post(self, dx, dy, ksize, delta):
        get_image("newImage.jpg")
        image, err = imageprocessor.read_image("newImage.jpg")
        image, err = imageprocessor.sobel(image, dx, dy, ksize, delta)
        imageprocessor.write_image(image, "newImage.jpg")
        return {'id': "newImage", 'error': err}

class Laplacian(Resource):
    def post(self, ksize, delta):
        get_image("newImage.jpg")
        image, err = imageprocessor.read_image("newImage.jpg")
        image, err = imageprocessor.laplacian(image, ksize, delta)
        imageprocessor.write_image(image, "newImage.jpg")
        return {'id': "newImage", 'error': err}

class Canny(Resource):
    def post(self, threshold1, threshold2):
        get_image("newImage.jpg")
        image, err = imageprocessor.read_image("newImage.jpg")
        image, err = imageprocessor.canny_edge_detection(image, threshold1, threshold2)
        imageprocessor.write_image(image, "newImage.jpg")
        return {'id': "newImage", 'error': err}

class FrontalFace(Resource):
    def post(self, min_neighbours, scale):
        get_image("newImage.jpg")
        image, err = imageprocessor.read_image("newImage.jpg")
        image, err = imageprocessor.haar_frontal_face_detection(image, min_neighbours, scale)
        imageprocessor.write_image(image, "newImage.jpg")
        return {'id': "newImage", 'error': err}


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

api.add_resource(HomePage, "/")
api.add_resource(LoadImage, "/image")
api.add_resource(Grayscale, "/gray")
api.add_resource(Median, "/median")
api.add_resource(Average, "/average/<int:ksize_x>/<int:ksize_y>")
api.add_resource(GaussianBlur, "/gauss/<int:ksize_x>/<int:ksize_y>")
api.add_resource(Bilateral, "/bilateral/<int:d>/<float:sigma>")
api.add_resource(GlobalThresh, "/thresh/global/<int:threshold>/<int:value>")
api.add_resource(MeanThreshold, "/thresh/mean/<int:blocksize>/<float:c>/<int:value>")
api.add_resource(GaussianThreshold, "/thresh/gauss/<int:blocksize>/<float:c>/<int:value>")
api.add_resource(Sobel, "/sobel/<int:dx>/<int:dy>/<int:ksize>/<float:delta>")
api.add_resource(Laplacian,"/laplacian/<int:ksize>/<float:delta>")
api.add_resource(Canny, "/canny/<int:threshold1>/<int:threshold2>")
api.add_resource(FrontalFace, "/frontal/<int:min_neighbours>/<float:scale>")
if __name__ == "__main__":
    app.run(debug=True)
    