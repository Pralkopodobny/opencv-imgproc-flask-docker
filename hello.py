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
        res = method(img, *params)
    else:
        res = method(img)
    
    if(res[1]):
        return {'success' : False, 'err' : res[1]}
        
    img = Image.fromarray(res[0].astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'success' : True, 'status':str(img_base64)})

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
            var2="none",
            var3="none",
            var4="none"))
    def post(self):   
        return image_process(request.files['image'].read(), imageprocessor.to_grayscale)

class Median(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Median", 
            endpoint = "/median", 
            var1="block", var1Name="ksize", 
            var2="none",
            var3="none",
            var4="none"))
    def post(self):
        return image_process(request.files['image'].read(), improc.median_blur, [int(request.form['ksize'])])

class Average(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Average", 
            endpoint = "/average", 
            var1="block", var1Name="ksize_x", 
            var2="block", var2Name="ksize_y",
            var3="none",
            var4="none"))
    def post(self):
        return image_process(request.files['image'].read(), improc.average_blur, [int(request.form['ksize_x']), int(request.form['ksize_y'])])

class GaussianBlur(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Gaussian Blur", 
            endpoint = "/gauss", 
            var1="block", var1Name="ksize_x", 
            var2="block", var2Name="ksize_y",
            var3="none",
            var4="none"))
    def post(self):
        return image_process(request.files['image'].read(), improc.gaussian_blur, [int(request.form['ksize_x']), int(request.form['ksize_y'])])

class Bilateral(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Bilateral filter", 
            endpoint = "/bilateral", 
            var1="block",  var1Name="d", 
            var2="block", var2Name="sigma",
            var3="none",
            var4="none"))
    def post(self):
        return image_process(request.files['image'].read(), improc.bilateral_filter, [int(request.form['d']), float(request.form['sigma'])])

class GlobalThresh(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Global threshold", 
            endpoint = "/thresh/global", 
            var1="block", var1Name="threshold", 
            var2="block", var2Name="value",
            var3="none",
            var4="none"))
    def post(self):
        return image_process(request.files['image'].read(), improc.global_threshold, [int(request.form['threshold']), int(request.form['value'])])

class MeanThreshold(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Mean threshold", 
            endpoint = "/thresh/mean", 
            var1="block", var1Name="blocksize", 
            var2="block", var2Name="c",
            var3="block", var3Name="value",
            var4="none"))
    def post(self):
        return image_process(request.files['image'].read(), improc.mean_threshold, [int(request.form['blocksize']), float(request.form['c']), int(request.form['value'])])

class GaussianThreshold(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Gaussian threshold", 
            endpoint = "/thresh/gauss", 
            var1="block", var1Name="blocksize", 
            var2="block", var2Name="c",
            var3="block", var3Name="value",
            var4="none"))
    def post(self):
        return image_process(request.files['image'].read(), improc.gaussian_threshold, [int(request.form['blocksize']), float(request.form['c']), int(request.form['value'])])

class Sobel(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Sobel", 
            endpoint = "/sobel", 
            var1="block", var1Name="dx", 
            var2="block", var2Name="dy",
            var3="block", var3Name="ksize",
            var4="block", var4Name="delta"))
    def post(self):
        return image_process(request.files['image'].read(), improc.sobel, [int(request.form['dx']), float(request.form['dy']), int(request.form['ksize']), float(request.form['delta'])])

class Laplacian(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Global threshold", 
            endpoint = "/laplacian", 
            var1="block", var1Name="ksize", 
            var2="block", var2Name="delta",
            var3="none",
            var4="none"))
    def post(self):
        return image_process(request.files['image'].read(), improc.laplacian, [int(request.form['ksize']), float(request.form['delta'])])

class Canny(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Canny", 
            endpoint = "/canny", 
            var1="block", var1Name="threshold1", 
            var2="block", var2Name="threshold2",
            var3="none",
            var4="none"))
    def post(self):
        return image_process(request.files['image'].read(), improc.laplacian, [int(request.form['threshold1']), int(request.form['threshold2'])])

class FrontalFace(Resource):
    def get(self):
        return make_response(render_template(
            'methodPage.html', 
            methodName="Canny", 
            endpoint = "/frontal", 
            var1="block", var1Name="min_neighbours", 
            var2="block", var2Name="scale",
            var3="none",
            var4="none"))
    def post(self):
        return image_process(request.files['image'].read(), improc.laplacian, [int(request.form['min_neighbours']), float(request.form['scale'])])


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
api.add_resource(Average, "/average")
api.add_resource(GaussianBlur, "/gauss")
api.add_resource(Bilateral, "/bilateral")
api.add_resource(GlobalThresh, "/thresh/global")
api.add_resource(MeanThreshold, "/thresh/mean")
api.add_resource(GaussianThreshold, "/thresh/gauss")
api.add_resource(Sobel, "/sobel")
api.add_resource(Laplacian,"/laplacian")
api.add_resource(Canny, "/canny")
api.add_resource(FrontalFace, "/frontal")

if __name__ == "__main__":
    app.run(debug=True)
    