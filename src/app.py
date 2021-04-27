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
    """
        Decorator for function that converts image from bgr to rgb. Version for functions that returns 2 parameters

        :type function: function
        :param function: function to wrap

    """
    def wrapper(*args):
        image, error = function(*args)
        image, _ = imageprocessor.bgr2rgb(image)
        return image, error, None
    return wrapper


def bgr2rgb_3_params(function):
    """
        Decorator for function that converts image from bgr to rgb. Version for functions that returns 3 parameters

        :type function: function
        :param function: function to wrap

    """
    def wrapper(*args):
        image, error, values = function(*args)
        image, _ = imageprocessor.bgr2rgb(image)
        return image, error, values
    return wrapper  

def image_process(file, method, params = None):

    """
        Function that processes image with chosen opencv method.

        :type file: bytes
        :param function: bytes stream containing image data

        :type method: funtion
        :param function: function to process image with

        :rtype: flask.wrappers.Response
        :return: HTTP response containing json with process information - whether it succeeded, error message, processed image in string (status), extra data in string.
    """
    

    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, flags=1)

    if params:
        res = method(img, *params)
    else:
        res = method(img)
    
    if(res[1]):
        return {'success' : False, 'err' : res[1]}

    extraRes = res[-1]
    img = Image.fromarray(res[0].astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())

    if(extraRes is not None):
        return jsonify({'success' : True, 'status':str(img_base64), 'extra' : str(extraRes)})
    else:
        a = jsonify({'success' : True, 'status':str(img_base64)})
        print(type(a))
        return jsonify({'success' : True, 'status':str(img_base64)})

class HomePage(Resource):
    """
        API endpoint to load homepage
    """
    def get(self):
        """
            GET /

            :rtype: HTTP Response
            :return: renders and returns homepage
        """
        return make_response(render_template('index.html'))

class Grayscale(Resource):
    """
        API endpoint for grayscale method
    """
    def get(self):
        """
            GET /

            :rtype: HTTP Response
            :return: renders and returns grayscale page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Grayscale", 
            endpoint = "/gray", 
            var1="none", var1Name = "",
            var2="none", var2Name = "",
            var3="none", var3Name = "",
            var4="none", var4Name = "",))
    def post(self):
        """
            POST /grey

            :param request.files['image']: formData field containing image
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTML response
        """
        return image_process(request.files['image'].read(), imageprocessor.to_grayscale)

class Median(Resource):
    """
        API endpoint for median blur method
    """
    def get(self):
        """
            GET /

            :rtype: HTTP Response
            :return: renders and returns median page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Median", 
            endpoint = "/median", 
            var1="block", var1Name="ksize", 
            var2="none", var2Name = "",
            var3="none", var3Name = "",
            var4="none", var4Name = "",))
    def post(self):
        """
            POST /grey

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing ksize parameter
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), bgr2rgb_2_params(imageprocessor.median_blur), [int(request.form['ksize'])])

class Average(Resource):
    """
        API endpoint for average blur method
    """
    def get(self):
        """
            GET /average

            :rtype: HTTP Response
            :return: renders and returns average blur page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Average", 
            endpoint = "/average", 
            var1="block", var1Name="ksize_x", 
            var2="block", var2Name="ksize_y",
            var3="none", var3Name = "",
            var4="none", var4Name = "",))
    def post(self):
        """
            POST /average

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing ksize_x and ksize_y parameters
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), bgr2rgb_2_params(imageprocessor.average_blur), [int(request.form['ksize_x']), int(request.form['ksize_y'])])

class GaussianBlur(Resource):
    """
        API endpoint for gaussian blur method
    """
    def get(self):
        """
            GET /gauss

            :rtype: HTTP Response
            :return: renders and returns average page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Gaussian Blur", 
            endpoint = "/gauss", 
            var1="block", var1Name="ksize_x", 
            var2="block", var2Name="ksize_y",
            var3="none", var3Name = "",
            var4="none", var4Name = "",))
    def post(self):
        """
            POST /gauss

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing ksize_x and ksize_y parameters
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), bgr2rgb_2_params(imageprocessor.gaussian_blur), [int(request.form['ksize_x']), int(request.form['ksize_y'])])

class Bilateral(Resource):
    """
        API endpoint for bilateral method
    """
    def get(self):
        """
            GET /bilateral

            :rtype: HTTP Response
            :return: renders and returns bilateral page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Bilateral filter", 
            endpoint = "/bilateral", 
            var1="block",  var1Name="d", 
            var2="block", var2Name="sigma",
            var3="none",
            var4="none"))
    def post(self):
        """
            POST /gauss

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing d and sigma parameters
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), bgr2rgb_2_params(imageprocessor.bilateral_filter), [int(request.form['d']), float(request.form['sigma'])])

class GlobalThresh(Resource):
    """
        API endpoint for global threshold method
    """
    def get(self):
        """
            GET /thresh/global

            :rtype: HTTP Response
            :return: renders and returns global threshold page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Global threshold", 
            endpoint = "/thresh/global", 
            var1="block", var1Name="threshold", 
            var2="block", var2Name="value",
            var3="none",
            var4="none"))
    def post(self):
        """
            POST /thresh/global

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing threshold and value parameters
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), imageprocessor.global_threshold, [int(request.form['threshold']), int(request.form['value'])])

class MeanThreshold(Resource):
    """
        API endpoint for mean threshold method
    """
    def get(self):
        """
            GET /thresh/mean

            :rtype: HTTP Response
            :return: renders and returns mean threshold page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Mean threshold", 
            endpoint = "/thresh/mean", 
            var1="block", var1Name="blocksize", 
            var2="block", var2Name="c",
            var3="block", var3Name="value",
            var4="none"))
    def post(self):
        """
            POST /thresh/mean

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing blocksize, c and value parameters
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), imageprocessor.mean_threshold, [int(request.form['blocksize']), float(request.form['c']), int(request.form['value'])])

class GaussianThreshold(Resource):
    """
        API endpoint for gaussian threshold method
    """
    def get(self):
        """
            GET /thresh/gauss

            :rtype: HTTP Response
            :return: renders and returns gaussian threshold page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Gaussian threshold", 
            endpoint = "/thresh/gauss", 
            var1="block", var1Name="blocksize", 
            var2="block", var2Name="c",
            var3="block", var3Name="value",
            var4="none"))
    def post(self):
        """
            POST /thresh/gauss

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing blocksize, c and value parameters
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), imageprocessor.gaussian_threshold, [int(request.form['blocksize']), float(request.form['c']), int(request.form['value'])])

class Sobel(Resource):
    """
        API endpoint for sobel method
    """
    def get(self):
        """
            GET /sobel

            :rtype: HTTP Response
            :return: renders and returns sobel page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Sobel", 
            endpoint = "/sobel", 
            var1="block", var1Name="dx", 
            var2="block", var2Name="dy",
            var3="block", var3Name="ksize",
            var4="block", var4Name="delta"))
    def post(self):
        """
            POST /sobel

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing dx, dy, ksize and delta parameters
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), imageprocessor.sobel, [int(request.form['dx']), int(request.form['dy']), int(request.form['ksize']), float(request.form['delta'])])

class Laplacian(Resource):
    """
        API endpoint for laplacian method
    """
    def get(self):
        """
            GET /laplacian

            :rtype: HTTP Response
            :return: renders and returns laplacian page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Laplacian", 
            endpoint = "/laplacian", 
            var1="block", var1Name="ksize", 
            var2="block", var2Name="delta",
            var3="none",
            var4="none"))
    def post(self):
        """
            POST /laplacian

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing ksize and delta parameters
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), imageprocessor.laplacian, [int(request.form['ksize']), float(request.form['delta'])])

class Canny(Resource):
    """
        API endpoint for canny method
    """
    def get(self):
        """
            GET /canny

            :rtype: HTTP Response
            :return: renders and returns canny page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Canny", 
            endpoint = "/canny", 
            var1="block", var1Name="threshold1", 
            var2="block", var2Name="threshold2",
            var3="none",
            var4="none"))
    def post(self):
        """
            POST /canny

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing threshold and threshold2 parameters
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), imageprocessor.canny_edge_detection, [int(request.form['threshold1']), int(request.form['threshold2'])])

class FrontalFace(Resource):
    """
        API endpoint for frontal method
    """
    def get(self):
        """
            GET /frontal

            :rtype: HTTP Response
            :return: renders and returns frontal face page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Frontal face detection", 
            endpoint = "/frontal", 
            var1="block", var1Name="min_neighbours", 
            var2="block", var2Name="scale",
            var3="none",
            var4="none"))
    def post(self):
        """
            POST /frontal

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing min_neighbours and scale parameters
            :rtype: flask.wrappers.Response
            :return: returns processed image along with detected face coordinates wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), bgr2rgb_3_params(imageprocessor.haar_frontal_face_detection), [int(request.form['min_neighbours']), float(request.form['scale'])])

class RotationNaive(Resource):
    """
        API endpoint for naive rotation method
    """
    def get(self):
        """
            GET /rotation/naive

            :rtype: HTTP Response
            :return: renders and returns naive rotation page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Naive rotation", 
            endpoint = "/rotation/naive", 
            var1="block", var1Name="angle", 
            var2="none",
            var3="none",
            var4="none"))
    def post(self):
        """
            POST /rotation/naive

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing angle parameter
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), bgr2rgb_2_params(imageprocessor.naive_rotate), [int(request.form['angle'])])

class Rotation(Resource):
    """
        API endpoint for rotation method
    """
    def get(self):
        """
            GET /rotation

            :rtype: HTTP Response
            :return: renders and returns rotation page
        """
        return make_response(render_template(
            'methodPage.html', 
            methodName="Rotation", 
            endpoint = "/rotation", 
            var1="block", var1Name="angle", 
            var2="none",
            var3="none",
            var4="none"))
    def post(self):
        """
            POST /rotation

            :param request.files['image']: formData field containing image
            :param request.form: formData fields containing angle parameter
            :rtype: flask.wrappers.Response
            :return: returns processed image wrapped in HTTP response
        """
        return image_process(request.files['image'].read(), bgr2rgb_2_params(imageprocessor.rotate), [int(request.form['angle'])])


@app.after_request
def after_request(response):
    """
        Method handling CORS mechanism to allow data flow
    """
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

api.add_resource(HomePage, "/")
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
api.add_resource(RotationNaive, "/rotation/naive")
api.add_resource(Rotation, "/rotation")

if __name__ == "__main__":
    app.run(debug=True)
    