from flask import Flask, render_template , request , jsonify, make_response
from flask_restful import Resource, Api
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64


app = Flask(__name__)
api = Api(app)


class HomePage(Resource):
    def get(self):
        return make_response(render_template('index.html'))
        
class LoadImage(Resource):
    def post(self):
        print("YEAH", file = sys.stderr)
        file = request.files['image'].read()
        npimg = np.fromstring(file, np.uint8)
        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        
        img = Image.fromarray(img.astype("uint8"))
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        return jsonify({'status':str(img_base64)})


class Test(Resource):
    def post(self):
	    print("log: got at test" , file=sys.stderr)
	    return jsonify({'status':'succces'})


@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


api.add_resource(HomePage, "/")
api.add_resource(LoadImage, "/image")
api.add_resource(Test, "/test")


if __name__ == "__main__":
    app.run(debug=True)